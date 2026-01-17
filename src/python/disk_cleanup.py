#!/usr/bin/env python3
"""
OFC AI Training - Disk Cleanup Script
学習中のディスク容量管理

- 古いチェックポイントの削除（最新2つ + マイルストーンを保持）
- 古いTensorBoardログの削除
- 定期実行で容量不足を防止

使用方法:
    python disk_cleanup.py                    # 1回実行
    python disk_cleanup.py --watch            # 10分ごとに監視
    python disk_cleanup.py --dry-run          # 削除せずに確認
"""

import os
import glob
import time
import argparse
import shutil
from datetime import datetime


def get_size_mb(path: str) -> float:
    """ファイル/ディレクトリのサイズ(MB)を取得"""
    if os.path.isfile(path):
        return os.path.getsize(path) / (1024 * 1024)
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            try:
                total += os.path.getsize(fp)
            except:
                pass
    return total / (1024 * 1024)


def cleanup_checkpoints(model_dir: str = "models", dry_run: bool = False) -> dict:
    """
    古いチェックポイントを削除

    保持するもの:
    - 最新2つのチェックポイント
    - マイルストーン (1M, 5M, 10M, 15M, 20M)
    - *_final.zip
    """
    milestones = {1_000_000, 5_000_000, 10_000_000, 15_000_000, 20_000_000}
    stats = {'deleted': 0, 'kept': 0, 'freed_mb': 0}

    # 各モデルプレフィックスごとに処理
    prefixes = ['p8_selfplay', 'aggressive', 'teacher', 'p7_parallel']

    for prefix in prefixes:
        pattern = os.path.join(model_dir, f"{prefix}_*.zip")
        checkpoints = sorted(glob.glob(pattern))

        if len(checkpoints) <= 2:
            stats['kept'] += len(checkpoints)
            continue

        # 最新2つを保持
        keep_latest = set(checkpoints[-2:])

        for cp in checkpoints:
            basename = os.path.basename(cp)

            # finalは保持
            if '_final' in basename:
                stats['kept'] += 1
                continue

            # マイルストーン抽出
            try:
                step = int(basename.split('_')[-1].replace('.zip', ''))
                if step in milestones:
                    stats['kept'] += 1
                    continue
            except:
                pass

            # 最新2つは保持
            if cp in keep_latest:
                stats['kept'] += 1
                continue

            # 削除対象
            size_mb = get_size_mb(cp)
            if dry_run:
                print(f"[DRY-RUN] Would delete: {cp} ({size_mb:.1f}MB)")
            else:
                try:
                    os.remove(cp)
                    print(f"[DELETED] {cp} ({size_mb:.1f}MB)")
                except Exception as e:
                    print(f"[ERROR] Failed to delete {cp}: {e}")
                    continue

            stats['deleted'] += 1
            stats['freed_mb'] += size_mb

    return stats


def cleanup_tensorboard_logs(log_dir: str = "logs", max_runs: int = 3, dry_run: bool = False) -> dict:
    """
    古いTensorBoardログを削除

    各サブディレクトリで最新N個のrunを保持
    """
    stats = {'deleted': 0, 'kept': 0, 'freed_mb': 0}

    if not os.path.exists(log_dir):
        return stats

    # 各学習タイプのログディレクトリ
    for subdir in os.listdir(log_dir):
        subdir_path = os.path.join(log_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue

        # MaskablePPO_* ディレクトリを取得
        runs = sorted(glob.glob(os.path.join(subdir_path, "MaskablePPO_*")),
                      key=lambda x: os.path.getmtime(x) if os.path.exists(x) else 0)

        if len(runs) <= max_runs:
            stats['kept'] += len(runs)
            continue

        # 古いrunを削除
        for run in runs[:-max_runs]:
            size_mb = get_size_mb(run)
            if dry_run:
                print(f"[DRY-RUN] Would delete log: {run} ({size_mb:.1f}MB)")
            else:
                try:
                    shutil.rmtree(run)
                    print(f"[DELETED] Log: {run} ({size_mb:.1f}MB)")
                except Exception as e:
                    print(f"[ERROR] Failed to delete {run}: {e}")
                    continue

            stats['deleted'] += 1
            stats['freed_mb'] += size_mb

        stats['kept'] += max_runs

    return stats


def get_disk_usage() -> dict:
    """ディスク使用量を取得"""
    import subprocess
    result = subprocess.run(['df', '-h', '/home'], capture_output=True, text=True)
    lines = result.stdout.strip().split('\n')
    if len(lines) >= 2:
        parts = lines[1].split()
        return {
            'total': parts[1],
            'used': parts[2],
            'available': parts[3],
            'use_percent': parts[4]
        }
    return {}


def run_cleanup(dry_run: bool = False):
    """クリーンアップを実行"""
    print(f"\n{'='*50}")
    print(f"  OFC AI Disk Cleanup - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*50}")

    # ディスク使用量（前）
    disk_before = get_disk_usage()
    print(f"\nDisk usage before: {disk_before.get('used', '?')} / {disk_before.get('total', '?')} ({disk_before.get('use_percent', '?')})")

    # チェックポイントクリーンアップ
    print("\n--- Checkpoint Cleanup ---")
    cp_stats = cleanup_checkpoints(dry_run=dry_run)

    # TensorBoardログクリーンアップ
    print("\n--- TensorBoard Log Cleanup ---")
    tb_stats = cleanup_tensorboard_logs(dry_run=dry_run)

    # サマリー
    total_freed = cp_stats['freed_mb'] + tb_stats['freed_mb']
    total_deleted = cp_stats['deleted'] + tb_stats['deleted']

    print(f"\n--- Summary ---")
    print(f"Checkpoints: {cp_stats['deleted']} deleted, {cp_stats['kept']} kept, {cp_stats['freed_mb']:.1f}MB freed")
    print(f"TB Logs: {tb_stats['deleted']} deleted, {tb_stats['kept']} kept, {tb_stats['freed_mb']:.1f}MB freed")
    print(f"Total: {total_deleted} items deleted, {total_freed:.1f}MB freed")

    if not dry_run:
        disk_after = get_disk_usage()
        print(f"\nDisk usage after: {disk_after.get('used', '?')} / {disk_after.get('total', '?')} ({disk_after.get('use_percent', '?')})")


def watch_mode(interval_minutes: int = 10, dry_run: bool = False):
    """定期実行モード"""
    print(f"Starting disk cleanup watch mode (interval: {interval_minutes} minutes)")
    print("Press Ctrl+C to stop")

    while True:
        run_cleanup(dry_run=dry_run)
        print(f"\nNext cleanup in {interval_minutes} minutes...")
        time.sleep(interval_minutes * 60)


def main():
    parser = argparse.ArgumentParser(description="OFC AI Disk Cleanup")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be deleted without actually deleting")
    parser.add_argument("--watch", action="store_true", help="Run continuously every 10 minutes")
    parser.add_argument("--interval", type=int, default=10, help="Watch interval in minutes (default: 10)")
    args = parser.parse_args()

    if args.watch:
        watch_mode(interval_minutes=args.interval, dry_run=args.dry_run)
    else:
        run_cleanup(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
