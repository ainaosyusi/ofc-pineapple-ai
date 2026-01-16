import os
import re
import glob
from typing import List

def cleanup_checkpoints(directory: str, keep_last: int = 5, keep_milestones: int = 1000000):
    """
    モデルディレクトリ内の古いチェックポイントを削除する
    - 最新の keep_last 個は保持
    - keep_milestones ステップごとの区切りは保持
    """
    print(f"Cleaning up {directory}...")
    
    # .zip ファイルをリストアップ
    pattern = os.path.join(directory, "*.zip")
    files = glob.glob(pattern)
    
    if not files:
        print("No checkpoint files found.")
        return

    # ファイル名からステップ数を抽出してソート
    checkpoint_files = []
    for f in files:
        # e.g. ofc_phase5_3max_20260116_133943_4000_steps.zip
        match = re.search(r'(\d+)_steps', os.path.basename(f))
        if match:
            step = int(match.group(1))
            checkpoint_files.append((step, f))
        elif f.endswith("final.zip"):
            checkpoint_files.append((float('inf'), f)) # Always keep final

    checkpoint_files.sort()

    # 削除対象の特定
    to_delete = []
    
    # 最後の5つは保持リスト
    keep_indices = set()
    num_files = len(checkpoint_files)
    
    # 最新の count 個を保持
    for i in range(max(0, num_files - keep_last), num_files):
        keep_indices.add(i)
        
    # ミルストーンを保持 (e.g. 1M, 2M, ...)
    for i, (step, f) in enumerate(checkpoint_files):
        if step != float('inf') and step % keep_milestones == 0:
            keep_indices.add(i)
        if step == float('inf'):
            keep_indices.add(i)

    # 削除実行
    deleted_count = 0
    freed_space = 0
    for i, (step, f) in enumerate(checkpoint_files):
        if i not in keep_indices:
            size = os.path.getsize(f)
            try:
                os.remove(f)
                deleted_count += 1
                freed_space += size
            except Exception as e:
                print(f"Error deleting {f}: {e}")

    print(f"Done. Deleted {deleted_count} files, freed {freed_space / 1024 / 1024:.2f} MB.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Cleanup old checkpoint files")
    parser.add_argument("directory", help="Directory containing .zip checkpoints")
    parser.add_argument("--keep", type=int, default=5, help="Number of latest checkpoints to keep")
    parser.add_argument("--milestone", type=int, default=1000000, help="Keep checkpoints at these step intervals")
    
    args = parser.parse_args()
    
    if os.path.isdir(args.directory):
        cleanup_checkpoints(args.directory, args.keep, args.milestone)
    else:
        print(f"Error: {args.directory} is not a directory.")
