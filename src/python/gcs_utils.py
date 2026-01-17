"""
OFC Pineapple AI - GCS (Google Cloud Storage) Utilities
チェックポイントとログのGCS管理

S3Managerと同じインターフェースを提供し、AWS/GCP両対応を実現
"""

import os
import glob
from datetime import datetime
from typing import Optional, List

try:
    from google.cloud import storage
    from google.api_core import exceptions as gcp_exceptions
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False
    print("[GCS] google-cloud-storage not installed. Run: pip install google-cloud-storage")


class GCSManager:
    """
    GCSストレージ管理クラス
    チェックポイントとログのアップロード/ダウンロード
    S3Managerと互換性のあるインターフェースを提供
    """

    def __init__(
        self,
        bucket_name: Optional[str] = None,
        project: Optional[str] = None,
        prefix: str = "ofc-training"
    ):
        """
        Args:
            bucket_name: GCSバケット名
            project: GCPプロジェクトID
            prefix: GCSキープレフィックス
        """
        self.bucket_name = bucket_name or os.getenv("GCS_BUCKET")
        self.project = project or os.getenv("GCP_PROJECT")
        self.prefix = prefix

        self.enabled = bool(self.bucket_name) and GCS_AVAILABLE

        if self.enabled:
            try:
                self.client = storage.Client(project=self.project)
                self.bucket = self.client.bucket(self.bucket_name)
                print(f"[GCS] Connected to bucket: {self.bucket_name}")
            except Exception as e:
                print(f"[GCS] Failed to connect: {e}")
                self.enabled = False
                self.client = None
                self.bucket = None
        else:
            self.client = None
            self.bucket = None
            if not GCS_AVAILABLE:
                print("[GCS] google-cloud-storage library not available.")
            elif not self.bucket_name:
                print("[GCS] No bucket configured. GCS storage disabled.")

    def upload_checkpoint(
        self,
        local_path: str,
        step: Optional[int] = None,
        metadata: Optional[dict] = None
    ) -> Optional[str]:
        """
        チェックポイントをGCSにアップロード

        Args:
            local_path: ローカルファイルパス
            step: 学習ステップ数（ファイル名に使用）
            metadata: 追加メタデータ

        Returns:
            GCS URI (gs://bucket/key) or None
        """
        if not self.enabled:
            return None

        filename = os.path.basename(local_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if step is not None:
            gcs_key = f"{self.prefix}/checkpoints/step_{step:08d}_{timestamp}_{filename}"
        else:
            gcs_key = f"{self.prefix}/checkpoints/{timestamp}_{filename}"

        try:
            blob = self.bucket.blob(gcs_key)

            if metadata:
                blob.metadata = {k: str(v) for k, v in metadata.items()}

            blob.upload_from_filename(local_path)

            gcs_uri = f"gs://{self.bucket_name}/{gcs_key}"
            print(f"[GCS] Uploaded checkpoint: {gcs_uri}")
            return gcs_uri

        except Exception as e:
            print(f"[GCS] Upload failed: {e}")
            return None

    def download_latest_checkpoint(
        self,
        local_dir: str = "models"
    ) -> Optional[str]:
        """
        最新のチェックポイントをダウンロード

        Args:
            local_dir: ダウンロード先ディレクトリ

        Returns:
            ダウンロードしたファイルのローカルパス or None
        """
        if not self.enabled:
            return None

        try:
            blobs = list(self.client.list_blobs(
                self.bucket_name,
                prefix=f"{self.prefix}/checkpoints/"
            ))

            if not blobs:
                print("[GCS] No checkpoints found")
                return None

            latest = sorted(blobs, key=lambda x: x.name)[-1]

            os.makedirs(local_dir, exist_ok=True)
            filename = os.path.basename(latest.name)
            local_path = os.path.join(local_dir, filename)

            latest.download_to_filename(local_path)

            print(f"[GCS] Downloaded checkpoint: {local_path}")
            return local_path

        except Exception as e:
            print(f"[GCS] Download failed: {e}")
            return None

    def upload_logs(self, log_dir: str = "logs") -> List[str]:
        """
        ログディレクトリをGCSにアップロード

        Args:
            log_dir: ログディレクトリパス

        Returns:
            アップロードしたGCS URIのリスト
        """
        if not self.enabled:
            return []

        uploaded = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for log_file in glob.glob(os.path.join(log_dir, "*.log")):
            filename = os.path.basename(log_file)
            gcs_key = f"{self.prefix}/logs/{timestamp}_{filename}"

            try:
                blob = self.bucket.blob(gcs_key)
                blob.upload_from_filename(log_file)
                gcs_uri = f"gs://{self.bucket_name}/{gcs_key}"
                uploaded.append(gcs_uri)
                print(f"[GCS] Uploaded log: {gcs_uri}")
            except Exception as e:
                print(f"[GCS] Failed to upload {filename}: {e}")

        return uploaded

    def list_checkpoints(self, limit: int = 10) -> List[dict]:
        """
        チェックポイント一覧を取得

        Args:
            limit: 最大取得数

        Returns:
            チェックポイント情報のリスト
        """
        if not self.enabled:
            return []

        try:
            blobs = list(self.client.list_blobs(
                self.bucket_name,
                prefix=f"{self.prefix}/checkpoints/"
            ))

            checkpoints = []
            for blob in sorted(blobs, key=lambda x: x.name, reverse=True)[:limit]:
                checkpoints.append({
                    "key": blob.name,
                    "size": blob.size,
                    "last_modified": blob.updated.isoformat() if blob.updated else "",
                    "uri": f"gs://{self.bucket_name}/{blob.name}"
                })

            return checkpoints

        except Exception as e:
            print(f"[GCS] List failed: {e}")
            return []


# === Convenience Functions ===

_gcs_manager: Optional[GCSManager] = None

def init_gcs(
    bucket_name: Optional[str] = None,
    project: Optional[str] = None
) -> GCSManager:
    """グローバルGCSマネージャを初期化"""
    global _gcs_manager
    _gcs_manager = GCSManager(bucket_name=bucket_name, project=project)
    return _gcs_manager

def get_gcs() -> Optional[GCSManager]:
    """グローバルGCSマネージャを取得"""
    global _gcs_manager
    if _gcs_manager is None:
        _gcs_manager = GCSManager()
    return _gcs_manager


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test GCS utilities")
    parser.add_argument("--bucket", type=str, help="GCS bucket name")
    parser.add_argument("--list", action="store_true", help="List checkpoints")
    args = parser.parse_args()

    gcs = GCSManager(bucket_name=args.bucket)

    if args.list and gcs.enabled:
        checkpoints = gcs.list_checkpoints()
        print(f"\nCheckpoints ({len(checkpoints)}):")
        for cp in checkpoints:
            print(f"  - {cp['key']} ({cp['size']} bytes)")
