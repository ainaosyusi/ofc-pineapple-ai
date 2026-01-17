"""
OFC Pineapple AI - Cloud Storage Abstraction Layer
AWS S3 / GCP GCS の統一インターフェース

使用方法:
    from cloud_storage import init_cloud_storage

    # 自動検出（環境変数から）
    storage = init_cloud_storage()

    # 明示的に指定
    storage = init_cloud_storage(provider="gcs", bucket="my-bucket")
"""

import os
from typing import Optional, List
from abc import ABC, abstractmethod


class CloudStorageBase(ABC):
    """クラウドストレージの抽象基底クラス"""

    @property
    @abstractmethod
    def enabled(self) -> bool:
        pass

    @abstractmethod
    def upload_checkpoint(
        self,
        local_path: str,
        step: Optional[int] = None,
        metadata: Optional[dict] = None
    ) -> Optional[str]:
        pass

    @abstractmethod
    def download_latest_checkpoint(
        self,
        local_dir: str = "models"
    ) -> Optional[str]:
        pass

    @abstractmethod
    def upload_logs(self, log_dir: str = "logs") -> List[str]:
        pass

    @abstractmethod
    def list_checkpoints(self, limit: int = 10) -> List[dict]:
        pass


class CloudStorageManager:
    """
    統一クラウドストレージマネージャ
    S3/GCSの両方をサポート
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        bucket_name: Optional[str] = None,
        **kwargs
    ):
        """
        Args:
            provider: "s3", "gcs", または None (自動検出)
            bucket_name: バケット名
            **kwargs: 各プロバイダ固有のオプション
        """
        self.provider = self._detect_provider(provider)
        self._manager = None
        self._enabled = False

        if self.provider == "s3":
            self._init_s3(bucket_name, **kwargs)
        elif self.provider == "gcs":
            self._init_gcs(bucket_name, **kwargs)
        else:
            print("[CloudStorage] No cloud provider configured. Storage disabled.")

    def _detect_provider(self, provider: Optional[str]) -> Optional[str]:
        """プロバイダを自動検出"""
        if provider:
            return provider.lower()

        # 環境変数から自動検出
        if os.getenv("CLOUD_PROVIDER"):
            return os.getenv("CLOUD_PROVIDER").lower()
        if os.getenv("GCS_BUCKET") or os.getenv("GCP_PROJECT"):
            return "gcs"
        if os.getenv("S3_BUCKET") or os.getenv("AWS_REGION"):
            return "s3"
        return None

    def _init_s3(self, bucket_name: Optional[str], **kwargs):
        """S3マネージャを初期化"""
        try:
            from s3_utils import S3Manager
            region = kwargs.get("region") or os.getenv("AWS_REGION", "ap-northeast-1")
            self._manager = S3Manager(
                bucket_name=bucket_name,
                region=region,
                prefix=kwargs.get("prefix", "ofc-training")
            )
            self._enabled = self._manager.enabled
            if self._enabled:
                print(f"[CloudStorage] Using S3: {bucket_name}")
        except ImportError as e:
            print(f"[CloudStorage] S3 initialization failed: {e}")

    def _init_gcs(self, bucket_name: Optional[str], **kwargs):
        """GCSマネージャを初期化"""
        try:
            from gcs_utils import GCSManager
            project = kwargs.get("project") or os.getenv("GCP_PROJECT")
            self._manager = GCSManager(
                bucket_name=bucket_name,
                project=project,
                prefix=kwargs.get("prefix", "ofc-training")
            )
            self._enabled = self._manager.enabled
            if self._enabled:
                print(f"[CloudStorage] Using GCS: {bucket_name}")
        except ImportError as e:
            print(f"[CloudStorage] GCS initialization failed: {e}")

    @property
    def enabled(self) -> bool:
        return self._enabled

    def upload_checkpoint(
        self,
        local_path: str,
        step: Optional[int] = None,
        metadata: Optional[dict] = None
    ) -> Optional[str]:
        """チェックポイントをアップロード"""
        if not self._enabled or not self._manager:
            return None
        return self._manager.upload_checkpoint(local_path, step, metadata)

    def download_latest_checkpoint(
        self,
        local_dir: str = "models"
    ) -> Optional[str]:
        """最新のチェックポイントをダウンロード"""
        if not self._enabled or not self._manager:
            return None
        return self._manager.download_latest_checkpoint(local_dir)

    def upload_logs(self, log_dir: str = "logs") -> List[str]:
        """ログをアップロード"""
        if not self._enabled or not self._manager:
            return []
        return self._manager.upload_logs(log_dir)

    def list_checkpoints(self, limit: int = 10) -> List[dict]:
        """チェックポイント一覧を取得"""
        if not self._enabled or not self._manager:
            return []
        return self._manager.list_checkpoints(limit)


# === Convenience Functions ===

_cloud_storage: Optional[CloudStorageManager] = None


def init_cloud_storage(
    provider: Optional[str] = None,
    bucket_name: Optional[str] = None,
    **kwargs
) -> CloudStorageManager:
    """グローバルクラウドストレージマネージャを初期化"""
    global _cloud_storage
    _cloud_storage = CloudStorageManager(
        provider=provider,
        bucket_name=bucket_name,
        **kwargs
    )
    return _cloud_storage


def get_cloud_storage() -> Optional[CloudStorageManager]:
    """グローバルクラウドストレージマネージャを取得"""
    global _cloud_storage
    if _cloud_storage is None:
        _cloud_storage = CloudStorageManager()
    return _cloud_storage


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test cloud storage abstraction")
    parser.add_argument("--provider", type=str, choices=["s3", "gcs"], help="Cloud provider")
    parser.add_argument("--bucket", type=str, help="Bucket name")
    parser.add_argument("--list", action="store_true", help="List checkpoints")
    args = parser.parse_args()

    storage = CloudStorageManager(provider=args.provider, bucket_name=args.bucket)

    print(f"Provider: {storage.provider}")
    print(f"Enabled: {storage.enabled}")

    if args.list and storage.enabled:
        checkpoints = storage.list_checkpoints()
        print(f"\nCheckpoints ({len(checkpoints)}):")
        for cp in checkpoints:
            print(f"  - {cp['key']} ({cp['size']} bytes)")
