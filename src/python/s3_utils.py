"""
OFC Pineapple AI - S3 Utilities
チェックポイントとログのS3管理
"""

import os
import boto3
from botocore.exceptions import ClientError
from datetime import datetime
from typing import Optional, List
import glob


class S3Manager:
    """
    S3ストレージ管理クラス
    チェックポイントとログのアップロード/ダウンロード
    """
    
    def __init__(
        self,
        bucket_name: Optional[str] = None,
        region: Optional[str] = None,
        prefix: str = "ofc-training"
    ):
        """
        Args:
            bucket_name: S3バケット名
            region: AWSリージョン
            prefix: S3キープレフィックス
        """
        self.bucket_name = bucket_name or os.getenv("S3_BUCKET")
        self.region = region or os.getenv("AWS_REGION", "ap-northeast-1")
        self.prefix = prefix
        
        self.enabled = bool(self.bucket_name)
        
        if self.enabled:
            self.s3 = boto3.client("s3", region_name=self.region)
            print(f"[S3] Connected to bucket: {self.bucket_name}")
        else:
            self.s3 = None
            print("[S3] No bucket configured. S3 storage disabled.")
    
    def upload_checkpoint(
        self,
        local_path: str,
        step: Optional[int] = None,
        metadata: Optional[dict] = None
    ) -> Optional[str]:
        """
        チェックポイントをS3にアップロード
        
        Args:
            local_path: ローカルファイルパス
            step: 学習ステップ数（ファイル名に使用）
            metadata: 追加メタデータ
        
        Returns:
            S3 URI (s3://bucket/key) or None
        """
        if not self.enabled:
            return None
        
        filename = os.path.basename(local_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if step is not None:
            s3_key = f"{self.prefix}/checkpoints/step_{step:08d}_{timestamp}_{filename}"
        else:
            s3_key = f"{self.prefix}/checkpoints/{timestamp}_{filename}"
        
        try:
            extra_args = {}
            if metadata:
                extra_args["Metadata"] = {k: str(v) for k, v in metadata.items()}
            
            self.s3.upload_file(local_path, self.bucket_name, s3_key, ExtraArgs=extra_args)
            
            s3_uri = f"s3://{self.bucket_name}/{s3_key}"
            print(f"[S3] Uploaded checkpoint: {s3_uri}")
            return s3_uri
            
        except ClientError as e:
            print(f"[S3] Upload failed: {e}")
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
            # チェックポイント一覧を取得
            response = self.s3.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=f"{self.prefix}/checkpoints/"
            )
            
            if "Contents" not in response or len(response["Contents"]) == 0:
                print("[S3] No checkpoints found")
                return None
            
            # 最新のファイルを取得（キー名でソート）
            latest = sorted(response["Contents"], key=lambda x: x["Key"])[-1]
            s3_key = latest["Key"]
            
            # ダウンロード
            os.makedirs(local_dir, exist_ok=True)
            filename = os.path.basename(s3_key)
            local_path = os.path.join(local_dir, filename)
            
            self.s3.download_file(self.bucket_name, s3_key, local_path)
            
            print(f"[S3] Downloaded checkpoint: {local_path}")
            return local_path
            
        except ClientError as e:
            print(f"[S3] Download failed: {e}")
            return None
    
    def upload_logs(self, log_dir: str = "logs") -> List[str]:
        """
        ログディレクトリをS3にアップロード
        
        Args:
            log_dir: ログディレクトリパス
        
        Returns:
            アップロードしたS3 URIのリスト
        """
        if not self.enabled:
            return []
        
        uploaded = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for log_file in glob.glob(os.path.join(log_dir, "*.log")):
            filename = os.path.basename(log_file)
            s3_key = f"{self.prefix}/logs/{timestamp}_{filename}"
            
            try:
                self.s3.upload_file(log_file, self.bucket_name, s3_key)
                s3_uri = f"s3://{self.bucket_name}/{s3_key}"
                uploaded.append(s3_uri)
                print(f"[S3] Uploaded log: {s3_uri}")
            except ClientError as e:
                print(f"[S3] Failed to upload {filename}: {e}")
        
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
            response = self.s3.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=f"{self.prefix}/checkpoints/"
            )
            
            if "Contents" not in response:
                return []
            
            checkpoints = []
            for obj in sorted(response["Contents"], key=lambda x: x["Key"], reverse=True)[:limit]:
                checkpoints.append({
                    "key": obj["Key"],
                    "size": obj["Size"],
                    "last_modified": obj["LastModified"].isoformat(),
                    "uri": f"s3://{self.bucket_name}/{obj['Key']}"
                })
            
            return checkpoints
            
        except ClientError as e:
            print(f"[S3] List failed: {e}")
            return []


# === Convenience Functions ===

_s3_manager: Optional[S3Manager] = None

def init_s3(
    bucket_name: Optional[str] = None,
    region: Optional[str] = None
) -> S3Manager:
    """グローバルS3マネージャを初期化"""
    global _s3_manager
    _s3_manager = S3Manager(bucket_name=bucket_name, region=region)
    return _s3_manager

def get_s3() -> Optional[S3Manager]:
    """グローバルS3マネージャを取得"""
    global _s3_manager
    if _s3_manager is None:
        _s3_manager = S3Manager()
    return _s3_manager


if __name__ == "__main__":
    # テスト
    import argparse
    
    parser = argparse.ArgumentParser(description="Test S3 utilities")
    parser.add_argument("--bucket", type=str, help="S3 bucket name")
    parser.add_argument("--list", action="store_true", help="List checkpoints")
    args = parser.parse_args()
    
    s3 = S3Manager(bucket_name=args.bucket)
    
    if args.list and s3.enabled:
        checkpoints = s3.list_checkpoints()
        print(f"\nCheckpoints ({len(checkpoints)}):")
        for cp in checkpoints:
            print(f"  - {cp['key']} ({cp['size']} bytes)")
