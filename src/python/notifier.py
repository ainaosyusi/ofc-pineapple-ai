"""
OFC Pineapple AI - Training Notifier
Discord/Slack Webhook通知システム
"""

import os
import json
import requests
from datetime import datetime
from typing import Optional, Dict, Any


class TrainingNotifier:
    """
    学習進捗通知クラス
    Discord/Slack Webhook対応
    """
    
    def __init__(
        self,
        discord_webhook: Optional[str] = None,
        slack_webhook: Optional[str] = None,
        project_name: str = "OFC Pineapple AI"
    ):
        """
        Args:
            discord_webhook: Discord Webhook URL
            slack_webhook: Slack Incoming Webhook URL
            project_name: プロジェクト名（通知に表示）
        """
        self.discord_webhook = discord_webhook or os.getenv("DISCORD_WEBHOOK_URL")
        self.slack_webhook = slack_webhook or os.getenv("SLACK_WEBHOOK_URL")
        self.project_name = project_name
        self.enabled = bool(self.discord_webhook or self.slack_webhook)
        
        if not self.enabled:
            print("[Notifier] No webhook configured. Notifications disabled.")
    
    def send_start(self, config: Dict[str, Any]):
        """学習開始通知"""
        message = f"🚀 **{self.project_name}** - Training Started\n\n"
        message += f"```\n"
        message += f"Timesteps: {config.get('timesteps', 'N/A'):,}\n"
        message += f"Opponent Update: {config.get('opponent_update', 'N/A'):,}\n"
        message += f"Learning Rate: {config.get('lr', 'N/A')}\n"
        message += f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        message += f"```"
        
        self._send(message, color=0x00ff00)  # Green
    
    def send_progress(
        self,
        step: int,
        total_steps: int,
        metrics: Dict[str, Any]
    ):
        """定期進捗レポート"""
        progress = step / total_steps * 100
        
        message = f"📊 **{self.project_name}** - Progress Update\n\n"
        message += f"**Progress:** {progress:.1f}% ({step:,} / {total_steps:,})\n"
        message += f"```\n"
        message += f"Games: {metrics.get('games', 0):,}\n"
        message += f"Win Rate: {metrics.get('win_rate', 0):.1f}%\n"
        message += f"Foul Rate: {metrics.get('foul_rate', 0):.1f}%\n"
        message += f"Mean Score: {metrics.get('mean_score', 0):.2f}\n"
        message += f"FPS: {metrics.get('fps', 0):.0f}\n"
        message += f"```"
        
        self._send(message, color=0x0099ff)  # Blue
    
    def send_checkpoint(self, checkpoint_path: str, step: int):
        """チェックポイント保存通知"""
        message = f"💾 **{self.project_name}** - Checkpoint Saved\n\n"
        message += f"Step: {step:,}\n"
        message += f"Path: `{checkpoint_path}`"
        
        self._send(message, color=0xffff00)  # Yellow
    
    def send_error(self, error: str, traceback: Optional[str] = None):
        """エラー通知"""
        message = f"❌ **{self.project_name}** - Training Error\n\n"
        message += f"**Error:** {error}\n"
        if traceback:
            message += f"```\n{traceback[:500]}...\n```"
        
        self._send(message, color=0xff0000)  # Red
    
    def send_complete(self, summary: Dict[str, Any]):
        """学習完了通知"""
        message = f"✅ **{self.project_name}** - Training Complete!\n\n"
        message += f"```\n"
        message += f"Total Steps: {summary.get('total_steps', 0):,}\n"
        message += f"Total Games: {summary.get('total_games', 0):,}\n"
        message += f"Final Win Rate: {summary.get('win_rate', 0):.1f}%\n"
        message += f"Final Foul Rate: {summary.get('foul_rate', 0):.1f}%\n"
        message += f"Elapsed Time: {summary.get('elapsed_time', 'N/A')}\n"
        message += f"Model Path: {summary.get('model_path', 'N/A')}\n"
        message += f"```"
        
        self._send(message, color=0x00ff00)  # Green
    
    def _send(self, message: str, color: int = 0x0099ff):
        """実際に通知を送信"""
        if not self.enabled:
            return
        
        try:
            if self.discord_webhook:
                self._send_discord(message, color)
            if self.slack_webhook:
                self._send_slack(message)
        except Exception as e:
            print(f"[Notifier] Failed to send notification: {e}")
    
    def _send_discord(self, message: str, color: int):
        """Discord Webhook送信"""
        payload = {
            "embeds": [{
                "description": message,
                "color": color,
                "timestamp": datetime.utcnow().isoformat()
            }]
        }
        
        response = requests.post(
            self.discord_webhook,
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
    
    def _send_slack(self, message: str):
        """Slack Webhook送信"""
        # Markdownをslack形式に変換
        slack_message = message.replace("**", "*")
        
        payload = {
            "text": slack_message,
            "mrkdwn": True
        }
        
        response = requests.post(
            self.slack_webhook,
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()


# === Convenience Functions ===

_notifier: Optional[TrainingNotifier] = None

def init_notifier(
    discord_webhook: Optional[str] = None,
    slack_webhook: Optional[str] = None,
    project_name: str = "OFC Pineapple AI"
) -> TrainingNotifier:
    """グローバル通知インスタンスを初期化"""
    global _notifier
    _notifier = TrainingNotifier(
        discord_webhook=discord_webhook,
        slack_webhook=slack_webhook,
        project_name=project_name
    )
    return _notifier

def get_notifier() -> Optional[TrainingNotifier]:
    """グローバル通知インスタンスを取得"""
    global _notifier
    if _notifier is None:
        _notifier = TrainingNotifier()
    return _notifier


if __name__ == "__main__":
    # テスト
    import argparse
    
    parser = argparse.ArgumentParser(description="Test notification system")
    parser.add_argument("--discord", type=str, help="Discord webhook URL")
    parser.add_argument("--slack", type=str, help="Slack webhook URL")
    args = parser.parse_args()
    
    notifier = TrainingNotifier(
        discord_webhook=args.discord,
        slack_webhook=args.slack
    )
    
    if notifier.enabled:
        print("Sending test notification...")
        notifier.send_start({
            "timesteps": 1000000,
            "opponent_update": 50000,
            "lr": 3e-4
        })
        print("Notification sent!")
    else:
        print("No webhook configured. Use --discord or --slack to test.")
