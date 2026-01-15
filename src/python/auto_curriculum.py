"""
OFC Pineapple AI - Auto-Curriculum & Feedback System
学習結果を評価し、動的にカリキュラム（学習フェーズ）を更新する
"""

import os
import json
import numpy as np
from typing import Dict, Any, List, Optional
from notifier import get_notifier

class CurriculumManager:
    """カリキュラム管理クラス"""
    
    def __init__(self, config_path: str = "curriculum_config.json"):
        self.config_path = config_path
        self.load_config()
        
    def load_config(self):
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                self.state = json.load(f)
        else:
            self.state = {
                "current_phase": 0,
                "history": [],
                "best_win_rate": 0.0,
                "best_foul_rate": 1.0
            }
            
    def save_config(self):
        with open(self.config_path, 'w') as f:
            json.dump(self.state, f, indent=4)
            
    def evaluate_and_progress(self, metrics: Dict[str, Any]) -> str:
        """メトリクスに基づきフィードバックを生成し、フェーズを進めるか判断"""
        foul_rate = metrics.get('foul_rate', 100.0) / 100.0
        win_rate = metrics.get('win_rate', 0.0) / 100.0
        avg_score = metrics.get('avg_score', 0.0)
        
        feedback = []
        phase_updated = False
        
        # フィードバック生成
        if foul_rate > 0.5:
            feedback.append("⚠️ ファウル率が依然として高いです。報酬関数でのファウルペナルティを強化するか、より基本的な配置の学習が必要です。")
        elif foul_rate < 0.15:
            feedback.append("✅ ファウル率が安定しています。よりアグレッシブにロイヤリティを狙うフェーズに移行可能です。")
            
        if win_rate > 0.35:
            feedback.append(f"🔥 勝率 {win_rate:.1%} は良好です。対戦相手のレベルを引き上げ（Poolの更新）、Self-playの難易度を上げます。")
            phase_updated = True
        
        # 状態更新
        self.state["best_win_rate"] = max(self.state["best_win_rate"], win_rate)
        self.state["best_foul_rate"] = min(self.state["best_foul_rate"], foul_rate)
        self.state["history"].append({
            "step": metrics.get('step', 0),
            "win_rate": win_rate,
            "foul_rate": foul_rate
        })
        
        if phase_updated:
            self.state["current_phase"] += 1
            
        self.save_config()
        
        # 通知用メッセージ
        report = "📋 **Auto-Curriculum Feedback**\n"
        report += f"Current Phase: {self.state['current_phase']}\n"
        report += "\n".join(feedback)
        
        notifier = get_notifier()
        if notifier:
            notifier._send(report, color=0x9b59b6) # Purple for curriculum updates
            
        return report

# シングルトン
_manager = None
def get_curriculum_manager():
    global _manager
    if _manager is None:
        _manager = CurriculumManager()
    return _manager
