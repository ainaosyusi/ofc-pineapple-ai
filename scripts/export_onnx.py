"""
PyTorchモデルをONNX形式にエクスポート
mix-poker-appのNode.jsサーバーで直接推論するため

Usage:
    python scripts/export_onnx.py --model models/phase9/p9_fl_mastery_150000000.zip --output models/onnx/ofc_ai.onnx
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
from sb3_contrib import MaskablePPO


class OFCPolicyNetwork(nn.Module):
    """推論専用のポリシーネットワーク（action_netのみ）"""

    def __init__(self, sb3_model: MaskablePPO):
        super().__init__()
        policy = sb3_model.policy

        # Feature extractor (Flatten)は単純な結合なのでスキップ
        # 入力は既に881次元にflattenされている前提

        # MLP部分を抽出
        self.policy_net = policy.mlp_extractor.policy_net
        self.action_net = policy.action_net

    def forward(self, x: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 観測ベクトル [batch, 881]
            action_mask: 有効アクションマスク [batch, 243] (1=有効, 0=無効)

        Returns:
            action: 選択されたアクション [batch]
        """
        # MLPを通す
        features = self.policy_net(x)

        # アクションのlogitsを計算
        logits = self.action_net(features)

        # マスクを適用（無効アクションは-inf）
        masked_logits = logits + (1 - action_mask) * (-1e8)

        # argmaxでアクション選択（deterministic）
        action = torch.argmax(masked_logits, dim=-1)

        return action


def export_onnx(model_path: str, output_path: str):
    """ONNXにエクスポート"""
    print(f"Loading model from {model_path}...")
    sb3_model = MaskablePPO.load(model_path)

    print("Creating inference network...")
    policy_net = OFCPolicyNetwork(sb3_model)
    policy_net.eval()

    # ダミー入力
    batch_size = 1
    obs_dim = 881
    action_dim = 243

    dummy_obs = torch.randn(batch_size, obs_dim, dtype=torch.float32)
    dummy_mask = torch.ones(batch_size, action_dim, dtype=torch.float32)

    # テスト推論
    with torch.no_grad():
        test_action = policy_net(dummy_obs, dummy_mask)
        print(f"Test inference OK, action shape: {test_action.shape}")

    # ONNXエクスポート
    print(f"Exporting to {output_path}...")
    torch.onnx.export(
        policy_net,
        (dummy_obs, dummy_mask),
        output_path,
        input_names=['observation', 'action_mask'],
        output_names=['action'],
        dynamic_axes={
            'observation': {0: 'batch_size'},
            'action_mask': {0: 'batch_size'},
            'action': {0: 'batch_size'}
        },
        opset_version=14,
        do_constant_folding=True
    )

    print("Export complete!")

    # ファイルサイズ確認
    import os
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"ONNX file size: {size_mb:.2f} MB")

    # ONNX検証
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model validation: OK")
    except ImportError:
        print("(onnx package not installed, skipping validation)")


def verify_onnx(onnx_path: str, model_path: str):
    """ONNXモデルの出力を元のPyTorchモデルと比較"""
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed, skipping verification")
        return

    print("\nVerifying ONNX output...")

    # PyTorchモデル
    sb3_model = MaskablePPO.load(model_path)
    policy_net = OFCPolicyNetwork(sb3_model)
    policy_net.eval()

    # ONNXセッション
    ort_session = ort.InferenceSession(onnx_path)

    # テストケース
    np.random.seed(42)
    for i in range(5):
        obs = np.random.randn(1, 881).astype(np.float32)
        mask = np.random.randint(0, 2, (1, 243)).astype(np.float32)
        mask[0, 0] = 1  # 少なくとも1つは有効に

        # PyTorch
        with torch.no_grad():
            pt_action = policy_net(torch.from_numpy(obs), torch.from_numpy(mask)).numpy()

        # ONNX
        ort_inputs = {'observation': obs, 'action_mask': mask}
        ort_action = ort_session.run(None, ort_inputs)[0]

        match = np.array_equal(pt_action, ort_action)
        print(f"  Test {i+1}: PyTorch={pt_action[0]}, ONNX={ort_action[0]}, Match={match}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export OFC AI model to ONNX")
    parser.add_argument("--model", type=str, required=True, help="Path to .zip model")
    parser.add_argument("--output", type=str, required=True, help="Output .onnx path")
    parser.add_argument("--verify", action="store_true", help="Verify ONNX output")
    args = parser.parse_args()

    import os
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    export_onnx(args.model, args.output)

    if args.verify:
        verify_onnx(args.output, args.model)
