"""
PyTorchモデルをONNX形式にエクスポート
mix-poker-appのNode.jsサーバーで直接推論するため

Usage:
    # logits出力（mix-poker-app用、推奨）
    python scripts/export_onnx.py --model models/phase10_gcp/p10_fl_stay_150000000.zip --output models/onnx/ofc_ai.onnx

    # action出力（旧形式）
    python scripts/export_onnx.py --model models/phase10_gcp/p10_fl_stay_150000000.zip --output models/onnx/ofc_ai.onnx --mode action
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
from sb3_contrib import MaskablePPO


class OFCPolicyLogits(nn.Module):
    """logits出力版（mix-poker-app OFCBot.ts用）"""

    def __init__(self, sb3_model: MaskablePPO):
        super().__init__()
        policy = sb3_model.policy
        self.policy_net = policy.mlp_extractor.policy_net
        self.action_net = policy.action_net

    def forward(self, x: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
        features = self.policy_net(x)
        logits = self.action_net(features)
        masked_logits = logits + (1 - action_mask) * (-1e8)
        return masked_logits


class OFCPolicyAction(nn.Module):
    """action出力版（旧形式互換）"""

    def __init__(self, sb3_model: MaskablePPO):
        super().__init__()
        policy = sb3_model.policy
        self.policy_net = policy.mlp_extractor.policy_net
        self.action_net = policy.action_net

    def forward(self, x: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
        features = self.policy_net(x)
        logits = self.action_net(features)
        masked_logits = logits + (1 - action_mask) * (-1e8)
        action = torch.argmax(masked_logits, dim=-1)
        return action


def export_onnx(model_path: str, output_path: str, mode: str = 'logits'):
    """ONNXにエクスポート"""
    print(f"Loading model from {model_path}...")
    sb3_model = MaskablePPO.load(model_path)

    print(f"Creating inference network (mode={mode})...")
    if mode == 'logits':
        policy_net = OFCPolicyLogits(sb3_model)
        output_name = 'logits'
    else:
        policy_net = OFCPolicyAction(sb3_model)
        output_name = 'action'
    policy_net.eval()

    # ダミー入力
    batch_size = 1
    obs_dim = 881
    action_dim = 243

    dummy_obs = torch.randn(batch_size, obs_dim, dtype=torch.float32)
    dummy_mask = torch.ones(batch_size, action_dim, dtype=torch.float32)

    # テスト推論
    with torch.no_grad():
        test_output = policy_net(dummy_obs, dummy_mask)
        print(f"Test inference OK, output shape: {test_output.shape}")

    # ONNXエクスポート
    print(f"Exporting to {output_path}...")
    torch.onnx.export(
        policy_net,
        (dummy_obs, dummy_mask),
        output_path,
        input_names=['observation', 'action_mask'],
        output_names=[output_name],
        dynamic_axes={
            'observation': {0: 'batch_size'},
            'action_mask': {0: 'batch_size'},
            output_name: {0: 'batch_size'}
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


def verify_onnx(onnx_path: str, model_path: str, mode: str = 'logits'):
    """ONNXモデルの出力を元のPyTorchモデルと比較"""
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed, skipping verification")
        return

    print("\nVerifying ONNX output...")

    # PyTorchモデル
    sb3_model = MaskablePPO.load(model_path)
    if mode == 'logits':
        policy_net = OFCPolicyLogits(sb3_model)
    else:
        policy_net = OFCPolicyAction(sb3_model)
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
            pt_output = policy_net(torch.from_numpy(obs), torch.from_numpy(mask)).numpy()

        # ONNX
        ort_inputs = {'observation': obs, 'action_mask': mask}
        ort_output = ort_session.run(None, ort_inputs)[0]

        if mode == 'logits':
            # logitsの場合はargmaxで比較
            pt_action = np.argmax(pt_output[0])
            ort_action = np.argmax(ort_output[0])
            match = pt_action == ort_action
            print(f"  Test {i+1}: PyTorch action={pt_action}, ONNX action={ort_action}, Match={match}")
        else:
            match = np.array_equal(pt_output, ort_output)
            print(f"  Test {i+1}: PyTorch={pt_output[0]}, ONNX={ort_output[0]}, Match={match}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export OFC AI model to ONNX")
    parser.add_argument("--model", type=str, required=True, help="Path to .zip model")
    parser.add_argument("--output", type=str, required=True, help="Output .onnx path")
    parser.add_argument("--mode", type=str, default="logits", choices=["logits", "action"],
                        help="Output mode: logits (for OFCBot.ts) or action (legacy)")
    parser.add_argument("--verify", action="store_true", help="Verify ONNX output")
    args = parser.parse_args()

    import os
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    export_onnx(args.model, args.output, args.mode)

    if args.verify:
        verify_onnx(args.output, args.model, args.mode)
