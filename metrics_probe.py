import os
import json
import pickle
import torch
from pathlib import Path
import torch.nn as nn

# === 1. 加入和训练时一模一样的 TrainingConfig 定义 ===
from dataclasses import dataclass
from typing import Optional, Tuple, List

@dataclass
class TrainingConfig:
    weight_decay: float = 0.001
    test_size: float = 0.2
    val_size: float = 0.2
    random_state: int = 42
    n_samples: Optional[int] = None
    token_type: str = 'slt'
    min_layers: int = 5
    layer_range: Optional[Tuple[int, int]] = None
    hidden_dims: List[int] = None
    learning_rate: float = 0.001
    batch_size: int = 64
    epochs: int = 100
    loss_type: str = 'huber'
    huber_delta: float = 1.0
    entropy_min: float = -1.0
    entropy_max: float = 3.0
    save_dir: str = './models'
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 128]
# === END TrainingConfig ===

class ProbeModel(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout=0.2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Mish(),
                nn.Dropout(dropout * (1 + i * 0.1))
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.model = nn.Sequential(*layers)
        self._init_weights()
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    def forward(self, x):
        return self.model(x)

def load_probe(probe_pkl_path, device=None):
    with open(probe_pkl_path, 'rb') as f:
        save_data = pickle.load(f)
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ProbeModel(save_data['input_dim'], save_data['config'].hidden_dims)
    model.load_state_dict(save_data['model_state_dict'])
    model.to(device)
    model.eval()
    # 注意这里返回 layer_range 和 config，便于主程序选择层
    return model, save_data.get('layer_range', None), save_data['config'], device

def probe_score(model, h_range, tensor, device):
    # tensor shape: [n_layers, hidden]，只用h_range区间的层
    x = torch.tensor(tensor[h_range[0]:h_range[1]].reshape(-1)).float().to(device)
    with torch.no_grad():
        pred = model(x.unsqueeze(0)).cpu().item()
    return pred

def update_json_metrics_with_probe(json_dir: str, pkl_dir: str, probe_path: str, token_key='last_token_embedding', layer_range=None):
    # 支持外部传入层范围
    model, default_h_range, config, device = load_probe(probe_path)
    # 优先使用参数
    h_range = layer_range if layer_range is not None else default_h_range
    if h_range is None:
        raise ValueError("必须指定 --layer_range，或模型pkl文件中有 layer_range 信息！")
    if isinstance(h_range, str):  # 命令行传入是字符串
        h_range = tuple(int(x) for x in h_range.split(","))
    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]

    for fname in json_files:
        json_path = os.path.join(json_dir, fname)
        pkl_path = os.path.join(pkl_dir, fname.replace('.json', '.pkl'))
        if not os.path.exists(pkl_path):
            print(f"[WARN] {fname} 无对应pkl，跳过。")
            continue

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        with open(pkl_path, 'rb') as f:
            hidden_states = pickle.load(f)

        changed = False
        for step, rec in enumerate(data.get('reasoning_chain', [])):
            step_h = hidden_states.get(step, {}).get(token_key)
            if step_h is None:
                print(f"[WARN] {fname} step{step} 缺{token_key}，跳过。")
                continue
            try:
                pred = probe_score(model, h_range, step_h, device)
            except Exception as e:
                print(f"[ERROR] {fname} step{step} 计算失败：{e}")
                continue
            rec.setdefault('metrics', {})
            rec['metrics']['probe_pred'] = float(pred)
            changed = True

        if changed:
            # 安全写回
            tmp_path = json_path + ".tmp"
            with open(tmp_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, json_path)
            print(f"[OK] {fname} 写回完成。")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_dir", type=str, required=False, help="推理结果json目录")
    parser.add_argument("--pkl_dir", type=str, required=False, help="hidden_state pkl目录")
    parser.add_argument("--probe_path", type=str, required=False, help="探针pkl路径")
    parser.add_argument("--token_key", type=str, default='last_token_embedding', help="选用哪个hidden key")
    parser.add_argument("--layer_range", type=str, default=None, help="手动指定层范围，如 13,21")

    # ======= 明文测试参数（只需修改此处） =======
    parser.set_defaults(
        json_dir = "./outputs/xxx",
        pkl_dir = "./outputs/xxx",
        probe_path = "./probe/xxx",
        token_key = "sec_last_token_embedding",
        layer_range = "13,21"    # 如需手动，写成 "13,21"
    )
    # ======= END =======

    args = parser.parse_args()
    # 优先级处理
    manual_layer_range = args.layer_range
    if manual_layer_range is not None:
        if isinstance(manual_layer_range, str):
            manual_layer_range = tuple(int(x) for x in manual_layer_range.split(","))
    else:
        manual_layer_range = None

    update_json_metrics_with_probe(
        args.json_dir,
        args.pkl_dir,
        args.probe_path,
        args.token_key,
        manual_layer_range
    )
