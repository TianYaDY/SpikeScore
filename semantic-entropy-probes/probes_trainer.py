import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from copy import deepcopy
import datetime
from neuralndcg import neuralNDCGLoss


torch.set_float32_matmul_precision('high')
# 设置默认设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


@dataclass
class TrainingConfig:
    """训练配置"""
    weight_decay: float = 0.001
    test_size: float = 0.2
    val_size: float = 0.2
    random_state: int = 42
    n_samples: Optional[int] = None
    token_type: str = 'slt'  # 'slt' or 'tbg'
    min_layers: int = 5
    layer_range: Optional[Tuple[int, int]] = None     # 原始
    layer_indices: Optional[List[int]] = None         # 新增，优先级更高
    layer_mode: str = 'range'   # 'range' | 'first_n' | 'last_n' | 'indices'
    n_layers_select: Optional[int] = None             # 用于 first_n/last_n
    # MLP相关配置
    hidden_dims: List[int] = None
    learning_rate: float = 0.001
    batch_size: int = 64
    epochs: int = 100
    # 损失函数配置
    loss_type: str = 'huber'  # 'mse', 'mae', 'huber'
    huber_delta: float = 1.0  # Huber loss的阈值
    # 语义熵合法范围
    entropy_min: float = -1.0
    entropy_max: float = 3.0
    # 保存路径
    save_dir: str = './models'

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 128]


class ProbeModel(nn.Module):
    """探针MLP模型"""

    def __init__(self, input_dim: int, hidden_dims: List[int], dropout: float = 0.2):
        super().__init__()

        layers = []
        prev_dim = input_dim

        # 构建隐藏层
        for i, hidden_dim in enumerate(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                # nn.GELU(),
                nn.Mish(),
                nn.Dropout(dropout * (1 + i * 0.1))
            ])
            prev_dim = hidden_dim

        # 输出层
        layers.append(nn.Linear(prev_dim, 1))

        self.model = nn.Sequential(*layers)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.model(x)


class DataLoader:
    """数据加载器，专注于语义熵"""

    @staticmethod
    def scan_directory(directory: str, config: TrainingConfig) -> Dict:
        """扫描目录，提取语义熵和样本信息"""
        directory = Path(directory)
        data = {
            'directory': str(directory),
            'valid_samples': [],
            'n_samples': 0,
            'n_discarded': 0
        }

        # 1. 加载validation_generations.pkl获取样本ID
        gen_path = directory / 'validation_generations.pkl'
        if not gen_path.exists():
            print(f"Warning: {gen_path} not found")
            return data

        with open(gen_path, 'rb') as f:
            generations = pickle.load(f)

        sample_ids = list(generations.keys())

        # 2. 加载uncertainty_measures.pkl获取语义熵
        measures_path = directory / 'uncertainty_measures.pkl'
        if not measures_path.exists():
            print(f"Warning: {measures_path} not found")
            return data

        with open(measures_path, 'rb') as f:
            measures = pickle.load(f)

        # 获取语义熵数据
        try:
            entropy_data = measures['uncertainty_measures']['cluster_assignment_entropy']
        except KeyError:
            print(f"Error: No cluster_assignment_entropy found in {measures_path}")
            return data

        # 3. 匹配样本和语义熵
        if isinstance(entropy_data, list):
            # 列表格式：按顺序匹配
            if len(entropy_data) != len(sample_ids):
                print(f"Warning: Sample count mismatch - IDs: {len(sample_ids)}, Entropy: {len(entropy_data)}")

            for idx, (sample_id, entropy) in enumerate(zip(sample_ids, entropy_data)):
                # 检查语义熵是否在合理范围内
                if config.entropy_min <= entropy <= config.entropy_max:
                    # 检查是否有隐藏状态
                    record = generations[sample_id]
                    has_hidden = False

                    if config.token_type == 'slt' and 'emb_tok_before_eos' in record['most_likely_answer']:
                        has_hidden = True
                    elif config.token_type == 'tbg' and 'emb_last_tok_before_gen' in record['most_likely_answer']:
                        has_hidden = True

                    if has_hidden:
                        data['valid_samples'].append({
                            'id': sample_id,
                            'entropy': float(entropy),
                            'idx': idx  # 保留原始索引以便调试
                        })
                    else:
                        data['n_discarded'] += 1
                else:
                    data['n_discarded'] += 1
                    if idx < 5:  # 只打印前几个异常值
                        print(f"  Discarded sample {idx}: entropy={entropy:.4f} (out of range)")

        elif isinstance(entropy_data, dict):
            # 字典格式
            for sample_id, entropy in entropy_data.items():
                if sample_id in generations and config.entropy_min <= entropy <= config.entropy_max:
                    record = generations[sample_id]
                    has_hidden = False

                    if config.token_type == 'slt' and 'emb_tok_before_eos' in record['most_likely_answer']:
                        has_hidden = True
                    elif config.token_type == 'tbg' and 'emb_last_tok_before_gen' in record['most_likely_answer']:
                        has_hidden = True

                    if has_hidden:
                        data['valid_samples'].append({
                            'id': sample_id,
                            'entropy': float(entropy)
                        })
                    else:
                        data['n_discarded'] += 1
                else:
                    data['n_discarded'] += 1

        # 限制样本数量
        if config.n_samples and len(data['valid_samples']) > config.n_samples:
            data['valid_samples'] = data['valid_samples'][:config.n_samples]

        data['n_samples'] = len(data['valid_samples'])

        # 打印统计
        print(f"  Total records: {len(sample_ids)}")
        print(f"  Valid samples: {data['n_samples']}")
        print(f"  Discarded: {data['n_discarded']}")

        if data['n_samples'] > 0:
            entropies = [s['entropy'] for s in data['valid_samples']]
            print(f"  Entropy range: [{min(entropies):.4f}, {max(entropies):.4f}]")
            print(f"  Entropy mean: {np.mean(entropies):.4f} ± {np.std(entropies):.4f}")

        return data


def get_selected_layer_indices(n_layers, config: TrainingConfig):
    if config.layer_indices is not None:
        return list(config.layer_indices)
    if config.layer_mode == 'range':
        start, end = config.layer_range or (0, n_layers)
        return list(range(start, end))
    elif config.layer_mode == 'first_n':
        n = config.n_layers_select or 1
        return list(range(n))
    elif config.layer_mode == 'last_n':
        n = config.n_layers_select or 1
        return list(range(n_layers - n, n_layers))
    else:
        raise ValueError(f"Unknown layer selection mode: {config.layer_mode}")





class LazyHiddenStateDataset(Dataset):
    """延迟加载数据集"""

    def __init__(self, data_info: List[Dict], config: TrainingConfig, device: torch.device):
        self.data_info = data_info
        self.token_type = config.token_type
        self.device = device
        self.config = config
        self.cache = {}
        self.cache_size_limit = 10

    def __len__(self):
        return len(self.data_info)

    def _load_directory_data(self, directory: str, needed_ids: set):
        """加载指定目录中需要的样本"""
        gen_path = Path(directory) / 'validation_generations.pkl'

        with open(gen_path, 'rb') as f:
            generations = pickle.load(f)

        hidden_dict = {}

        for sample_id in needed_ids:
            if sample_id not in generations:
                continue

            record = generations[sample_id]
            try:
                if self.token_type == 'slt':
                    hidden = record['most_likely_answer']['emb_tok_before_eos'].squeeze(-2)
                else:  # tbg
                    hidden = record['most_likely_answer']['emb_last_tok_before_gen'].squeeze(-2)

                hidden_dict[sample_id] = hidden.to(torch.float32).to(self.device)
            except Exception as e:
                print(f"Warning: Failed to load hidden states for sample {sample_id}: {e}")

        self.cache[directory] = hidden_dict

        # 缓存管理
        if len(self.cache) > self.cache_size_limit:
            oldest_key = list(self.cache.keys())[0]
            del self.cache[oldest_key]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def __getitem__(self, idx):
        info = self.data_info[idx]
        cache_key = info['directory']
        sample_id = info['id']

        # 检查缓存
        if cache_key not in self.cache:
            # 收集这个目录中需要的所有ID
            needed_ids = {item['id'] for item in self.data_info if item['directory'] == info['directory']}
            self._load_directory_data(cache_key, needed_ids)

        # 获取隐藏状态
        if sample_id not in self.cache[cache_key]:
            raise ValueError(f"Sample {sample_id} not found in cached data")

        hidden_states_all = self.cache[cache_key][sample_id]
        n_layers = hidden_states_all.shape[0]
        layer_indices = get_selected_layer_indices(n_layers, self.config)
        hidden_states = hidden_states_all[layer_indices]  # 支持任意index list
        hidden_states = hidden_states.reshape(-1)

        return hidden_states, torch.tensor(info['entropy'], dtype=torch.float32, device=self.device)


class SemanticEntropyProbeTrainer:
    """语义熵探针训练器"""

    def __init__(self, config: TrainingConfig = TrainingConfig()):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.best_model = None
        self.last_model = None
        self.layer_range = None
        self.training_history = []

        # 设置随机种子（修复版）
        torch.manual_seed(config.random_state)
        np.random.seed(config.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.random_state)  # 使用manual_seed_all
            # 确保CUDNN的确定性
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    # def _get_loss_function(self):
    #     """根据配置返回损失函数"""
    #     if self.config.loss_type == 'mse':
    #         return nn.MSELoss()
    #     elif self.config.loss_type == 'mae':
    #         return nn.L1Loss()
    #     elif self.config.loss_type == 'huber':
    #         return nn.HuberLoss(delta=self.config.huber_delta)
    #     else:
    #         raise ValueError(f"Unknown loss type: {self.config.loss_type}")

    def _get_loss_function(self):
        if self.config.loss_type == 'mse':
            return nn.MSELoss()
        elif self.config.loss_type == 'mae':
            return nn.L1Loss()
        elif self.config.loss_type == 'huber':
            return nn.HuberLoss(delta=self.config.huber_delta)
        elif self.config.loss_type == 'correlation':
            # 用于最大化Pearson correlation，负相关损失
            def correlation_loss(pred, target):
                pred = pred.squeeze()
                target = target.squeeze()
                vx = pred - pred.mean()
                vy = target - target.mean()
                corr = (vx * vy).sum() / (torch.sqrt((vx ** 2).sum() + 1e-8) * torch.sqrt((vy ** 2).sum() + 1e-8))
                return 1 - corr  # 最大化相关性

            return correlation_loss
        elif self.config.loss_type == 'neuralndcg':
            # NeuralNDCG（可微NDCG排序损失，越小越好）
            def neuralndcg_loss(pred, target):
                # 保证 pred 和 target 形状 [B, n]，如果是一维，需要 reshape
                if pred.dim() == 1:
                    pred = pred.unsqueeze(0)
                if target.dim() == 1:
                    target = target.unsqueeze(0)
                return neuralNDCGLoss(pred, target)

            return neuralndcg_loss
        else:
            raise ValueError(f"Unknown loss type: {self.config.loss_type}")

    def train_from_directories(self, directories: List[str], dataset_names: List[str] = None):
        """从多个目录训练统一的语义熵回归探针"""

        # 1. 扫描所有目录
        all_data_info = []
        entropy_values = []

        print(f"\nScanning {len(directories)} directories...")

        for i, directory in enumerate(directories):
            dataset_name = dataset_names[i] if dataset_names else f"dataset_{i}"
            print(f"\nProcessing {dataset_name} from {directory}...")

            # 扫描目录
            data = DataLoader.scan_directory(directory, self.config)

            if not data['valid_samples']:
                print(f"  ✗ No valid samples found, skipping...")
                continue

            # 添加数据集信息
            for sample in data['valid_samples']:
                all_data_info.append({
                    'directory': directory,
                    'id': sample['id'],
                    'entropy': sample['entropy'],
                    'dataset_name': dataset_name
                })
                entropy_values.append(sample['entropy'])

        if not all_data_info:
            raise ValueError("No valid data found in any directory!")

        print(f"\nTotal samples collected: {len(all_data_info)}")
        print(f"Overall entropy statistics:")
        print(f"  Mean: {np.mean(entropy_values):.4f}")
        print(f"  Std: {np.std(entropy_values):.4f}")
        print(f"  Range: [{min(entropy_values):.4f}, {max(entropy_values):.4f}]")

        # 2. 选择最佳层范围
        # 优先级：layer_indices > first_n/last_n > range > 自动
        if self.config.layer_indices is not None or self.config.layer_mode in ['first_n', 'last_n', 'indices']:
            print("\nUsing user-defined layer selection:", end=' ')
            if self.config.layer_indices is not None:
                print(f"layer_indices={self.config.layer_indices}")
            else:
                print(
                    f"layer_mode={self.config.layer_mode}, n_layers_select={self.config.n_layers_select}, layer_range={self.config.layer_range}")
            # 这里 layer_range 变量只是记录保存用，不会实际用于 get_selected_layer_indices
            layer_range = self.config.layer_range
        elif self.config.layer_range:
            print(f"\nUsing predefined layer range: {self.config.layer_range}")
            layer_range = self.config.layer_range
        else:
            print("\nSelecting best layer range...")
            layer_range = self._select_best_layers(all_data_info)
            self.config.layer_range = layer_range  # 自动选择后，写入config以便保存

        # 3. 创建数据集和训练
        print(f"\nTraining regression model with {self.config.loss_type} loss...")

        # 创建延迟加载数据集
        dataset = LazyHiddenStateDataset(all_data_info, self.config, self.device)

        # 分割训练/测试集
        n_samples = len(dataset)
        indices = torch.randperm(n_samples)
        n_train = int(n_samples * (1 - self.config.test_size))

        train_indices = indices[:n_train]
        test_indices = indices[n_train:]

        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        test_dataset = torch.utils.data.Subset(dataset, test_indices)

        train_loader = TorchDataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        test_loader = TorchDataLoader(test_dataset, batch_size=self.config.batch_size, shuffle=False)

        # 获取输入维度
        sample_x, _ = dataset[0]
        input_dim = sample_x.shape[0]

        # 创建模型
        model = ProbeModel(input_dim, self.config.hidden_dims).to(self.device)

        # 训练
        result = self._train_model(model, train_loader, test_loader)

        # 添加元数据
        result['metadata'] = {
            'datasets': list(set(info['dataset_name'] for info in all_data_info)),
            'total_samples': len(all_data_info),
            'layer_range': layer_range,
            'token_type': self.config.token_type,
            'input_dim': input_dim,
            'loss_type': self.config.loss_type
        }

        return result

    def _select_best_layers(self, all_data_info: List[Dict]) -> Tuple[int, int]:
        """选择最佳层范围（使用配置的损失函数）"""
        # 采样一部分数据
        sample_size = min(1000, len(all_data_info))
        sample_indices = np.random.choice(len(all_data_info), sample_size, replace=False)
        sample_info = [all_data_info[i] for i in sample_indices]

        # 按目录分组
        samples_by_dir = defaultdict(list)
        for info in sample_info:
            samples_by_dir[info['directory']].append((info['id'], info['entropy']))

        # 加载样本隐藏状态
        sample_hidden_states = []
        sample_labels = []

        for directory, samples in samples_by_dir.items():
            gen_path = Path(directory) / 'validation_generations.pkl'
            with open(gen_path, 'rb') as f:
                generations = pickle.load(f)

            for sample_id, entropy in samples:
                if sample_id not in generations:
                    continue

                record = generations[sample_id]
                try:
                    if self.config.token_type == 'slt':
                        hidden = record['most_likely_answer']['emb_tok_before_eos'].squeeze(-2)
                    else:
                        hidden = record['most_likely_answer']['emb_last_tok_before_gen'].squeeze(-2)

                    sample_hidden_states.append(hidden.to(torch.float32).to(self.device))
                    sample_labels.append(entropy)
                except:
                    continue

        if not sample_hidden_states:
            raise ValueError("Failed to load sample hidden states!")

        sample_hidden_states = torch.stack(sample_hidden_states, dim=1)
        sample_labels = torch.tensor(sample_labels, device=self.device, dtype=torch.float32)

        # 搜索最佳层范围
        n_layers = sample_hidden_states.shape[0]
        best_score = float('inf')
        best_range = (0, min(10, n_layers))

        # 使用配置的损失函数
        criterion = self._get_loss_function()

        for start in range(0, min(n_layers - self.config.min_layers, 20)):
            for end in range(start + self.config.min_layers, min(n_layers + 1, start + 15)):
                # 拼接层
                X_concat = sample_hidden_states[start:end].transpose(0, 1).reshape(len(sample_labels), -1)

                # 快速线性回归评估
                probe = nn.Linear(X_concat.shape[1], 1).to(self.device)
                optimizer = optim.AdamW(probe.parameters(), lr=0.01)

                # 快速训练
                n_train = int(len(sample_labels) * 0.8)
                train_idx = torch.randperm(len(sample_labels))[:n_train]
                val_idx = torch.randperm(len(sample_labels))[n_train:]

                for _ in range(20):
                    optimizer.zero_grad()
                    output = probe(X_concat[train_idx]).squeeze()
                    loss = criterion(output, sample_labels[train_idx])
                    loss.backward()
                    optimizer.step()

                # 验证
                with torch.no_grad():
                    val_output = probe(X_concat[val_idx]).squeeze()
                    val_loss = criterion(val_output, sample_labels[val_idx]).item()

                if val_loss < best_score:
                    best_score = val_loss
                    best_range = (start, end)

                del probe, optimizer

        # 清理
        del sample_hidden_states, sample_labels
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"Selected layer range: {best_range} ({self.config.loss_type} loss: {best_score:.4f})")
        return best_range

    def _train_model(self, model, train_loader, test_loader):
        """训练回归模型"""
        criterion = self._get_loss_function()
        optimizer = optim.AdamW(model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

        best_val_loss = float('inf')
        best_model_state = None
        best_epoch = 0

        # 训练历史
        self.training_history = []

        # 训练循环
        pbar = tqdm(range(self.config.epochs), desc="Training")
        for epoch in pbar:
            # 训练阶段
            model.train()
            train_loss = 0

            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_x).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # 验证阶段
            model.eval()
            val_loss = 0
            all_val_outputs = []
            all_val_labels = []

            with torch.no_grad():
                for batch_x, batch_y in test_loader:
                    outputs = model(batch_x).squeeze()
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
                    all_val_outputs.extend(outputs.cpu().numpy())
                    all_val_labels.extend(batch_y.cpu().numpy())

            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(test_loader)

            # 保存训练历史
            self.training_history.append({
                'epoch': epoch,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss
            })

            # 更新学习率
            scheduler.step(avg_val_loss)

            # 保存最佳模型（修复：使用深拷贝）
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = deepcopy(model.state_dict())  # 使用深拷贝
                best_epoch = epoch

            # 更新进度条
            pbar.set_postfix({
                'train_loss': f'{avg_train_loss:.4f}',
                'val_loss': f'{avg_val_loss:.4f}',
                'best_epoch': best_epoch
            })

        # 保存最后一个模型的状态（也使用深拷贝）
        last_model_state = deepcopy(model.state_dict())

        # 创建两个模型实例
        input_dim = model.model[0].in_features
        self.best_model = ProbeModel(input_dim, self.config.hidden_dims).to(self.device)
        self.last_model = ProbeModel(input_dim, self.config.hidden_dims).to(self.device)

        # 加载对应的权重
        self.best_model.load_state_dict(best_model_state)
        self.last_model.load_state_dict(last_model_state)

        print(f"\nTraining completed!")
        print(f"Best model was at epoch {best_epoch} with {self.config.loss_type} loss={best_val_loss:.4f}")
        print(f"Last model at epoch {self.config.epochs - 1} with {self.config.loss_type} loss={avg_val_loss:.4f}")

        # 计算最终指标（使用最佳模型）
        from scipy.stats import pearsonr, spearmanr

        self.best_model.eval()
        all_outputs = []
        all_labels = []

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                outputs = self.best_model(batch_x).squeeze()
                all_outputs.extend(outputs.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())

        all_outputs = np.array(all_outputs)
        all_labels = np.array(all_labels)

        pearson_corr, _ = pearsonr(all_outputs, all_labels)
        spearman_corr, _ = spearmanr(all_outputs, all_labels)

        # 计算多种误差指标
        mse = np.mean((all_outputs - all_labels) ** 2)
        mae = np.mean(np.abs(all_outputs - all_labels))

        # 计算Huber loss (numpy版本)
        residual = np.abs(all_outputs - all_labels)
        huber_loss = np.where(residual <= self.config.huber_delta,
                              0.5 * residual ** 2,
                              self.config.huber_delta * (residual - 0.5 * self.config.huber_delta))
        huber_loss = np.mean(huber_loss)

        return {
            'best_model': self.best_model,
            'last_model': self.last_model,
            'metrics': {
                'best_val_loss': best_val_loss,
                'last_val_loss': avg_val_loss,
                'best_epoch': best_epoch,
                'pearson_correlation': pearson_corr,
                'spearman_correlation': spearman_corr,
                'mse': mse,
                'mae': mae,
                'huber': huber_loss,
                'primary_loss': self.config.loss_type
            }
        }

    # def save_models(self, save_dir: Optional[str] = None, prefix: str = 'semantic_entropy_probe'):
    #     """保存最佳模型和最后模型"""
    #     if self.best_model is None or self.last_model is None:
    #         raise ValueError("No trained models to save!")
    #
    #     # 使用指定的路径或配置中的路径
    #     save_dir = save_dir or self.config.save_dir
    #     save_dir = Path(save_dir)
    #     save_dir.mkdir(parents=True, exist_ok=True)
    #
    #     # 准备保存数据
    #     save_data = {
    #         'best_model_state_dict': self.best_model.cpu().state_dict(),
    #         'last_model_state_dict': self.last_model.cpu().state_dict(),
    #         'layer_range': self.layer_range,
    #         'config': self.config,
    #         'input_dim': self.best_model.model[0].in_features,
    #         'training_history': self.training_history
    #     }
    #
    #     # 保存文件
    #     filepath = save_dir / f'{prefix}.pkl'
    #     with open(filepath, 'wb') as f:
    #         pickle.dump(save_data, f)
    #
    #     # 找出最佳epoch
    #     best_epoch = min(self.training_history, key=lambda x: x['val_loss'])['epoch']
    #
    #     print(f"\nModels saved to {filepath}")
    #     print(f"  - Best model (epoch {best_epoch})")
    #     print(f"  - Last model (epoch {len(self.training_history) - 1})")
    #
    #     # 将模型移回原设备
    #     self.best_model.to(self.device)
    #     self.last_model.to(self.device)
    #
    #     return filepath

    def save_models(self, save_dir: Optional[str] = None, prefix: str = 'semantic_entropy_probe'):
        """分别保存最佳模型和最后模型为两个文件"""
        if self.best_model is None or self.last_model is None:
            raise ValueError("No trained models to save!")

        save_dir = save_dir or self.config.save_dir
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # 通用保存信息
        common_info = {
            'layer_range': self.layer_range,
            'config': self.config,
            'input_dim': self.best_model.model[0].in_features,
            'training_history': self.training_history
        }

        # 保存best模型
        best_path = save_dir / f'{prefix}_best.pkl'
        save_data_best = {'model_state_dict': self.best_model.cpu().state_dict(), **common_info}
        with open(best_path, 'wb') as f:
            pickle.dump(save_data_best, f)

        # 保存last模型
        last_path = save_dir / f'{prefix}_last.pkl'
        save_data_last = {'model_state_dict': self.last_model.cpu().state_dict(), **common_info}
        with open(last_path, 'wb') as f:
            pickle.dump(save_data_last, f)

        print(f"\nBest model saved to {best_path}")
        print(f"Last model saved to {last_path}")

        # 将模型移回原设备
        self.best_model.to(self.device)
        self.last_model.to(self.device)

        return str(best_path), str(last_path)

    @classmethod
    def load_models(cls, filepath: str, device: Optional[torch.device] = None):
        """加载保存的模型（返回最佳和最后模型）"""
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 创建模型
        best_model = ProbeModel(
            save_data['input_dim'],
            save_data['config'].hidden_dims
        ).to(device)

        last_model = ProbeModel(
            save_data['input_dim'],
            save_data['config'].hidden_dims
        ).to(device)

        # 加载权重
        best_model.load_state_dict(save_data['best_model_state_dict'])
        last_model.load_state_dict(save_data['last_model_state_dict'])

        best_model.eval()
        last_model.eval()

        return {
            'best_model': best_model,
            'last_model': last_model,
            'layer_range': save_data['layer_range'],
            'config': save_data['config'],
            'training_history': save_data.get('training_history', [])
        }

    @staticmethod
    def load_single_model(filepath: str, device: Optional[torch.device] = None):
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = ProbeModel(
            save_data['input_dim'],
            save_data['config'].hidden_dims
        ).to(device)
        model.load_state_dict(save_data['model_state_dict'])
        model.eval()
        return model, save_data['layer_range'], save_data['config'], save_data.get('training_history', [])


# 使用示例
if __name__ == "__main__":
    # 创建配置
    config = TrainingConfig(
        n_samples=None,  # 使用全部数据
        test_size=0.1,
        hidden_dims=[1024, 512, 128],  # 2层MLP
        learning_rate=0.0001,
        weight_decay = 0.001,
        batch_size=16,
        epochs=200,
        token_type='tbg',  # 或 'slt'
        # token_type='slt',  # 或 'slt'
        loss_type='huber',  # 使用Huber loss, 或 'mse', 'mae', 'correlation', 'neuralndcg'
        huber_delta=0.5,  # 对于语义熵范围[0, 2.3]，0.5是个合理的阈值
        entropy_min=-0.1,  # 语义熵合法范围
        entropy_max=2.5,
        save_dir='./probe_output',  # 指定保存目录
        # layer_range=(13, 21),  # 可选：手动指定层范围
        # layer_mode='range',
        # layer_range=(13, 21)
        layer_mode='last_n',
        n_layers_select=8
    )

    # 创建训练器
    trainer = SemanticEntropyProbeTrainer(config)

    # 从多个目录训练
    results = trainer.train_from_directories(
        directories=[
            './semantic_uncertainty/xxxx',
            './semantic_uncertainty/xxxx',
            # 添加更多目录...
        ],
        dataset_names=['nq', 'nq']  # 对应的数据集名称
    )

    # 查看结果
    print("\n=== Training Results ===")
    print(f"Datasets: {results['metadata']['datasets']}")
    print(f"Total samples: {results['metadata']['total_samples']}")
    print(f"Layer range: {results['metadata']['layer_range']}")
    print(f"Token type: {results['metadata']['token_type']}")
    print(f"Loss type: {results['metadata']['loss_type']}")
    print("\nMetrics (Best Model):")
    for metric, value in results['metrics'].items():
        if isinstance(value, (int, float)):
            print(f"  {metric}: {value:.4f}")

    # 保存模型


    exp_folder = datetime.datetime.now().strftime("llama3.2-3b_all_probe_%Y%m%d_%H%M%S")
    saved_path = trainer.save_models(save_dir=f"./probe_output/{exp_folder}", prefix="llama3.2-3b_all_tbg_probe")

    # # 测试加载
    # print("\n=== Testing Model Loading ===")
    # loaded = SemanticEntropyProbeTrainer.load_models(saved_path)
    # print(f"Successfully loaded best and last models")
    # print(f"Layer range: {loaded['layer_range']}")
    # print(f"Loss type used: {loaded['config'].loss_type}")