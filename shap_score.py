import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'stix'

TOKEN_KEYS = ["last_token_embedding", "sec_last_token_embedding", "last_tok_bef_gen_embedding"]
# --------- 特征名/符号/中文解释表 ---------
TOKEN_SYMBOLS = [r'$t_1$', r'$t_2$', r'$t_3$']
DYN_STATS = [
    ("mean", r"$\mu$", "所有步的均值"),
    ("std", r"$\sigma$", "所有步的标准差"),
    ("last_first", r"$\Delta$", "最后一步-第一步"),
    ("range", r"$\rho$", "极差（最大-最小）"),
    ("max_jump", r"$J$", "最大单步变化"),
    ("sum_jump", r"$T$", "总变化量"),
    ("max_accel", r"$A$", "最大二阶变化"),
    ("slope", r"$k$", "变化斜率")
]

MEAN_SEQ_SYMBOL = r'$\bar{s}_i$'
MEAN_DYN_LABELS = [
    ("mean_score_mean", r"$\mu$", "步均值的均值"),
    ("mean_score_std", r"$\sigma$", "步均值的标准差"),
    ("mean_score_last_first", r"$\Delta$", "步均值最后一步-第一步"),
    ("mean_score_range", r"$\rho$", "步均值极差"),
    ("mean_score_max_jump", r"$J$", "步均值最大单步变化"),
    ("mean_score_sum_jump", r"$T$", "步均值总变化量"),
    ("mean_score_max_accel", r"$A$", "步均值最大二阶变化"),
    ("mean_score_slope", r"$k$", "步均值变化斜率"),
]


def extract_features_labels(data_dir, min_steps=2):
    json_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
    all_features = []
    all_labels = []
    file_ids = []
    max_steps = 0
    for f in json_files:
        with open(os.path.join(data_dir, f), encoding='utf-8') as jf:
            data = json.load(jf)
        steps = data.get('reasoning_chain', [])
        if len(steps) > max_steps:
            max_steps = len(steps)
    for f in json_files:
        try:
            with open(os.path.join(data_dir, f), encoding='utf-8') as jf:
                data = json.load(jf)
            label = data.get('nli_label', None)
            if label not in (0, 1):
                continue
            steps = data.get('reasoning_chain', [])
            if len(steps) < min_steps:
                continue
            scores = []
            for i in range(max_steps):
                if i < len(steps):
                    metrics = steps[i].get('metrics', {})
                    token_scores = metrics.get('reasoning_score_per_token', {})
                    per_token = []
                    for key in TOKEN_KEYS:
                        v = token_scores.get(key, None)
                        per_token.append(float(v) if v is not None else 0.0)
                    scores.append(per_token)
                else:
                    scores.append([0.0] * len(TOKEN_KEYS))
            scores = np.array(scores)  # [steps, tokens]
            features_static = scores.flatten().tolist()  # 静态特征

            # ----- Token纵向动态特征 -----
            features_token_dyn = []
            for ti, key in enumerate(TOKEN_KEYS):
                seq = scores[:, ti]
                d1 = np.diff(seq)
                d2 = np.diff(d1)
                features_token_dyn.append(np.mean(seq))  # mean
                features_token_dyn.append(np.std(seq))  # std
                features_token_dyn.append(seq[-1] - seq[0])  # last_first
                features_token_dyn.append(np.max(seq) - np.min(seq))  # range
                features_token_dyn.append(np.max(np.abs(d1)) if len(d1) else 0)  # max_jump
                features_token_dyn.append(np.sum(np.abs(d1)))  # sum_jump
                features_token_dyn.append(np.max(np.abs(d2)) if len(d2) else 0)  # max_accel
                features_token_dyn.append(np.polyfit(np.arange(len(seq)), seq, 1)[0] if len(seq) > 1 else 0)  # slope

            # ----- 步均值动态特征 -----
            mean_seq = scores.mean(axis=1)
            d1 = np.diff(mean_seq)
            d2 = np.diff(d1)
            features_mean_dyn = [
                np.mean(mean_seq),
                np.std(mean_seq),
                mean_seq[-1] - mean_seq[0],
                np.max(mean_seq) - np.min(mean_seq),
                np.max(np.abs(d1)) if len(d1) else 0,
                np.sum(np.abs(d1)),
                np.max(np.abs(d2)) if len(d2) else 0,
                np.polyfit(np.arange(len(mean_seq)), mean_seq, 1)[0] if len(mean_seq) > 1 else 0,
            ]

            all_features.append(features_static + features_token_dyn + features_mean_dyn)
            all_labels.append(label)
            file_ids.append(f)
        except Exception as e:
            print(f"Skip {f}: {e}")

    # ---- 生成特征名/符号/中文 ----
    static_feature_names, static_feature_symbols, static_feature_zh = [], [], []
    for i in range(max_steps):
        for j, key in enumerate(TOKEN_KEYS):
            static_feature_names.append(f"step{i}_{key}")
            static_feature_symbols.append(f"$s_{{{i + 1},{j + 1}}}$")
            static_feature_zh.append(f"第{i + 1}步 {key} 原始分数")
    # token动态
    token_dyn_names, token_dyn_symbols, token_dyn_zh = [], [], []
    for j, key in enumerate(TOKEN_KEYS):
        for stat_en, stat_sym, stat_zh in DYN_STATS:
            token_dyn_names.append(f"{key}_{stat_en}")
            token_dyn_symbols.append(f"{TOKEN_SYMBOLS[j]}{stat_sym}")
            token_dyn_zh.append(f"{key} 步方向{stat_zh}")
    # 均值动态
    mean_dyn_names = [x[0] for x in MEAN_DYN_LABELS]
    mean_dyn_symbols = [MEAN_SEQ_SYMBOL + x[1] for x in MEAN_DYN_LABELS]
    mean_dyn_zh = [x[2] for x in MEAN_DYN_LABELS]
    feature_names = static_feature_names + token_dyn_names + mean_dyn_names
    feature_symbols = static_feature_symbols + token_dyn_symbols + mean_dyn_symbols
    feature_zh = static_feature_zh + token_dyn_zh + mean_dyn_zh
    return np.array(all_features), np.array(all_labels), file_ids, feature_names, feature_symbols, feature_zh


# --------- 2. MLP网络定义 ---------
class HallucinationMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


# --------- 3. 主流程入口 ---------
def main(data_dir, report_dir):
    os.makedirs(report_dir, exist_ok=True)
    print("==== 数据加载 & 特征工程 ====")
    X, y, file_ids, feature_names, feature_symbols, feature_zh = extract_features_labels(data_dir, min_steps=2)
    X = np.nan_to_num(X, nan=0.0)
    print(f"Total samples: {len(X)}; X.shape={X.shape}")
    print("First 5 feature_symbols:", feature_symbols[:5])
    print("First 2 feature vectors:\n", X[:2])

    if len(X) == 0:
        print("没有有效样本！")
        return

    # 划分
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    print("==== 训练MLP分类器 ====")
    model = HallucinationMLP(input_dim=X.shape[1], hidden_dim=128)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    loss_fn = nn.BCELoss()

    # 训练
    for epoch in range(5000):
        model.train()
        out = model(X_train_t)
        loss = loss_fn(out, y_train_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0 or epoch == 499:
            model.eval()
            with torch.no_grad():
                y_pred = model(X_test_t).cpu().detach().numpy().flatten()
                auc = roc_auc_score(y_test, y_pred)
            print(f"Epoch {epoch} | Loss {loss.item():.4f} | Test AUC {auc:.4f}")

    # 最终评估与绘图
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_t).cpu().detach().numpy().flatten()
    final_auc = roc_auc_score(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    print("Final Test AUC:", final_auc)

    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (AUC={final_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("MLP ROC Curve: Hallucination")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(report_dir, "mlp_roc_curve.png"))
    plt.close()

    plt.figure()
    plt.hist([y_pred[y_test == 0], y_pred[y_test == 1]], bins=20, alpha=0.7,
             label=["Hallucination", "No Hallucination"])
    plt.xlabel("MLP Predicted Score")
    plt.ylabel("Sample Count")
    plt.legend()
    plt.title("MLP Prediction Score Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(report_dir, "mlp_score_dist.png"))
    plt.close()
    print(f"评估结果保存在：{report_dir}")

    # --------- 4. SHAP 可解释性 ---------
    print("==== 进行SHAP特征归因分析 ====")
    bg_num = min(100, len(X_train))
    test_num = min(100, len(X_test))
    background = X_train[:bg_num]
    explainer = shap.DeepExplainer(model, torch.tensor(background, dtype=torch.float32))
    shap_values = explainer.shap_values(torch.tensor(X_test[:test_num], dtype=torch.float32))

    # 处理SHAP值的形状
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    # 确保shap_values是2D数组
    print(f"原始SHAP值形状: {shap_values.shape}")
    print(f"SHAP值类型: {type(shap_values)}")

    # 如果是3D，需要正确处理
    if shap_values.ndim == 3:
        # 通常是 (1, samples, features) 或 (samples, 1, features)
        if shap_values.shape[0] == 1:
            shap_values = shap_values[0]
        elif shap_values.shape[1] == 1:
            shap_values = shap_values[:, 0, :]

    # 确保是numpy数组
    if torch.is_tensor(shap_values):
        shap_values = shap_values.cpu().numpy()

    shap_values = np.array(shap_values)  # 再次确保是numpy数组
    print(f"处理后SHAP值形状: {shap_values.shape}")

    # ---- 正确计算特征分组索引 ----
    static_len = len([n for n in feature_names if n.startswith("step")])
    token_dyn_len = len(TOKEN_KEYS) * len(DYN_STATS)  # 3个token × 8个统计量 = 24
    mean_dyn_len = len(MEAN_DYN_LABELS)  # 8个统计量

    print(f"特征分组: 静态={static_len}, Token动态={token_dyn_len}, 均值动态={mean_dyn_len}")
    print(f"总特征数: {len(feature_names)}")

    # ------ 静态特征 SHAP ------
    if static_len > 0:
        # 计算平均绝对SHAP值
        static_shap = shap_values[:, :static_len]
        mean_abs_shap_static = np.abs(static_shap).mean(axis=0)

        print(f"静态SHAP形状: {static_shap.shape}")
        print(f"平均绝对SHAP静态形状: {mean_abs_shap_static.shape}")
        print(f"平均绝对SHAP静态类型: {type(mean_abs_shap_static)}")

        # 确保是1D数组
        mean_abs_shap_static = np.squeeze(mean_abs_shap_static)

        # 按重要性排序（降序）
        static_indices = np.argsort(mean_abs_shap_static)[::-1]

        # 只显示前20个最重要的静态特征
        top_n = min(20, static_len)
        top_static_indices = static_indices[:top_n]

        # 提取对应的SHAP值
        top_static_values = []
        for idx in top_static_indices:
            top_static_values.append(float(mean_abs_shap_static[idx]))
        top_static_values = np.array(top_static_values)

        print(f"Top静态值形状: {top_static_values.shape}")
        print(f"Top静态值: {top_static_values[:5]}")

        plt.figure(figsize=(10, 8))
        y_pos = np.arange(top_n)
        plt.barh(y_pos, top_static_values)
        plt.yticks(y_pos, [feature_symbols[i] for i in top_static_indices])
        plt.xlabel('Mean |SHAP value|')
        plt.title('Top 20 Static Features by Importance')
        plt.gca().invert_yaxis()  # 最重要的在上面
        plt.tight_layout()
        plt.savefig(os.path.join(report_dir, "shap_static_bar.png"), dpi=150)
        plt.close()

    # ------ 动态特征 SHAP ------
    dyn_start = static_len
    dyn_end = static_len + token_dyn_len + mean_dyn_len

    # 计算动态特征的平均绝对SHAP值
    dyn_shap = shap_values[:, dyn_start:dyn_end]
    mean_abs_shap_dynamic = np.abs(dyn_shap).mean(axis=0)

    # 确保是1D数组
    mean_abs_shap_dynamic = np.squeeze(mean_abs_shap_dynamic)

    # 按重要性排序
    dyn_indices = np.argsort(mean_abs_shap_dynamic)[::-1]

    # 提取对应的SHAP值
    dyn_values = []
    for idx in dyn_indices:
        dyn_values.append(float(mean_abs_shap_dynamic[idx]))
    dyn_values = np.array(dyn_values)

    plt.figure(figsize=(10, 12))
    y_pos = np.arange(len(dyn_indices))
    plt.barh(y_pos, dyn_values)
    plt.yticks(y_pos, [feature_symbols[dyn_start + i] for i in dyn_indices])
    plt.xlabel('Mean |SHAP value|')
    plt.title('Dynamic Features by Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(report_dir, "shap_dynamic_bar.png"), dpi=150)
    plt.close()

    # ------ 分组特征重要性对比 ------
    # 计算每个特征组的平均重要性
    static_importance = float(np.abs(shap_values[:, :static_len]).mean()) if static_len > 0 else 0
    token_dyn_importance = float(np.abs(shap_values[:, static_len:static_len + token_dyn_len]).mean())
    mean_dyn_importance = float(np.abs(shap_values[:, static_len + token_dyn_len:]).mean())

    plt.figure(figsize=(8, 6))
    groups = ['Static\nFeatures', 'Token Dynamic\nFeatures', 'Mean Dynamic\nFeatures']
    importances = [static_importance, token_dyn_importance, mean_dyn_importance]
    plt.bar(groups, importances)
    plt.ylabel('Average |SHAP value|')
    plt.title('Feature Group Importance Comparison')
    plt.tight_layout()
    plt.savefig(os.path.join(report_dir, "shap_group_comparison.png"))
    plt.close()

    # ------ Top 10 最重要特征（所有特征） ------
    mean_abs_shap_all = np.abs(shap_values).mean(axis=0)
    mean_abs_shap_all = np.squeeze(mean_abs_shap_all)
    top_10_indices = np.argsort(mean_abs_shap_all)[-10:][::-1]

    # 提取对应的SHAP值
    top_10_values = []
    for idx in top_10_indices:
        top_10_values.append(float(mean_abs_shap_all[idx]))
    top_10_values = np.array(top_10_values)

    plt.figure(figsize=(10, 6))
    y_pos = np.arange(10)
    plt.barh(y_pos, top_10_values)
    # 修改标签显示，避免过长
    labels = []
    for i in top_10_indices:
        name = feature_names[i]
        if len(name) > 20:
            name = name[:17] + "..."
        labels.append(f"{feature_symbols[i]}\n({name})")
    plt.yticks(y_pos, labels)
    plt.xlabel('Mean |SHAP value|')
    plt.title('Top 10 Most Important Features Overall')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(report_dir, "shap_top10_overall.png"), dpi=150)
    plt.close()

    print("SHAP归因分析可视化完成，已保存在评估目录。")

    # -------- 输出符号-英文-中文对照表 ---------
    table_lines = []
    table_lines.append("| 符号 | 英文名 | 中文解释 |")
    table_lines.append("|------|--------|----------|")
    for sym, name, zh in zip(feature_symbols, feature_names, feature_zh):
        table_lines.append(f"| {sym} | {name} | {zh} |")
    with open(os.path.join(report_dir, "feature_table.md"), "w", encoding="utf-8") as ftab:
        ftab.write('\n'.join(table_lines))
    print("特征对照表已保存为 feature_table.md")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=False, help="数据集json目录")
    parser.add_argument("--report_dir", type=str, required=False, help="评估报告目录")
    parser.set_defaults(
        data_dir="./outputs/test_experiment",
        report_dir="./shap_results/test_eval_report",
    )
    args = parser.parse_args()
    main(args.data_dir, args.report_dir)