import os
import json
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

def hallucination_score(scores: np.ndarray, mode='peaked'):
    scores = np.array(scores, dtype=np.float32)
    scores = np.nan_to_num(scores, nan=0.0)
    step_mean = scores.mean(axis=1)
    if mode == 'vanilla':
        score = step_mean.mean() + 0.5 * step_mean.max()
        return float(np.tanh(score))
    elif mode == 'peaked':
        diffs = np.diff(step_mean)
        max_increase = np.max(diffs.clip(min=0)) if len(diffs) > 0 else 0
        score = 0.5 * step_mean.max() + 0.3 * max_increase + 0.2 * step_mean.mean()
        return float(np.tanh(score))
    elif mode == 'softmax':
        T = 3.0
        weights = np.exp(T * (step_mean - step_mean.mean()))
        weights /= weights.sum()
        agg = (weights * step_mean).sum()
        return float(np.tanh(agg))
    else:
        raise ValueError("mode not supported")

def evaluate_hallucination_auc(data_dir, report_dir, score_mode='peaked'):
    os.makedirs(report_dir, exist_ok=True)
    json_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
    y_true, y_score, file_ids = [], [], []

    for f in json_files:
        try:
            with open(os.path.join(data_dir, f), encoding='utf-8') as jf:
                data = json.load(jf)
            label = data.get('nli_label', None)
            if label not in (0, 1):  # 跳过无标签或标签异常
                continue

            steps = data.get('reasoning_chain', [])
            scores = []
            for step in steps:
                metrics = step.get('metrics', {})
                token_scores = metrics.get('reasoning_score_per_token', {})
                # 只考虑3个标准token（按保存顺序或显式key名，视你实际情况调整）
                per_token = []
                for key in ["last_token_embedding", "sec_last_token_embedding", "last_tok_bef_gen_embedding"]:
                    v = token_scores.get(key, None)
                    per_token.append(float(v) if v is not None else np.nan)
                scores.append(per_token)
            # 至少有一步有分数
            scores = np.array(scores)
            if scores.shape[0] == 0:
                continue
            score = hallucination_score(scores, mode=score_mode)

            y_true.append(label)
            y_score.append(score)
            file_ids.append(f)
        except Exception as e:
            print(f"Skip {f}: {e}")

    # 计算AUC
    auc = roc_auc_score(y_true, y_score)
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    print(f"AUC: {auc:.4f}")

    # 保存评估结果
    result_path = os.path.join(report_dir, "hallucination_auc.txt")
    with open(result_path, 'w', encoding='utf-8') as out:
        out.write(f"AUC: {auc:.4f}\n")
        out.write("ID\tLabel\tScore\n")
        for fid, label, score in zip(file_ids, y_true, y_score):
            out.write(f"{fid}\t{label}\t{score:.6f}\n")

    # 画图保存
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (AUC={auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve: Hallucination Score")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(report_dir, "roc_curve.png"))
    plt.close()

    # 分数分布直方图
    plt.figure()
    plt.hist([np.array(y_score)[np.array(y_true)==0], np.array(y_score)[np.array(y_true)==1]],
             bins=20, alpha=0.7, label=["Hallucination", "No Hallucination"])
    plt.xlabel("Hallucination Score")
    plt.ylabel("Sample Count")
    plt.legend()
    plt.title("Hallucination Score Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(report_dir, "score_dist.png"))
    plt.close()
    print(f"评估结果保存在：{report_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=False, help="数据集json目录")
    parser.add_argument("--report_dir", type=str, required=False, help="评估报告目录")
    parser.add_argument("--score_mode", type=str, default="peaked", choices=["vanilla", "peaked", "softmax"], help="分数聚合模式")

    # ==== 明文测试参数（无需命令行）====
    parser.set_defaults(
        data_dir = "./outputs/defan_experiment",  # <<-- 这里写你的默认数据目录
        report_dir = "./eval_results/defan_eval_report",  # <<-- 这里写你的默认输出目录
        score_mode = "peaked"
    )
    # ==== END ====

    args = parser.parse_args()
    evaluate_hallucination_auc(args.data_dir, args.report_dir, args.score_mode)
