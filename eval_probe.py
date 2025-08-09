import os
import json
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

def evaluate_entropy_auc(data_dir, report_dir):
    os.makedirs(report_dir, exist_ok=True)
    json_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
    y_true, y_score, file_ids = [], [], []

    for f in json_files:
        try:
            with open(os.path.join(data_dir, f), encoding='utf-8') as jf:
                data = json.load(jf)
            label = data.get('nli_label', None)
            if label not in (0, 1):
                continue
            steps = data.get('reasoning_chain', [])


            entropies = []
            for step in steps:
                metrics = step.get('metrics', {})
                entropy = metrics.get('probe_pred', None)
                # 支持浮点/字符串/None
                if entropy is not None:
                    try:
                        entropies.append(float(entropy))
                    except:
                        entropies.append(0.0)
            if not entropies:
                continue


            delta = np.diff(entropies)  # 一阶差分
            delta2 = np.diff(delta)  # 二阶差分
            score = np.sum(np.abs(delta2))  # 所有加速度绝对值之和
            score = float(np.var(entropies)) + score



            y_true.append(label)
            y_score.append(score)
            file_ids.append(f)
        except Exception as e:
            print(f"Skip {f}: {e}")

    if len(y_true) < 2 or len(set(y_true)) < 2:
        print("AUC无法计算，正负样本不足！")
        return

    # 计算AUC
    auc = roc_auc_score(y_true, y_score)
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    print(f"AUC: {auc:.4f}")

    # 保存评估结果
    result_path = os.path.join(report_dir, "entropy_auc.txt")
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
    plt.title("ROC Curve: Semantic Entropy (Sum per sample)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(report_dir, "roc_curve.png"))
    plt.close()

    # 分数分布直方图
    plt.figure()
    plt.hist([np.array(y_score)[np.array(y_true)==0], np.array(y_score)[np.array(y_true)==1]],
             bins=20, alpha=0.7, label=["Hallucination", "No Hallucination"])
    plt.xlabel("Sum Semantic Entropy")
    plt.ylabel("Sample Count")
    plt.legend()
    plt.title("Semantic Entropy Score Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(report_dir, "score_dist.png"))
    plt.close()
    print(f"评估结果保存在：{report_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=False, help="数据集json目录")
    parser.add_argument("--report_dir", type=str, required=False, help="评估报告目录")

    # ==== 明文测试参数 ====
    parser.set_defaults(
        data_dir = "./outputs/xxx",
        report_dir = "./eval_results/xxx"
    )
    # ==== END ====

    args = parser.parse_args()
    evaluate_entropy_auc(args.data_dir, args.report_dir)
