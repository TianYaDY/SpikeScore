import pickle
import os
import numpy as np


def pretty_print(obj, max_depth=2, cur_depth=0, max_items=5, prefix=""):
    if cur_depth > max_depth:
        print(f"{prefix}... (depth limit reached)")
        return
    if isinstance(obj, dict):
        print(f"{prefix}dict with {len(obj)} keys:")
        for i, (k, v) in enumerate(obj.items()):
            if i >= max_items:
                print(f"{prefix}  ... ({len(obj) - max_items} more keys)")
                break
            print(f"{prefix}  [{repr(k)}]: ", end="")
            pretty_print(v, max_depth, cur_depth + 1, max_items, prefix + "    ")
    elif isinstance(obj, list):
        print(f"{prefix}list of len {len(obj)}")
        for i, item in enumerate(obj[:max_items]):
            print(f"{prefix}  [{i}]: ", end="")
            pretty_print(item, max_depth, cur_depth + 1, max_items, prefix + "    ")
        if len(obj) > max_items:
            print(f"{prefix}  ... ({len(obj) - max_items} more items)")
    elif isinstance(obj, tuple):
        print(f"{prefix}tuple of len {len(obj)}")
        for i, item in enumerate(obj[:max_items]):
            print(f"{prefix}  [{i}]: ", end="")
            pretty_print(item, max_depth, cur_depth + 1, max_items, prefix + "    ")
        if len(obj) > max_items:
            print(f"{prefix}  ... ({len(obj) - max_items} more items)")
    elif hasattr(obj, 'shape') and hasattr(obj, 'dtype'):
        print(f"{prefix}{type(obj).__name__} shape={obj.shape} dtype={obj.dtype}")
    else:
        try:
            summary = str(obj)
            if len(summary) > 120:
                summary = summary[:120] + "..."
            print(f"{prefix}{type(obj).__name__}: {summary}")
        except Exception as e:
            print(f"{prefix}{type(obj).__name__}: <unprintable>")


def analyze_entropy_values(entropy_list):
    """分析熵值的统计信息"""
    print("\n=== Semantic Entropy (cluster_assignment_entropy) Analysis ===")
    print(f"Total samples: {len(entropy_list)}")

    # 显示前10个值
    print("\nFirst 10 values:")
    for i, val in enumerate(entropy_list[:10]):
        print(f"  [{i}]: {val:.6f}")

    # 转换为numpy数组进行统计
    entropy_array = np.array(entropy_list)

    print("\nStatistical Summary:")
    print(f"  Mean: {np.mean(entropy_array):.6f}")
    print(f"  Std: {np.std(entropy_array):.6f}")
    print(f"  Min: {np.min(entropy_array):.6f}")
    print(f"  Max: {np.max(entropy_array):.6f}")
    print(f"  25th percentile: {np.percentile(entropy_array, 25):.6f}")
    print(f"  50th percentile (median): {np.percentile(entropy_array, 50):.6f}")
    print(f"  75th percentile: {np.percentile(entropy_array, 75):.6f}")

    # 显示分布
    print("\nValue Distribution:")
    bins = [0, 0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, np.inf]
    hist, _ = np.histogram(entropy_array, bins)
    for i in range(len(bins) - 1):
        count = hist[i]
        pct = 100 * count / len(entropy_array)
        if bins[i + 1] == np.inf:
            print(f"  [{bins[i]:.1f}, ∞): {count} samples ({pct:.1f}%)")
        else:
            print(f"  [{bins[i]:.1f}, {bins[i + 1]:.1f}): {count} samples ({pct:.1f}%)")

    # 显示一些特殊情况
    print("\nSpecial cases:")
    zero_count = np.sum(entropy_array == 0)
    print(f"  Exactly 0: {zero_count} samples ({100 * zero_count / len(entropy_array):.1f}%)")

    # 显示一些随机样本
    print("\n10 Random samples:")
    random_indices = np.random.choice(len(entropy_array), size=min(10, len(entropy_array)), replace=False)
    for idx in sorted(random_indices):
        print(f"  [{idx}]: {entropy_array[idx]:.6f}")


if __name__ == "__main__":
    # 路径
    file_path = "./semantic_uncertainty/xxxx.pkl"

    print(f"Loading {file_path}...")
    with open(file_path, "rb") as f:
        obj = pickle.load(f)

    print(f"Loaded successfully, type={type(obj).__name__}")
    print("\n=== File Structure ===")
    pretty_print(obj, max_depth=2, max_items=6)

    # 深入分析语义熵
    if isinstance(obj, dict) and 'uncertainty_measures' in obj:
        measures = obj['uncertainty_measures']
        if 'cluster_assignment_entropy' in measures:
            entropy_list = measures['cluster_assignment_entropy']
            analyze_entropy_values(entropy_list)
        else:
            print("\nNo 'cluster_assignment_entropy' found in uncertainty_measures")
            print("Available keys:", list(measures.keys()))
    else:
        print("\nUnexpected file structure")