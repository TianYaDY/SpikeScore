import os
import json
import pickle
from tqdm import tqdm
import torch
import numpy as np
from scipy.spatial.distance import jensenshannon
from transformers import AutoModelForCausalLM
import logging

# ===== 日志配置 =====
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

def get_unembedding_weight(model):
    if hasattr(model, "lm_head"):
        return model.lm_head.weight.detach()
    elif hasattr(model, "transformer") and hasattr(model.transformer, "lm_head"):
        return model.transformer.lm_head.weight.detach()
    else:
        raise ValueError("Cannot find unembedding weight in the model.")

def layerwise_probs(hidden, unembedding_weight, layer_range=None):
    # hidden: [n_layers+1, hidden]
    hidden = hidden.float()   # 新增：强制转换为float32
    if layer_range is not None:
        start, end = layer_range
        hidden = hidden[start:end+1]
    logits = hidden @ unembedding_weight.T  # [n_selected_layers, vocab_size]
    probs = torch.softmax(logits, dim=-1)
    return probs.cpu().numpy()


def jsd(p, q):
    return jensenshannon(p, q, base=2) ** 2

def compute_token_reasoning_score(token_hidden, unembedding_weight, layer_range=None, log_context=""):
    try:
        probs = layerwise_probs(token_hidden, unembedding_weight, layer_range)
        final_prob = probs[-1]
        jsd_scores = [jsd(probs[j], final_prob) for j in range(probs.shape[0]-1)]
        if len(jsd_scores) == 0:
            logger.warning(f"{log_context} 只选了一层，无法计算JSD。")
            return None
        return float(np.mean(jsd_scores))
    except Exception as e:
        logger.error(f"{log_context} 计算推理分数时异常: {e}")
        return None

def metric_on_folder(model_path, data_dir, layer_range=(13, 21)):
    logger.info(f"加载模型: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cpu", torch_dtype=torch.float32)
    model.eval()
    unembedding_weight = get_unembedding_weight(model)
    logger.info(f"开始遍历目录: {data_dir}")

    files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
    for f in tqdm(files, desc="Metric计算中"):
        basename = os.path.splitext(f)[0]
        json_path = os.path.join(data_dir, f"{basename}.json")
        pkl_path = os.path.join(data_dir, f"{basename}.pkl")
        if not os.path.exists(pkl_path):
            logger.warning(f"[{basename}] 缺少对应pkl，跳过。")
            continue

        # 读取json和pkl
        try:
            with open(json_path, "r", encoding="utf-8") as fj:
                data = json.load(fj)
            with open(pkl_path, "rb") as fp:
                all_hidden_states = pickle.load(fp)
        except Exception as e:
            logger.error(f"[{basename}] 读取json/pkl失败: {e}")
            continue

        steps = data.get("reasoning_chain", [])
        steps_changed = False

        for step_idx, step_info in enumerate(steps):
            metrics = step_info.get("metrics", {})
            step_hidden_pack = all_hidden_states.get(step_idx, None)
            if step_hidden_pack is None:
                logger.warning(f"[{basename}][step {step_idx}] 无隐藏层信息，metrics全部置为None。")
                metrics["reasoning_score_per_token"] = {
                    "last_token_embedding": None,
                    "sec_last_token_embedding": None,
                    "last_tok_bef_gen_embedding": None
                }
                step_info["metrics"] = metrics
                steps_changed = True
                continue

            token_keys = [
                "last_token_embedding",
                "sec_last_token_embedding",
                "last_tok_bef_gen_embedding"
            ]
            token_scores = {}
            for key in token_keys:
                token_hidden = step_hidden_pack.get(key, None)
                log_context = f"[{basename}][step {step_idx}][{key}]"
                if token_hidden is None:
                    logger.warning(f"{log_context} 隐藏状态缺失。")
                    token_scores[key] = None
                    continue
                if not isinstance(token_hidden, torch.Tensor):
                    logger.warning(f"{log_context} 类型异常，实际类型: {type(token_hidden)}。")
                    token_scores[key] = None
                    continue
                if len(token_hidden.shape) != 2:
                    logger.warning(f"{log_context} 形状异常，实际shape: {token_hidden.shape}。")
                    token_scores[key] = None
                    continue
                score = compute_token_reasoning_score(token_hidden, unembedding_weight, layer_range, log_context)
                if score is None:
                    logger.warning(f"{log_context} 推理分数为None。")
                token_scores[key] = score
            metrics["reasoning_score_per_token"] = token_scores

            step_info["metrics"] = metrics
            steps_changed = True

        # 若有更改则写回
        if steps_changed:
            try:
                with open(json_path, "w", encoding="utf-8") as fj:
                    json.dump(data, fj, ensure_ascii=False, indent=2)
                logger.info(f"[{basename}] 已写回更新的metrics。")
            except Exception as e:
                logger.error(f"[{basename}] 写回json失败: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=False, help="HF模型目录")
    parser.add_argument("--data_dir", type=str, required=False, help="数据文件夹（含json/pkl）")
    parser.add_argument("--layer_range", type=str, default="13,21", help="层范围，如13,21（包含端点）")

    # ======= 明文测试参数（只需修改此处） =======
    parser.set_defaults(
        model_path = "../models/Llama-3.2-3B-Instruct",
        data_dir = "./outputs/svamp_experiment",
        layer_range = "13,21"
    )
    # ======= END =======

    args = parser.parse_args()
    layer_range = tuple(int(x) for x in args.layer_range.split(","))

    metric_on_folder(args.model_path, args.data_dir, layer_range)
