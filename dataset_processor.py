import json
import hashlib
from pathlib import Path
from datasets import Dataset

def flatten_coqa(entry):
    story = entry.get("story", "")
    base_id = entry.get("id") or hashlib.md5(story.encode()).hexdigest()[:8]
    questions = entry.get("questions", [])
    answers = entry.get("answers", [])
    answer_map = {a["turn_id"]: a for a in answers if "turn_id" in a}
    samples = []
    for q in questions:
        turn_id = q.get("turn_id")
        q_text = q.get("input_text", "")
        ans = answer_map.get(turn_id, {})
        a_text = ans.get("input_text", "")
        new_id = f"{base_id}_{turn_id}" if turn_id is not None else base_id
        samples.append({
            "id": str(new_id),
            "question": str(q_text),
            "context": str(story),
            "answer": str(a_text),
            "turn_id": turn_id
        })
    return samples

def flatten_math(entry, idx=0):
    question = str(entry.get("en", ""))
    context = str(entry.get("solution", ""))  # solution 作为 context
    answer = str(entry.get("answer", ""))
    _id = entry.get("id", None)
    if not _id or str(_id).strip() == "":
        id_source = question + "|" + context + "|" + answer + "|" + str(idx)
        _id = hashlib.md5(id_source.encode("utf-8")).hexdigest()[:12]
    return {
        "id": str(_id),
        "question": question,
        "context": context,
        "answer": answer,
        "turn_id": None
    }

def flatten_svamp(entry, idx=0):
    question = str(entry.get("Question", ""))
    context = str(entry.get("Body", ""))
    answer = str(entry.get("Answer", ""))
    _id = entry.get("ID", None)
    if not _id or str(_id).strip() == "":
        id_source = question + "|" + context + "|" + answer + "|" + str(idx)
        _id = hashlib.md5(id_source.encode("utf-8")).hexdigest()[:12]
    return {
        "id": str(_id),
        "question": question,
        "context": context,
        "answer": answer,
        "turn_id": None
    }

def flatten_general(entry, idx=0):
    mapping = {
        "questions": "question", "query": "question", "prompt": "question", "input": "question",
        "knowledge": "context", "story": "context", "passage": "context", "text": "context",
        "answers": "answer", "response": "answer", "right_answer": "answer",
        "target": "answer", "output": "answer", "label": "answer"
    }
    sample = dict(entry)
    for src, tgt in mapping.items():
        if src in sample and tgt not in sample:
            sample[tgt] = sample[src]
    for key in ["question", "context", "answer"]:
        val = sample.get(key, "")
        sample[key] = "" if val is None else str(val)
    _id = sample.get("id", None)
    if not _id or str(_id).strip() == "":
        id_source = sample["question"] + "|" + sample["context"] + "|" + sample["answer"] + "|" + str(idx)
        _id = hashlib.md5(id_source.encode("utf-8")).hexdigest()[:12]
    sample["id"] = str(_id)
    return {
        "id": sample["id"],
        "question": sample["question"],
        "context": sample["context"],
        "answer": sample["answer"],
        "turn_id": sample.get("turn_id", None)
    }

def detect_dataset_type(raw):
    # COQA
    if isinstance(raw, list) and raw and all(("story" in item and "questions" in item and "answers" in item) for item in raw):
        return "coqa"
    # Math
    if isinstance(raw, list) and raw and all(("en" in item and "solution" in item and "answer" in item) for item in raw):
        return "math"
    # SVAMP
    if isinstance(raw, list) and raw and all(("Body" in item and "Question" in item and "Answer" in item) for item in raw):
        return "svamp"
    return "general"

def build_dataset(path_or_name, sampling=None, seed=42):
    path = str(path_or_name)
    ext = Path(path).suffix.lower()
    is_coqa = False
    raw = []

    # 兼容COQA原逻辑
    if "coqa" in path:
        is_coqa = True
    elif ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            temp_raw = json.load(f)
        if isinstance(temp_raw, dict) and "data" in temp_raw:
            temp_raw = temp_raw["data"]
        if isinstance(temp_raw, list) and temp_raw and all(("story" in item and "questions" in item and "answers" in item) for item in temp_raw):
            is_coqa = True
    # COQA
    if is_coqa:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, dict) and "data" in raw:
            raw = raw["data"]
        flat_samples = []
        for entry in raw:
            flat_samples.extend(flatten_coqa(entry))
        ds = Dataset.from_list(flat_samples)
    else:
        # 通用（含math/svamp等）
        if ext == ".jsonl":
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        raw.append(json.loads(line))
        elif ext == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict) and "data" in data:
                    data = data["data"]
                raw = data
        else:
            raise ValueError("只支持json/jsonl/COQA格式")
        ds_type = detect_dataset_type(raw)
        flat_samples = []
        if ds_type == "math":
            for idx, entry in enumerate(raw):
                flat_samples.append(flatten_math(entry, idx))
        elif ds_type == "svamp":
            for idx, entry in enumerate(raw):
                flat_samples.append(flatten_svamp(entry, idx))
        else:
            for idx, entry in enumerate(raw):
                flat_samples.append(flatten_general(entry, idx))
        ds = Dataset.from_list(flat_samples)
    real_seed = (sampling or {}).get("seed", seed)
    ds = _apply_sampling(ds, sampling, real_seed)
    return ds

def _apply_sampling(ds, sampling, seed=42):
    if not sampling or sampling.get("strategy", "all") == "all":
        return ds
    n = sampling.get("n", -1)
    if sampling["strategy"] == "random" and n > 0 and n < len(ds):
        return ds.shuffle(seed=seed).select(range(n))
    elif sampling["strategy"] == "sequential" and n > 0 and n < len(ds):
        return ds.select(range(n))
    else:
        return ds

# ====== 示例用法 ======
if __name__ == "__main__":
    # COQA
    coqa_ds = build_dataset("./datasets/coqa-train-v1.0.json", sampling={"strategy": "random", "n": 5})
    print("\nCOQA样本预览：")
    for i in range(3):
        print(coqa_ds[i])
    # Math 数据集
    math_ds = build_dataset("./datasets/math.jsonl", sampling={"strategy": "random", "n": 3})
    print("\nMath样本预览：")
    for i in range(3):
        print(math_ds[i])
    # SVAMP 数据集
    svamp_ds = build_dataset("./datasets/SVAMP.json", sampling={"strategy": "random", "n": 3})
    print("\nSVAMP样本预览：")
    for i in range(3):
        print(svamp_ds[i])
    # DefAn等普通QA
    defan_ds = build_dataset("./datasets/DefAn_public_combined.json", sampling={"strategy": "random", "n": 5})
    print("\nDefan样本预览：")
    for i in range(3):
        print(defan_ds[i])
