import datetime
import json
import logging
import os
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# 保持自定义模块调用不变
from dataset_processor import build_dataset
from prompt_builder import create_prompt_builder
from strategy_library import get_strategy_library
from utils import judge_answer

# 日志配置，和原本一致
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('experiment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class CoTReasoner:
    """现代化Chain-of-Thought多步推理器，仅保留主流程、隐藏层保存与日志"""

    def __init__(self, config_path: str = "config.json"):
        self.config = self._load_config(config_path)
        self._setup_directories()
        self._initialize_model()
        self._setup_modules()

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
            config.setdefault("strategy", "progressive")
            config.setdefault("max_steps", 20)
            config.setdefault("temperature", 0.7)
            config.setdefault("output_dir", "outputs")
            return config

    def _setup_directories(self):
        os.makedirs(self.config["output_dir"], exist_ok=True)

    def _initialize_model(self):
        model_path = self.config["model_path"]
        logger.info(f"🚀 正在加载模型: {model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        self.model.eval()
        # 获取token限制（尽量自动）
        self.max_length = getattr(self.tokenizer, 'model_max_length', 4096)
        self.max_gen_tokens = self.config.get("max_generation_tokens", min(512, self.max_length // 5))
        logger.info(f"📊 Token配置: 总限制: {self.max_length} | 最大生成: {self.max_gen_tokens}")

    def _setup_modules(self):
        self.dataset_processor = build_dataset(
            self.config["dataset_path"],
            sampling=self.config.get("sampling", {"strategy": "all"}),
            # seed=self.config.get("random_seed", 42)
        )
        self.prompt_builder = create_prompt_builder(
            self.tokenizer,
            system_prompt=self.config.get("system_prompt")
        )
        self.strategy_library = get_strategy_library()
        self.strategy_mode = self.config.get("strategy", "progressive")
        strategy_info = self.strategy_library.get_strategy_info(self.strategy_mode)
        logger.info(f"📋 使用策略模式: {self.strategy_mode} - {strategy_info['name']}")

    def _save_hidden_states(self, hidden_dict, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(hidden_dict, f)

    def _save_result(self, file_path, result):
        temp_file = f"{file_path}.tmp"
        try:
            if os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                if 'nli_label' in existing and existing['nli_label'] is not None:
                    result['nli_label'] = existing['nli_label']
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            if os.path.exists(file_path):
                os.replace(temp_file, file_path)
            else:
                os.rename(temp_file, file_path)
        except Exception as e:
            logger.error(f"保存结果失败: {e}")
            if os.path.exists(temp_file):
                os.remove(temp_file)
            raise

    # def _extract_hidden_states(self, outputs):
    #     """
    #     outputs.hidden_states:
    #       - tuple: (total_steps = input+生成,)
    #         每个元素: tuple (n_layers+1, batch, hidden)
    #     返回:
    #       - last_token_embedding: 最后生成token的最后一层 (shape: [hidden])
    #       - sec_last_token_embedding: 倒数第二token的所有层 (shape: [n_layers+1, hidden])
    #       - last_tok_bef_gen_embedding: 输入最后一个token的所有层 (shape: [n_layers+1, hidden])
    #     """
    #     hidden_states = outputs.hidden_states
    #     if not hidden_states:
    #         return (None, None, None)
    #
    #     # # 1. 最后生成token的最后一层
    #     # last_step_hidden = hidden_states[-1]  # (n_layers+1, batch, hidden)
    #     # last_layer = last_step_hidden[-1]  # (batch, hidden)
    #     # last_token_embedding = last_layer[0].cpu()  # batch=1，shape:[hidden]
    #
    #     # 1. 最后生成token的所有层
    #     last_step_hidden = hidden_states[-1]  # (n_layers+1, batch, hidden)
    #     last_token_embedding = torch.stack([layer[0].cpu() for layer in last_step_hidden])  # [n_layers+1, hidden]
    #
    #     # 2. 倒数第二生成token的所有层
    #     if len(hidden_states) > 1:
    #         sec_last_step_hidden = hidden_states[-2]  # (n_layers+1, batch, hidden)
    #         sec_last_token_embedding = torch.stack(
    #             [layer[0].cpu() for layer in sec_last_step_hidden])  # [n_layers+1, hidden]
    #     else:
    #         sec_last_token_embedding = torch.stack([layer[0].cpu() for layer in last_step_hidden])
    #
    #     # 3. 输入最后一个token的所有层
    #     first_step_hidden = hidden_states[0]
    #     last_tok_bef_gen_embedding = torch.stack(
    #         [layer[0].cpu() for layer in first_step_hidden])  # [n_layers+1, hidden]
    #
    #     return (last_token_embedding, sec_last_token_embedding, last_tok_bef_gen_embedding)

    def _extract_hidden_states(self, outputs):
        """
        outputs.hidden_states:
          - tuple: (total_steps = input+生成,)
            每个元素: tuple (n_layers+1, batch, seq_len, hidden)
        返回:
          - last_token_embedding: 最后生成token的所有层 (shape: [n_layers+1, hidden])
          - sec_last_token_embedding: 倒数第二token的所有层 (shape: [n_layers+1, hidden])
          - last_tok_bef_gen_embedding: 输入最后一个token的所有层 (shape: [n_layers+1, hidden])
        """
        hidden_states = outputs.hidden_states
        if not hidden_states:
            return (None, None, None)

        # 1. 最后生成token的所有层
        last_step_hidden = hidden_states[
            -1]  # (n_layers+1, batch, seq_len, hidden) or (n_layers+1, batch, hidden) if only 1 token
        last_token_embedding = torch.stack(
            [layer[0, -1, :].cpu() for layer in last_step_hidden])  # [n_layers+1, hidden]

        # 2. 倒数第二生成token的所有层
        if len(hidden_states) > 1:
            sec_last_step_hidden = hidden_states[-2]
            sec_last_token_embedding = torch.stack([layer[0, -1, :].cpu() for layer in sec_last_step_hidden])
        else:
            sec_last_token_embedding = last_token_embedding.clone()

        # 3. 输入最后一个token的所有层
        first_step_hidden = hidden_states[0]
        last_tok_bef_gen_embedding = torch.stack([layer[0, -1, :].cpu() for layer in first_step_hidden])

        return (last_token_embedding, sec_last_token_embedding, last_tok_bef_gen_embedding)

    def _build_result_dict(self, data_item, chain, truncation_triggered, truncation_at_step):
        # 按原格式输出，但无metrics字段
        return {
            "id": data_item.get("id", "unknown"),
            "question": data_item.get("question", ""),
            "gold_answer": data_item.get("answer", ""),
            "nli_label": None,
            "context_preview": data_item.get("context", "")[:200] + "..." if len(data_item.get("context", "")) > 200 else data_item.get("context", ""),
            "dataset_type": self.config.get("dataset_type", "unknown"),
            "model": self.config["model_path"],
            "temperature": self.config.get("temperature", 0.7),
            "strategy": self.strategy_mode,
            "max_steps": self.config["max_steps"],
            "actual_steps": len(chain) - 1,
            "max_generation_tokens": self.max_gen_tokens,
            "max_input_tokens": self.max_length - self.max_gen_tokens,
            "truncation_triggered": truncation_triggered,
            "truncation_at_step": truncation_at_step,
            "generation_timeout": None,
            "last_update": datetime.datetime.now().isoformat(),
            "reasoning_chain": [
                {
                    "step": h['step'],
                    "prompt_type": h['prompt_type'],
                    "full_input": h['full_input'],
                    "followup_prompt": h['followup_prompt'],
                    "response": h['response'],
                    "timestamp": h['timestamp'],
                    "token_count": h['token_count'],
                    "response_length": len(h['response'].split()),
                    "input_length": len(h['full_input'].split()),
                    "metrics": {}  # 兼容旧结构
                } for h in chain
            ]
        }

    def _get_followup_prompt(self, step):
        context = {
            'step': step,
            'strategy_mode': self.strategy_mode,
        }
        prompt_text = self.strategy_library.get_prompt_for_context(context)
        return prompt_text, f"{self.strategy_mode}_step{step}"

    @torch.no_grad()
    def process_question(self, data_item: Dict[str, Any]) -> Dict[str, Any]:
        question_id = data_item.get("id", "unknown")
        question = data_item.get("question", "")

        logger.info(f"\n{'=' * 60}")
        logger.info(f"❓ 处理问题 {question_id}")
        logger.info(f"   问题: {question[:100]}...")
        logger.info(f"   策略: {self.strategy_mode}")

        safe_id = str(question_id).replace(':', '_').replace('/', '_')
        result_file = os.path.join(self.config["output_dir"], f"{safe_id}.json")
        hidden_states_file = os.path.join(self.config["output_dir"], f"{safe_id}.pkl")
        all_hidden_states = {}

        # 第0步
        initial_prompt = self.prompt_builder.build_prompt(data_item)
        logger.info(f"\n📝 初始提示:\n{initial_prompt[:200]}...\n")
        inputs = self.tokenizer(initial_prompt, return_tensors="pt", truncation=True, max_length=self.max_length).to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_gen_tokens,
            temperature=self.config.get("temperature", 0.7),
            do_sample=(self.config.get("temperature", 0.7) > 0),
            pad_token_id=self.tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_hidden_states=True
        )
        # 解码
        generated_ids = outputs.sequences[0, inputs.input_ids.shape[1]:]
        qa_answer = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        initial_hidden = self._extract_hidden_states(outputs)

        all_hidden_states[0] = {
            'last_token_embedding': initial_hidden[0],
            'sec_last_token_embedding': initial_hidden[1],
            'last_tok_bef_gen_embedding': initial_hidden[2]
        }
        with open(hidden_states_file, 'wb') as f:
            pickle.dump(all_hidden_states, f)

        logger.info(f"💬 初始回答:\n{qa_answer[:200]}...\n")

        # 记录链
        chain = [{
            'step': 0,
            'prompt_type': "initial_qa",
            'full_input': initial_prompt,
            'followup_prompt': "",
            'response': qa_answer,
            'timestamp': datetime.datetime.now().isoformat(),
            'token_count': len(self.tokenizer.encode(initial_prompt + qa_answer))
        }]

        # 保存初始结果
        result = self._build_result_dict(
            data_item, chain, truncation_triggered=False, truncation_at_step=None
        )
        self._save_result(result_file, result)
        logger.info(f"💾 已保存步骤 0 的结果")

        # 异步NLI
        gold_answer = data_item.get("answer", "")
        async_nli_label(result_file, gold_answer, qa_answer, question, timeout=10)

        truncation_triggered = False
        truncation_at_step = None

        for step in range(1, self.config["max_steps"] + 1):
            logger.info(f"\n{'-' * 40}")
            logger.info(f"🔄 步骤 {step}/{self.config['max_steps']}")
            current_prompt_text, prompt_type = self._get_followup_prompt(step)
            logger.info(f"📝 使用提示: {current_prompt_text}")

            # 构建对话历史（只用已有chain的用户与助手轮流对话）
            messages = [
                {"role": "user", "content": data_item.get("question", "")},
                {"role": "assistant", "content": chain[0]['response']}
            ]
            for h in chain[1:]:
                messages.append({"role": "user", "content": h['followup_prompt']})
                messages.append({"role": "assistant", "content": h['response']})
            messages.append({"role": "user", "content": current_prompt_text})

            if hasattr(self.tokenizer, 'apply_chat_template'):
                enable_thinking = self.config.get("enable_thinking", True)
                conversation = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=enable_thinking
                )
            else:
                conversation = ""
                for msg in messages:
                    if msg["role"] == "user":
                        conversation += f"\nUser: {msg['content']}"
                    else:
                        conversation += f"\nAssistant: {msg['content']}"
                conversation += "\nAssistant:"

            inputs = self.tokenizer(conversation, return_tensors="pt", truncation=True, max_length=self.max_length).to(self.model.device)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_gen_tokens,
                temperature=self.config.get("temperature", 0.7),
                do_sample=(self.config.get("temperature", 0.7) > 0),
                pad_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_hidden_states=True
            )
            generated_ids = outputs.sequences[0, inputs.input_ids.shape[1]:]
            cot_answer = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            step_hidden = self._extract_hidden_states(outputs)

            all_hidden_states[step] = {
                'last_token_embedding': step_hidden[0],
                'sec_last_token_embedding': step_hidden[1],
                'last_tok_bef_gen_embedding': step_hidden[2]
            }
            with open(hidden_states_file, 'wb') as f:
                pickle.dump(all_hidden_states, f)

            logger.info(f"💬 生成回答 (长度: {len(cot_answer.split())} words)")

            step_record = {
                'step': step,
                'prompt_type': prompt_type,
                'full_input': conversation,
                'followup_prompt': current_prompt_text,
                'response': cot_answer,
                'timestamp': datetime.datetime.now().isoformat(),
                'token_count': len(self.tokenizer.encode(f"{current_prompt_text}\n{cot_answer}"))
            }
            chain.append(step_record)

            # 保存
            result = self._build_result_dict(
                data_item, chain, truncation_triggered, truncation_at_step
            )
            self._save_result(result_file, result)
            logger.info(f"💾 已保存步骤 {step} 的结果")

        logger.info(f"\n✅ 问题 {question_id} 处理完成，共 {len(chain)} 步")
        logger.info(f"💾 隐藏层数据已保存至: {hidden_states_file}")
        return result

    def process_dataset(self):
        logger.info(f"📚 处理 {self.config.get('dataset_type', 'unknown')} 数据集")
        logger.info(f"📊 数据集统计: 总条数: {len(self.dataset_processor)} | 字段: {self.dataset_processor.column_names}")

        total_items = len(self.dataset_processor)
        progress_file = os.path.join(self.config["output_dir"], "progress.json")
        for idx, data_item in enumerate(tqdm(self.dataset_processor, desc="处理问题")):
            try:
                logger.info(f"\n{'=' * 60}")
                logger.info(f"处理进度: {idx + 1}/{total_items}")
                progress = {
                    "current": idx + 1,
                    "total": total_items,
                    "percentage": (idx + 1) / total_items * 100,
                    "current_item": data_item.get("id", f"item_{idx}"),
                    "timestamp": datetime.datetime.now().isoformat()
                }
                with open(progress_file, "w") as f:
                    json.dump(progress, f, indent=2)

                self.process_question(data_item)

            except Exception as e:
                logger.error(f"❌ 处理问题 {data_item.get('id', 'unknown')} 失败: {e}")
                import traceback
                traceback.print_exc()
                error_file = os.path.join(
                    self.config["output_dir"],
                    f"{str(data_item.get('id', f'item_{idx}')).replace(':', '_').replace('/', '_')}_error.json"
                )
                error_info = {
                    "id": data_item.get("id", f"item_{idx}"),
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                    "timestamp": datetime.datetime.now().isoformat()
                }
                with open(error_file, "w") as f:
                    json.dump(error_info, f, indent=2)
                continue

    def run(self):
        logger.info(f"🚀 开始Chain-of-Thought实验")
        logger.info(f"   模型: {self.config['model_path']}")
        logger.info(f"   数据集: {self.config.get('dataset_type', 'unknown')}")
        logger.info(f"   策略: {self.strategy_mode}")
        start_time = datetime.datetime.now()
        try:
            self.process_dataset()
        except KeyboardInterrupt:
            logger.info("\n⚠️ 用户中断执行")
        except Exception as e:
            logger.error(f"❌ 处理失败: {e}")
            raise
        finally:
            duration = datetime.datetime.now() - start_time
            logger.info(f"\n✨ 实验完成！总用时: {duration}")


def async_nli_label(json_path, gold_answer, model_answer, question, timeout=10):
    import threading
    def nli_label_worker(json_path, gold_answer, model_answer, question):
        label = None
        try:
            label = judge_answer(gold_answer, model_answer, question)
        except Exception as e:
            logger.error(f"NLI判断失败: {e}")
        try:
            if os.path.exists(json_path):
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            else:
                data = {}
            if data.get('nli_label', None) != label:
                data['nli_label'] = label
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"写入nli_label失败: {e}")
    thread = threading.Thread(
        target=nli_label_worker,
        args=(json_path, gold_answer, model_answer, question)
    )
    thread.daemon = True
    thread.start()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Modern Chain-of-Thought Reasoning Experiment')
    parser.add_argument('--config', type=str, default='config.json', help='配置文件路径')
    args = parser.parse_args()
    try:
        reasoner = CoTReasoner(config_path=args.config)
        reasoner.run()
    except Exception as e:
        logger.error(f"❌ 程序执行失败: {e}")
        raise


if __name__ == "__main__":
    main()
