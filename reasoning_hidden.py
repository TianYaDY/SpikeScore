"""
Chain-of-Thought 实验主程序
串联所有模块，执行完整的实验流程
"""
import datetime
import json
import logging
import math
import os
import pickle
import queue
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# 导入自定义模块
from dataset_processor import build_dataset
from prompt_builder import create_prompt_builder
from strategy_library import get_strategy_library
from utils import judge_answer

torch.set_float32_matmul_precision('high')


# def write_label_to_json(json_path, label):
#     """增量写入nli_label字段"""
#     try:
#         with open(json_path, "r", encoding="utf-8") as f:
#             data = json.load(f)
#         data['nli_label'] = label
#         with open(json_path, "w", encoding="utf-8") as f:
#             json.dump(data, f, ensure_ascii=False, indent=2)
#     except Exception as e:
#         logger.error(f"写入nli_label失败: {e}")

def write_label_to_json(json_path, label):
    """增量写入nli_label字段（保留其他字段）"""
    try:
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = {}
        # 只有在label发生改变时才写
        if data.get('nli_label', None) != label:
            data['nli_label'] = label
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"写入nli_label失败: {e}")


# def nli_label_worker(json_path, gold_answer, model_answer, timeout=10):
#     """新线程内做判断，超时则写None"""
#     import concurrent.futures
#     label = None
#     try:
#         with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
#             future = executor.submit(judge_answer, gold_answer, model_answer)
#             label = future.result(timeout=timeout)
#     except Exception as e:
#         logger.error(f"NLI判断超时或失败: {e}")
#         label = None
#     write_label_to_json(json_path, label)

def nli_label_worker(json_path, gold_answer, model_answer, question, timeout=10):
    import concurrent.futures
    label = None
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(judge_answer, gold_answer, model_answer, question)
            label = future.result(timeout=timeout)
    except Exception as e:
        logger.error(f"NLI判断超时或失败: {e}")
        label = None
    write_label_to_json(json_path, label)


# def async_nli_label(json_path, gold_answer, model_answer, timeout=10):
#     thread = threading.Thread(
#         target=nli_label_worker,
#         args=(json_path, gold_answer, model_answer, timeout)
#     )
#     thread.daemon = True
#     thread.start()
def async_nli_label(json_path, gold_answer, model_answer, question, timeout=10):
    thread = threading.Thread(
        target=nli_label_worker,
        args=(json_path, gold_answer, model_answer, question, timeout)
    )
    thread.daemon = True
    thread.start()


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('experiment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class StepRecord:
    """推理步骤记录"""
    step: int
    prompt: str  # 完整输入
    response: str
    timestamp: str
    token_count: int = 0
    prompt_type: str = ""
    full_input: str = ""  # 模型看到的完整输入
    followup_prompt: str = ""  # 仅跟进提示部分（纯文本）
    metrics: Dict[str, Any] = field(default_factory=dict)  # 各种指标


class CustomLLM:
    """自定义的LLM包装器，支持PPL计算和超时处理"""

    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = config.get("max_generation_tokens", 1024)
        self.temperature = config.get("temperature", 0.6)
        self.do_sample = self.temperature > 0
        self.generation_timeout = config.get("generation_timeout", 300)  # 默认5分钟超时
        self.top_p = config.get("top_p", 0.95)

    def invoke(self, prompt: str) -> str:
        """基础生成方法（保持兼容性）"""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                top_p=self.top_p,
            )

        generated_ids = outputs[0][inputs.input_ids.shape[1]:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)

    def _generate_with_timeout(self, inputs, result_queue, error_queue):
        """在单独线程中执行生成"""
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    do_sample=self.do_sample,
                    return_dict_in_generate=True,
                    output_scores=True,
                    output_hidden_states=True,  # 添加这一行以获取隐藏层
                    pad_token_id=self.tokenizer.eos_token_id
                )
            result_queue.put(outputs)
        except Exception as e:
            error_queue.put(e)

    def _extract_hidden_states(self, outputs, n_input_token, n_generated):
        """提取三个关键位置的隐藏层状态"""
        # 获取隐藏层
        if hasattr(outputs, 'decoder_hidden_states') and outputs.decoder_hidden_states is not None:
            hidden = outputs.decoder_hidden_states  # For encoder-decoder
        elif hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            hidden = outputs.hidden_states  # For decoder-only
        else:
            return (None, None, None)

        # 处理n_generated为0的情况
        if n_generated == 0:
            logger.warning("Only stop_words were generated. For likelihoods and embeddings, taking stop word instead.")
            n_generated = 1

        # 1. last_token_embedding - 生成部分最后一个token的最后一层
        if len(hidden) == 1:
            last_input = hidden[0]
        elif (n_generated - 1) >= len(hidden):
            logger.error('Taking last state because n_generated is too large')
            last_input = hidden[-1]
        else:
            last_input = hidden[n_generated - 1]

        last_layer = last_input[-1]  # 最后一层
        last_token_embedding = last_layer[:, -1, :].cpu()

        # 2. sec_last_token_embedding - 生成部分倒数第二个token的所有层
        if len(hidden) == 1:
            sec_last_input = hidden[0]
        elif ((n_generated - 2) >= len(hidden)):
            sec_last_input = hidden[-2] if len(hidden) >= 2 else hidden[-1]
        else:
            sec_last_input = hidden[n_generated - 2]
        sec_last_token_embedding = torch.stack([layer[:, -1, :] for layer in sec_last_input]).cpu()

        # 3. last_tok_bef_gen_embedding - 生成前最后一个token的所有层
        last_tok_bef_gen_input = hidden[0]  # 第一个生成步的hidden包含输入的信息
        last_tok_bef_gen_embedding = torch.stack([layer[:, -1, :] for layer in last_tok_bef_gen_input]).cpu()

        return (last_token_embedding, sec_last_token_embedding, last_tok_bef_gen_embedding)

    def invoke_with_metrics(self, prompt: str) -> Tuple[str, Dict[str, Any], Tuple]:
        """生成文本并计算条件PPL和其他指标，支持超时，返回隐藏层"""
        device = self.model.device
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
        input_length = inputs.input_ids.shape[1]

        # 使用队列传递结果
        result_queue = queue.Queue()
        error_queue = queue.Queue()

        # 记录开始时间
        start_time = datetime.datetime.now()

        # 在单独线程中执行生成
        generation_thread = threading.Thread(
            target=self._generate_with_timeout,
            args=(inputs, result_queue, error_queue)
        )
        generation_thread.daemon = True
        generation_thread.start()

        # 等待生成完成或超时
        generation_thread.join(timeout=self.generation_timeout)

        if generation_thread.is_alive():
            # 超时处理
            logger.warning(f"⚠️ 生成超时！已等待 {self.generation_timeout} 秒")

            # 返回超时响应
            timeout_text = "[Generation timed out]"
            timeout_metrics = {
                "conditional_ppl": float('inf'),
                "num_tokens": 0,
                "avg_log_prob": None,
                "avg_entropy": None,
                "avg_top1_prob": None,
                "min_top1_prob": None,
                "low_conf_ratio": 1.0,
                "generation_time": self.generation_timeout,
                "timeout": True
            }
            timeout_hidden = (None, None, None)
            return timeout_text, timeout_metrics, timeout_hidden

        # 检查是否有错误
        if not error_queue.empty():
            error = error_queue.get()
            logger.error(f"生成过程出错: {error}")
            raise error

        # 获取生成结果
        if result_queue.empty():
            raise RuntimeError("生成完成但未获得结果")

        outputs = result_queue.get()

        # 计算生成时间
        generation_time = (datetime.datetime.now() - start_time).total_seconds()

        # 获取生成的token和scores
        generated_ids = outputs.sequences[0][input_length:]
        scores = torch.stack(outputs.scores, dim=1)[0]  # [num_generated_tokens, vocab_size]

        # 计算条件PPL
        log_probs = []
        entropies = []
        top1_probs = []

        for i, token_id in enumerate(generated_ids):
            if i < len(scores):  # 确保索引有效
                # 计算log softmax
                log_softmax_scores = torch.log_softmax(scores[i], dim=-1)
                softmax_scores = torch.softmax(scores[i], dim=-1)

                # Token的log概率
                log_prob = log_softmax_scores[token_id]
                log_probs.append(log_prob.item())

                # 计算熵
                mask = softmax_scores > 1e-8
                entropy = -torch.sum(softmax_scores[mask] * log_softmax_scores[mask]) if mask.any() else 0.0
                entropies.append(entropy.item())

                # Top-1概率
                top1_prob = softmax_scores.max().item()
                top1_probs.append(top1_prob)

        # 计算PPL
        if log_probs:
            avg_log_prob = sum(log_probs) / len(log_probs)
            ppl = math.exp(-avg_log_prob)
        else:
            ppl = float('inf')

        # 解码文本
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # 提取隐藏层
        n_generated = len(generated_ids)
        hidden_states = self._extract_hidden_states(outputs, input_length, n_generated)

        # 计算其他统计量
        metrics = {
            "conditional_ppl": ppl,
            "num_tokens": len(generated_ids),
            "avg_log_prob": avg_log_prob if log_probs else None,
            "avg_entropy": sum(entropies) / len(entropies) if entropies else None,
            "avg_top1_prob": sum(top1_probs) / len(top1_probs) if top1_probs else None,
            "min_top1_prob": min(top1_probs) if top1_probs else None,
            "low_conf_ratio": sum(1 for p in top1_probs if p < 0.5) / len(top1_probs) if top1_probs else 0,
            "generation_time": generation_time,
            "timeout": False
        }

        # 如果生成时间超过30秒，记录警告
        if generation_time > 30:
            logger.warning(f"⏱️ 生成耗时较长: {generation_time:.1f} 秒")

        return generated_text, metrics, hidden_states


class SmartTruncationManager:
    """智能截断管理器"""

    def __init__(self, tokenizer, max_input_length: int):
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.is_truncated = False
        self.max_recent_steps = None
        self.recent_steps: List[StepRecord] = []
        self.first_step_tokens = 0

    def add_step(self, step_record: StepRecord, first_step_text: str) -> None:
        """添加新的推理步骤"""
        if not self.is_truncated:
            # 未触发截断，无限增长
            self.recent_steps.append(step_record)

            # 检查是否需要触发截断
            total_tokens = self._calculate_total_tokens(first_step_text)
            if total_tokens > self.max_input_length:
                logger.info(f"触发智能截断：当前tokens: {total_tokens}, 上限: {self.max_input_length}")
                self.is_truncated = True
                # 计算能容纳的最大步数（当前步数减1）
                self.max_recent_steps = len(self.recent_steps) - 1
                logger.info(f"截断后保持最近 {self.max_recent_steps} 步")
                if self.max_recent_steps > 0:
                    # 移除最旧的步骤
                    removed = self.recent_steps.pop(0)
                    logger.debug(f"移除步骤 {removed.step}")
        else:
            # 已触发截断，保持定长（栈行为）
            self.recent_steps.append(step_record)
            if len(self.recent_steps) > self.max_recent_steps:
                removed = self.recent_steps.pop(0)  # 移除最旧的
                logger.debug(f"栈溢出，移除步骤 {removed.step}")

    def _calculate_total_tokens(self, first_step_text: str) -> int:
        """计算总token数"""
        total = len(self.tokenizer.encode(first_step_text))
        for step in self.recent_steps:
            total += step.token_count
        return total

    def get_truncated_steps(self) -> List[StepRecord]:
        """获取截断后的步骤列表"""
        return self.recent_steps


class CoTReasoner:
    """链式思考推理器 - 使用新模块"""

    def __init__(self, config_path: str = "config.json"):
        self.config = self._load_config(config_path)
        self._setup_directories()
        self._initialize_model()
        self._setup_modules()

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(config_path, encoding="utf-8") as f:
                config = json.load(f)
                # 确保包含必要的配置项
                config.setdefault("strategy", "progressive")  # 单一策略字段
                config.setdefault("enable_ppl_detection", True)
                config.setdefault("enable_early_stopping", False)
                config.setdefault("ppl_threshold", 100)
                config.setdefault("generation_timeout", 300)
                config.setdefault("max_steps", 20)
                config.setdefault("temperature", 0.7)
                return config
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            raise

    def _setup_directories(self) -> None:
        """创建输出目录"""
        os.makedirs(self.config["output_dir"], exist_ok=True)

    def _initialize_model(self) -> None:
        """初始化模型和分词器"""
        model_path = self.config["model_path"]
        logger.info(f"🚀 正在加载模型: {model_path}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                local_files_only=True
            )
            # self.bnb_config = BitsAndBytesConfig(
            #     load_in_4bit=True,
            #     bnb_4bit_quant_type="nf4",  # 你可以自定义量化类型
            #     bnb_4bit_use_double_quant=True,
            #     bnb_4bit_compute_dtype=torch.bfloat16
            # )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                local_files_only=True,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                # quantization_config=self.bnb_config,
            )

            # 启用 PyTorch 编译（提高推理速度）
            self.model = torch.compile(self.model)

            # 获取token限制
            total_tokens = self._get_safe_token_limit()

            # 保持原有的计算逻辑
            self.max_generation_tokens = (total_tokens // 5) - 5
            self.max_input_length = total_tokens - self.max_generation_tokens

            logger.info(f"📊 Token配置:")
            logger.info(f"   - 模型总限制: {total_tokens}")
            logger.info(f"   - 最大生成: {self.max_generation_tokens}")
            logger.info(f"   - 最大输入: {self.max_input_length}")

            # 创建自定义LLM包装器
            llm_config = {
                "max_generation_tokens": self.max_generation_tokens,
                "temperature": self.config.get("temperature", 0.7),
                "generation_timeout": self.config.get("generation_timeout", 300)
            }
            self.llm = CustomLLM(self.model, self.tokenizer, llm_config)
            # ======= 关闭梯度检查点 =======
            if hasattr(self.model, "gradient_checkpointing_disable"):
                self.model.gradient_checkpointing_disable()
            # ============================

        except Exception as e:
            logger.error(f"模型初始化失败: {e}")
            raise

    def _get_safe_token_limit(self) -> int:
        """安全地获取模型的token限制"""
        # 1. 优先使用配置文件中的值
        if "model_max_tokens" in self.config:
            return self.config["model_max_tokens"]

        # 2. 尝试从tokenizer获取
        tokenizer_limit = getattr(self.tokenizer, 'model_max_length', None)

        # 3. 检查是否为合理值（小于10万）
        if tokenizer_limit and tokenizer_limit < 100000:
            return tokenizer_limit

        # 4. 值不合理，使用默认值
        logger.warning(f"⚠️ 检测到异常的token限制: {tokenizer_limit}")
        logger.info("使用通用默认token限制: 4096")
        return 4096

    def _setup_modules(self) -> None:
        """设置各个模块"""
        # 初始化数据集处理器
        dataset_config = {
            "dataset_path": self.config["dataset_path"],
            "sampling": self.config.get("sampling", {
                "strategy": "all"  # 默认使用全部数据
            })
        }

        # 添加随机种子（如果配置中有）
        if "random_seed" in self.config:
            dataset_config["sampling"]["seed"] = self.config["random_seed"]

        logger.info(f"📊 数据集采样配置: {dataset_config['sampling']}")

        # self.dataset_processor = build_dataset(dataset_config)
        self.dataset_processor = build_dataset(
            self.config["dataset_path"],
            sampling=self.config.get("sampling", {"strategy": "all"}),
            seed=self.config.get("random_seed", 42)
        )

        # 从 processor 获取检测到的类型
        # self.dataset_type = self.dataset_processor.dataset_type

        self.dataset_type = "huggingface"
        logger.info(f"📊 检测到数据集类型: {self.dataset_type}")

        # 初始化prompt构建器 - 传入tokenizer
        self.prompt_builder = create_prompt_builder(
            self.tokenizer,
            system_prompt=self.config.get("system_prompt")  # 可选的系统提示
        )

        # 初始化策略库 - 不需要传入tokenizer
        self.strategy_library = get_strategy_library()

        # 获取策略模式
        self.strategy_mode = self.config.get("strategy", "progressive")

        # 获取并显示策略信息
        strategy_info = self.strategy_library.get_strategy_info(self.strategy_mode)
        logger.info(f"📋 使用策略模式: {self.strategy_mode}")
        logger.info(f"   - 名称: {strategy_info['name']}")
        logger.info(f"   - 描述: {strategy_info['description']}")

        # 用于自适应策略
        self._last_response_length = 0

        # 保存原始问题（用于构建消息历史）
        self.original_question = None

    def get_followup_prompt(self, step: int) -> tuple[str, str]:
        """
        获取跟进提示文本
        返回: (纯文本prompt, prompt类型标识)
        """
        # 构建上下文信息
        context = {
            'step': step,
            'strategy_mode': self.strategy_mode,
            'last_response_length': self._last_response_length,
            # 可以添加更多上下文信息，如历史PPL等
        }

        # 从策略库获取纯文本prompt
        prompt_text = self.strategy_library.get_prompt_for_context(context)

        # 返回纯文本和标识
        return prompt_text, f"{self.strategy_mode}_step{step}"

    def build_conversation(self, data_item: Dict[str, Any], qa_response: str,
                           recent_steps: List[StepRecord], current_step: int) -> str:
        """构建对话 - 使用tokenizer的chat template"""

        # 构建消息列表
        messages = []

        # 添加初始问答
        messages.append({"role": "user", "content": data_item.get("question", "")})
        messages.append({"role": "assistant", "content": qa_response})

        # 添加历史对话
        for step_record in recent_steps:
            # followup_prompt是纯文本
            messages.append({"role": "user", "content": step_record.followup_prompt})
            messages.append({"role": "assistant", "content": step_record.response})

        # 添加当前步骤的提示
        current_prompt_text, _ = self.get_followup_prompt(current_step)
        messages.append({"role": "user", "content": current_prompt_text})

        # 使用tokenizer格式化整个对话
        if hasattr(self.tokenizer, 'apply_chat_template'):
            conversation = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # Fallback：简单拼接
            conversation = ""
            for msg in messages:
                if msg["role"] == "user":
                    conversation += f"\nUser: {msg['content']}"
                else:
                    conversation += f"\nAssistant: {msg['content']}"
            conversation += "\nAssistant:"

        return conversation

    def _calculate_quality_score(self, current_step: int, current_response: str,
                                 current_metrics: Dict[str, Any], history: List[StepRecord]) -> float:
        """
        计算质量分数（0-100，越高表示越差）
        使用对数变换处理无界值
        """
        # 基础分数组件
        components = []

        # 1. PPL分数（使用对数变换）
        if 'conditional_ppl' in current_metrics:
            ppl = current_metrics['conditional_ppl']
            # 对数变换：ln(1+ppl) * 10，上限约100
            ppl_score = min(100, np.log1p(ppl) * 10)
            components.append(ppl_score * 0.3)  # 30%权重

        # 2. 置信度分数
        if 'avg_top1_prob' in current_metrics:
            # confidence = current_metrics['avg_top1_prob']
            # # 低置信度得高分
            # conf_score = (1 - confidence) * 100
            # components.append(conf_score * 0.2)  # 20%权重
            confidence = current_metrics.get('avg_top1_prob')
            if confidence is not None:
                conf_score = (1 - confidence) * 100
                components.append(conf_score * 0.2)

        # 3. 重复度分数（只在有历史时计算）
        if history and current_step > 0:
            current_words = set(current_response.lower().split())

            # 计算与最近3步的重复度
            recent_words = set()
            for h in history[-3:]:
                recent_words.update(h.response.lower().split())

            if current_words and recent_words:
                overlap_ratio = len(current_words & recent_words) / len(current_words)
                repetition_score = overlap_ratio * 100
                components.append(repetition_score * 0.3)  # 30%权重

        # 4. 长度异常分数
        response_length = len(current_response.split())
        if response_length < 20:
            length_score = (1 - response_length / 20) * 100
        elif response_length > 500:
            # 对数变换处理超长响应
            length_score = min(100, np.log(response_length / 500) * 50)
        else:
            length_score = 0
        components.append(length_score * 0.2)  # 20%权重

        # 5. 超时惩罚
        if current_metrics.get('timeout', False):
            components.append(100)  # 超时直接加100分

        # 综合分数
        if components:
            quality_score = sum(components)
        else:
            quality_score = 0

        # 确保在0-100范围内
        return min(100, max(0, quality_score))

    def _should_early_stop(self, history: List[StepRecord]) -> bool:
        """基于metrics判断是否应该早停"""
        if len(history) < 3:
            return False

        # 检查最近的PPL趋势
        recent_ppls = []
        for record in history[-3:]:
            if 'conditional_ppl' in record.metrics:
                recent_ppls.append(record.metrics['conditional_ppl'])

        if len(recent_ppls) >= 3:
            # 如果PPL连续上升且最后一个超过阈值
            if all(recent_ppls[i] > recent_ppls[i - 1] for i in range(1, len(recent_ppls))):
                if recent_ppls[-1] > self.config.get("ppl_threshold", 100):
                    return True

        return False

    def process_question(self, data_item: Dict[str, Any]) -> Dict[str, Any]:
        """处理单个问题的完整推理链"""
        question_id = data_item.get("id", "unknown")
        question = data_item.get("question", "")

        logger.info(f"\n{'=' * 60}")
        logger.info(f"❓ 处理问题 {question_id}")
        logger.info(f"   问题: {question[:100]}...")
        logger.info(f"   策略: {self.strategy_mode}")
        logger.info(f"{'=' * 60}")

        # 初始化结果文件路径
        safe_id = str(question_id).replace(':', '_').replace('/', '_')
        result_file = os.path.join(self.config["output_dir"], f"{safe_id}.json")
        hidden_states_file = os.path.join(self.config["output_dir"], f"{safe_id}.pkl")

        # 初始化隐藏层数据字典
        all_hidden_states = {}

        # 第0步：使用prompt_builder构建初始QA
        initial_prompt = self.prompt_builder.build_prompt(data_item)
        logger.info(f"\n📝 初始提示:\n{initial_prompt[:200]}...\n")

        # 使用新的invoke方法
        if self.config.get("enable_ppl_detection", True) and hasattr(self.llm, 'invoke_with_metrics'):
            qa_answer, initial_metrics, initial_hidden = self.llm.invoke_with_metrics(initial_prompt)
            qa_answer = qa_answer.strip()
        else:
            qa_answer = self.llm.invoke(initial_prompt).strip()
            initial_metrics = {}
            initial_hidden = (None, None, None)

        # 保存第0步的隐藏层
        all_hidden_states[0] = {
            'last_token_embedding': initial_hidden[0],
            'sec_last_token_embedding': initial_hidden[1],
            'last_tok_bef_gen_embedding': initial_hidden[2]
        }

        # 保存隐藏层数据
        with open(hidden_states_file, 'wb') as f:
            pickle.dump(all_hidden_states, f)

        # 计算初始质量分数
        initial_quality = self._calculate_quality_score(0, qa_answer, initial_metrics, [])
        initial_metrics['quality_score'] = initial_quality

        logger.info(f"💬 初始回答:\n{qa_answer[:200]}...\n")

        # 如果有PPL，记录它
        if 'conditional_ppl' in initial_metrics:
            logger.info(f"📊 初始PPL: {initial_metrics['conditional_ppl']:.2f}")
        logger.info(f"📊 初始质量分数: {initial_quality:.2f}")

        # 记录生成时间
        if 'generation_time' in initial_metrics:
            logger.info(f"⏱️ 生成时间: {initial_metrics['generation_time']:.1f} 秒")

        initial_qa_text = f"{initial_prompt}\n{qa_answer}"

        # 初始化历史记录
        history = [StepRecord(
            step=0,
            prompt=initial_prompt,
            response=qa_answer,
            timestamp=datetime.datetime.now().isoformat(),
            token_count=len(self.tokenizer.encode(initial_prompt + qa_answer)),
            prompt_type="initial_qa",
            full_input=initial_prompt,
            followup_prompt="",
            metrics=initial_metrics  # 包含quality_score
        )]

        # 构建初始结果
        result = self._build_result_dict(
            data_item, history,
            truncation_triggered=False,
            truncation_at_step=None
        )

        # 保存初始结果
        self._save_result(result_file, result)
        logger.info(f"💾 已保存步骤 0 的结果")

        # ==== 新增：异步打nli_label ====
        gold_answer = data_item.get("answer", "")  # 假设数据标准答案在'answer'字段
        model_answer = qa_answer  # 模型生成的答案
        # async_nli_label(result_file, gold_answer, model_answer, timeout=10)
        async_nli_label(result_file, gold_answer, model_answer, question, timeout=10)
        # =============================

        # 初始化智能截断管理器
        truncation_manager = SmartTruncationManager(
            self.tokenizer,
            self.max_input_length
        )

        # 多步推理
        for step in range(1, self.config["max_steps"] + 1):
            logger.info(f"\n{'-' * 40}")
            logger.info(f"🔄 步骤 {step}/{self.config['max_steps']}")

            # 获取跟进提示（纯文本）
            current_prompt_text, prompt_type = self.get_followup_prompt(step)
            logger.info(f"📝 使用提示: {current_prompt_text}")

            # 构建对话（使用chat template）
            recent_steps = truncation_manager.get_truncated_steps()
            conversation = self.build_conversation(data_item, qa_answer, recent_steps, step)

            # 检查当前对话的token数
            current_tokens = len(self.tokenizer.encode(conversation))
            logger.info(f"📊 当前输入tokens: {current_tokens}/{self.max_input_length} "
                        f"({'%.1f' % (current_tokens / self.max_input_length * 100)}%)")

            logger.info(f"📝 使用提示类型: {prompt_type}")

            # 生成回答
            try:
                # 使用新的invoke方法
                if self.config.get("enable_ppl_detection", True) and hasattr(self.llm, 'invoke_with_metrics'):
                    cot_answer, step_metrics, step_hidden = self.llm.invoke_with_metrics(conversation)
                    cot_answer = cot_answer.strip()
                else:
                    cot_answer = self.llm.invoke(conversation).strip()
                    step_metrics = {}
                    step_hidden = (None, None, None)

                # 保存当前步的隐藏层
                all_hidden_states[step] = {
                    'last_token_embedding': step_hidden[0],
                    'sec_last_token_embedding': step_hidden[1],
                    'last_tok_bef_gen_embedding': step_hidden[2]
                }

                # 增量保存隐藏层数据
                with open(hidden_states_file, 'wb') as f:
                    pickle.dump(all_hidden_states, f)

                response_length = len(cot_answer.split())

                # 计算质量分数
                quality_score = self._calculate_quality_score(step, cot_answer, step_metrics, history)
                step_metrics['quality_score'] = quality_score

                logger.info(f"💬 生成回答 (长度: {response_length} words)")
                logger.debug(f"   预览: {cot_answer[:150]}...")

                # 记录PPL信息
                if 'conditional_ppl' in step_metrics:
                    logger.info(f"📊 步骤 {step} PPL: {step_metrics['conditional_ppl']:.2f}")

                logger.info(f"📊 步骤 {step} 质量分数: {quality_score:.2f}")

                # 记录生成时间
                if 'generation_time' in step_metrics:
                    logger.info(f"⏱️ 生成时间: {step_metrics['generation_time']:.1f} 秒")

                # 检查是否超时
                if step_metrics.get('timeout', False):
                    logger.error(f"❌ 步骤 {step} 生成超时")

                # 保存响应长度供自适应策略使用
                self._last_response_length = response_length

            except Exception as e:
                logger.error(f"❌ 步骤 {step} 生成失败: {e}")
                # 记录失败的步骤
                step_metrics = {
                    'error': str(e),
                    'quality_score': 100,  # 错误得最高分
                    'timeout': False
                }
                cot_answer = f"[Generation failed: {str(e)}]"
                response_length = 0
                self._last_response_length = 0

                # 异常时也保存None值
                all_hidden_states[step] = {
                    'last_token_embedding': None,
                    'sec_last_token_embedding': None,
                    'last_tok_bef_gen_embedding': None
                }
                with open(hidden_states_file, 'wb') as f:
                    pickle.dump(all_hidden_states, f)

            # 计算这一步的token数
            step_tokens = len(self.tokenizer.encode(f"{current_prompt_text}\n{cot_answer}"))

            # 创建步骤记录
            step_record = StepRecord(
                step=step,
                prompt=conversation,
                response=cot_answer,
                timestamp=datetime.datetime.now().isoformat(),
                token_count=step_tokens,
                prompt_type=prompt_type,
                full_input=conversation,
                followup_prompt=current_prompt_text,  # 纯文本
                metrics=step_metrics  # 包含quality_score
            )

            # 添加到完整历史记录
            history.append(step_record)

            # 添加到截断管理器
            truncation_manager.add_step(step_record, initial_qa_text)

            # 状态信息
            logger.info(f"📌 截断状态: {'已触发' if truncation_manager.is_truncated else '未触发'}")
            if truncation_manager.is_truncated:
                logger.info(f"   保持最近 {len(truncation_manager.recent_steps)} 步 "
                            f"(最大: {truncation_manager.max_recent_steps})")

            # 可选：基于metrics的早停
            if self.config.get("enable_early_stopping", False):
                if self._should_early_stop(history):
                    logger.info("⚠️ 触发早停条件")
                    break

            # 更新结果并保存
            result = self._build_result_dict(
                data_item, history,
                truncation_triggered=truncation_manager.is_truncated,
                truncation_at_step=truncation_manager.max_recent_steps if truncation_manager.is_truncated else None
            )

            # 增量保存
            self._save_result(result_file, result)
            logger.info(f"💾 已保存步骤 {step} 的结果")

        logger.info(f"\n✅ 问题 {question_id} 处理完成，共 {len(history)} 步")
        logger.info(f"💾 隐藏层数据已保存至: {hidden_states_file}")
        return result

    def _build_result_dict(self, data_item: Dict[str, Any], history: List[StepRecord],
                           truncation_triggered: bool, truncation_at_step: Optional[int]) -> Dict[str, Any]:
        """构建结果字典"""
        return {
            "id": data_item.get("id", "unknown"),
            "question": data_item.get("question", ""),
            "nli_label": None,  # 一开始就有，方便观察
            "context_preview": data_item.get("context", "")[:200] + "..." if len(
                data_item.get("context", "")) > 200 else data_item.get("context", ""),
            "dataset_type": self.config.get("dataset_type", "unknown"),
            "model": self.config["model_path"],
            "temperature": self.config.get("temperature", 0.7),
            "strategy": self.strategy_mode,  # 简化的字段名
            "max_steps": self.config["max_steps"],
            "actual_steps": len(history) - 1,  # 不包括初始QA
            "max_generation_tokens": self.max_generation_tokens,
            "max_input_tokens": self.max_input_length,
            "truncation_triggered": truncation_triggered,
            "truncation_at_step": truncation_at_step,
            "generation_timeout": self.config.get("generation_timeout", 300),
            "last_update": datetime.datetime.now().isoformat(),
            "reasoning_chain": [
                {
                    "step": h.step,
                    "prompt_type": h.prompt_type,
                    "full_input": h.full_input,
                    "followup_prompt": h.followup_prompt,
                    "response": h.response,
                    "timestamp": h.timestamp,
                    "token_count": h.token_count,
                    "response_length": len(h.response.split()),
                    "input_length": len(h.full_input.split()),
                    "metrics": h.metrics  # 包含所有metrics和quality_score
                } for h in history
            ]
        }

    # def _save_result(self, filepath: str, result: Dict[str, Any]) -> None:
    #     """安全地保存结果（原子操作）"""
    #     # 先写入临时文件
    #     temp_file = f"{filepath}.tmp"
    #     try:
    #         with open(temp_file, "w", encoding="utf-8") as f:
    #             json.dump(result, f, ensure_ascii=False, indent=2)
    #
    #         # 原子性地替换原文件
    #         if os.path.exists(filepath):
    #             os.replace(temp_file, filepath)
    #         else:
    #             os.rename(temp_file, filepath)
    #
    #     except Exception as e:
    #         logger.error(f"保存结果失败: {e}")
    #         # 清理临时文件
    #         if os.path.exists(temp_file):
    #             os.remove(temp_file)
    #         raise
    def _save_result(self, filepath: str, result: Dict[str, Any]) -> None:
        """安全地保存结果（原子操作，保留已有nli_label字段）"""
        temp_file = f"{filepath}.tmp"
        try:
            # 先读取现有label（如果有且非None则保留）
            if os.path.exists(filepath):
                with open(filepath, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                if 'nli_label' in existing and existing['nli_label'] is not None:
                    result['nli_label'] = existing['nli_label']

            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            if os.path.exists(filepath):
                os.replace(temp_file, filepath)
            else:
                os.rename(temp_file, filepath)
        except Exception as e:
            logger.error(f"保存结果失败: {e}")
            if os.path.exists(temp_file):
                os.remove(temp_file)
            raise

    def process_dataset(self) -> None:
        """处理数据集"""
        logger.info(f"📚 处理 {self.dataset_type} 数据集")

        # # 获取数据集统计信息
        # stats = self.dataset_processor.get_statistics()
        # logger.info(f"📊 数据集统计:")
        # logger.info(f"   - 总条数: {stats['total_items']}")
        # logger.info(f"   - 字段: {stats['columns']}")

        # 获取数据集统计信息
        logger.info(f"📊 数据集统计:")
        logger.info(f"   - 总条数: {len(self.dataset_processor)}")
        logger.info(f"   - 字段: {self.dataset_processor.column_names}")

        # 创建进度文件
        progress_file = os.path.join(self.config["output_dir"], "progress.json")

        # 处理每个问题
        total_items = len(self.dataset_processor)
        for idx, data_item in enumerate(tqdm(self.dataset_processor, desc="处理问题")):
            try:
                logger.info(f"\n{'=' * 60}")
                logger.info(f"处理进度: {idx + 1}/{total_items}")

                # 更新进度
                progress = {
                    "current": idx + 1,
                    "total": total_items,
                    "percentage": (idx + 1) / total_items * 100,
                    "current_item": data_item.get("id", f"item_{idx}"),
                    "timestamp": datetime.datetime.now().isoformat()
                }
                with open(progress_file, "w") as f:
                    json.dump(progress, f, indent=2)

                result = self.process_question(data_item)

            except Exception as e:
                logger.error(f"❌ 处理问题 {data_item.get('id', 'unknown')} 失败: {e}")
                import traceback
                traceback.print_exc()

                # 保存错误信息
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

    def run(self) -> None:
        """运行主程序"""
        logger.info(f"🚀 开始Chain-of-Thought实验")
        logger.info(f"   模型: {self.config['model_path']}")
        logger.info(f"   数据集: {self.config.get('dataset_type', 'unknown')}")
        logger.info(f"   策略: {self.strategy_mode}")
        logger.info(f"   生成超时: {self.config.get('generation_timeout', 300)} 秒")

        start_time = datetime.datetime.now()

        try:
            self.process_dataset()
        except KeyboardInterrupt:
            logger.info("\n⚠️ 用户中断执行")
        except Exception as e:
            logger.error(f"❌ 处理失败: {e}")
            raise
        finally:
            end_time = datetime.datetime.now()
            duration = end_time - start_time
            logger.info(f"\n✨ 实验完成！总用时: {duration}")

            # 生成汇总报告
            self._generate_summary_report()

    def _generate_summary_report(self) -> None:
        """生成汇总报告"""
        output_dir = Path(self.config["output_dir"])
        results = []

        # 收集所有结果
        for json_file in output_dir.glob("*.json"):
            if json_file.name in ["summary_report.json", "progress.json"]:
                continue
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    results.append(json.load(f))
            except Exception as e:
                logger.error(f"读取结果文件 {json_file} 失败: {e}")

        if not results:
            logger.warning("没有找到任何结果文件")
            return

        # 计算统计信息
        summary = {
            "experiment_config": {
                "model": self.config["model_path"],
                "dataset_type": self.config.get("dataset_type", "unknown"),
                "strategy": self.strategy_mode,
                "max_steps": self.config["max_steps"],
                "temperature": self.config.get("temperature", 0.7),
                "enable_ppl_detection": self.config.get("enable_ppl_detection", True),
                "generation_timeout": self.config.get("generation_timeout", 300),
            },
            "statistics": {
                "total_questions": len(results),
                "avg_actual_steps": sum(r["actual_steps"] for r in results) / len(results),
                "truncation_rate": sum(1 for r in results if r["truncation_triggered"]) / len(results),
                "avg_final_response_length": sum(
                    r["reasoning_chain"][-1]["response_length"] for r in results
                ) / len(results),
            },
            "timestamp": datetime.datetime.now().isoformat()
        }

        # 如果启用了PPL检测，添加PPL统计
        if self.config.get("enable_ppl_detection", True):
            ppl_stats = self._calculate_ppl_statistics(results)
            summary["ppl_statistics"] = ppl_stats

        # 添加超时统计
        timeout_stats = self._calculate_timeout_statistics(results)
        summary["timeout_statistics"] = timeout_stats

        # 保存汇总报告
        summary_file = output_dir / "summary_report.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        logger.info(f"\n📊 实验汇总:")
        logger.info(f"   - 处理问题数: {summary['statistics']['total_questions']}")
        logger.info(f"   - 平均步数: {summary['statistics']['avg_actual_steps']:.1f}")
        logger.info(f"   - 截断率: {summary['statistics']['truncation_rate']:.1%}")
        logger.info(f"   - 平均最终响应长度: {summary['statistics']['avg_final_response_length']:.1f} words")

        if "ppl_statistics" in summary:
            logger.info(f"\n📊 PPL统计:")
            logger.info(f"   - 平均初始PPL: {summary['ppl_statistics']['avg_initial_ppl']:.2f}")
            logger.info(f"   - 平均最终PPL: {summary['ppl_statistics']['avg_final_ppl']:.2f}")
            logger.info(f"   - 平均PPL增长率: {summary['ppl_statistics']['avg_ppl_growth']:.2f}x")

        if "timeout_statistics" in summary:
            logger.info(f"\n⏱️ 超时统计:")
            logger.info(f"   - 超时步骤数: {summary['timeout_statistics']['timeout_count']}")
            logger.info(f"   - 超时率: {summary['timeout_statistics']['timeout_rate']:.1%}")
            logger.info(f"   - 平均生成时间: {summary['timeout_statistics']['avg_generation_time']:.1f} 秒")

        logger.info(f"   - 报告已保存至: {summary_file}")

    def _calculate_ppl_statistics(self, results: List[Dict]) -> Dict[str, float]:
        """计算PPL相关的统计信息"""
        initial_ppls = []
        final_ppls = []
        ppl_growths = []

        for result in results:
            chain = result["reasoning_chain"]

            # 初始PPL
            if chain and "metrics" in chain[0] and "conditional_ppl" in chain[0]["metrics"]:
                initial_ppl = chain[0]["metrics"]["conditional_ppl"]
                initial_ppls.append(initial_ppl)

                # 最终PPL
                if "metrics" in chain[-1] and "conditional_ppl" in chain[-1]["metrics"]:
                    final_ppl = chain[-1]["metrics"]["conditional_ppl"]
                    final_ppls.append(final_ppl)

                    # PPL增长率
                    if initial_ppl > 0:
                        ppl_growths.append(final_ppl / initial_ppl)

        return {
            "avg_initial_ppl": sum(initial_ppls) / len(initial_ppls) if initial_ppls else 0,
            "avg_final_ppl": sum(final_ppls) / len(final_ppls) if final_ppls else 0,
            "avg_ppl_growth": sum(ppl_growths) / len(ppl_growths) if ppl_growths else 0,
            "max_final_ppl": max(final_ppls) if final_ppls else 0,
            "num_samples_with_ppl": len(initial_ppls)
        }

    def _calculate_timeout_statistics(self, results: List[Dict]) -> Dict[str, Any]:
        """计算超时相关的统计信息"""
        timeout_count = 0
        total_steps = 0
        generation_times = []

        for result in results:
            chain = result["reasoning_chain"]
            for step in chain:
                total_steps += 1
                if "metrics" in step:
                    if step["metrics"].get("timeout", False):
                        timeout_count += 1
                    if "generation_time" in step["metrics"] and step["metrics"]["generation_time"] is not None:
                        generation_times.append(step["metrics"]["generation_time"])

        return {
            "timeout_count": timeout_count,
            "total_steps": total_steps,
            "timeout_rate": (timeout_count / total_steps * 100) if total_steps > 0 else 0,
            "avg_generation_time": sum(generation_times) / len(generation_times) if generation_times else 0,
            "max_generation_time": max(generation_times) if generation_times else 0,
            "min_generation_time": min(generation_times) if generation_times else 0
        }


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='Chain-of-Thought Multi-Step Reasoning Experiment')
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
