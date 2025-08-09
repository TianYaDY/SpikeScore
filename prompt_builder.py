"""
Prompt构造模块
自动识别数据字段并构建prompt
使用tokenizer的apply_chat_template处理模型特定格式
"""
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PromptBuilder:
    """自适应的Prompt构建器"""
    tokenizer: Any  # transformers tokenizer instance
    system_prompt: Optional[str] = None

    # 字段名到前缀的映射
    CONTEXT_FIELD_MAP = {
        'context': 'Passage',
        'passage': 'Passage',
        'story': 'Story',
        'background': 'Background',
        'document': 'Document',
        'text': 'Text',
        'paragraph': 'Paragraph',
        'article': 'Article',
        'content': 'Content',
        'source': 'Source',
        'reference': 'Reference',
    }

    QUESTION_FIELD_MAP = {
        'question': 'Question',
        'query': 'Query',
        'prompt': 'Prompt',
        'instruction': 'Instruction',
        'problem': 'Problem',
    }

    def __post_init__(self):
        # 默认系统提示
        if self.system_prompt is None:
            self.system_prompt = "You are a helpful AI assistant. Answer questions accurately based on any provided context."

        logger.info(f"📝 初始化自适应PromptBuilder")

        # 检查tokenizer是否支持chat template
        if not hasattr(self.tokenizer, 'apply_chat_template'):
            logger.warning("⚠️ Tokenizer不支持apply_chat_template，将使用fallback格式")

    def build_prompt(self, data: Dict[str, Any]) -> str:
        """
        自动识别数据字段并构建prompt

        Args:
            data: 包含问题和可能的上下文的字典
                 例如: {'question': '...', 'context': '...'}
                 或者: {'query': '...', 'story': '...'}

        Returns:
            格式化后的prompt
        """
        # 自动构建用户消息
        user_message = self._auto_format_message(data)

        # 构建消息列表
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message}
        ]

        # 使用tokenizer的apply_chat_template
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception as e:
            logger.warning(f"apply_chat_template失败: {e}，使用fallback")
            prompt = self._fallback_format(messages)

        logger.debug(f"自动构建的Prompt预览: {prompt[:200]}...")
        return prompt

    def _auto_format_message(self, data: Dict[str, Any]) -> str:
        """自动识别字段并格式化消息"""
        parts = []

        # 1. 查找并格式化上下文字段
        for field, prefix in self.CONTEXT_FIELD_MAP.items():
            if field in data and data[field]:
                parts.append(f"{prefix}: {data[field]}")
                break  # 只使用第一个找到的上下文字段

        # 2. 查找并格式化问题字段
        for field, prefix in self.QUESTION_FIELD_MAP.items():
            if field in data and data[field]:
                parts.append(f"{prefix}: {data[field]}")
                break  # 只使用第一个找到的问题字段

        # 3. 处理其他未识别的字段（可选）
        recognized_fields = set(self.CONTEXT_FIELD_MAP.keys()) | set(self.QUESTION_FIELD_MAP.keys())
        for field, value in data.items():
            if field not in recognized_fields and value and isinstance(value, str):
                # 将字段名首字母大写作为前缀
                prefix = field.capitalize()
                parts.append(f"{prefix}: {value}")

        # 如果没有找到任何字段，返回警告
        if not parts:
            logger.warning("⚠️ 没有找到可识别的字段")
            return "No content found"

        return "\n\n".join(parts)

    def build_prompt_simple(self, question: str, context: str = "") -> str:
        """
        简单接口：直接传入问题和上下文
        保持向后兼容
        """
        data = {'question': question}
        if context:
            data['context'] = context
        return self.build_prompt(data)

    def _fallback_format(self, messages: List[Dict[str, str]]) -> str:
        """备用格式化（当tokenizer不支持chat template时）"""
        parts = []
        for msg in messages:
            if msg['role'] == 'system':
                parts.append(f"System: {msg['content']}")
            elif msg['role'] == 'user':
                parts.append(f"User: {msg['content']}")
        parts.append("Assistant:")
        return "\n\n".join(parts) + " "

    def detect_fields(self, data: Dict[str, Any]) -> Dict[str, str]:
        """检测数据中的字段类型（用于调试）"""
        detected = {}

        for field in data:
            if field in self.CONTEXT_FIELD_MAP:
                detected[field] = 'context'
            elif field in self.QUESTION_FIELD_MAP:
                detected[field] = 'question'
            else:
                detected[field] = 'unknown'

        return detected


# ==================== 便捷函数 ====================

def create_prompt_builder(tokenizer, system_prompt: Optional[str] = None) -> PromptBuilder:
    """创建自适应Prompt构建器"""
    return PromptBuilder(tokenizer, system_prompt)


# ==================== 测试示例 ====================

if __name__ == "__main__":
    from transformers import AutoTokenizer

    # 测试各种数据格式
    test_data = [
        # SQuAD格式
        {
            'context': 'The Eiffel Tower is located in Paris, France.',
            'question': 'Where is the Eiffel Tower located?'
        },
        # CoQA格式
        {
            'story': 'Once upon a time, there was a princess who lived in a castle.',
            'question': 'Where did the princess live?'
        },
        # DefAn格式（只有问题）
        {
            'question': 'What is machine learning?'
        },
        # 自定义格式
        {
            'background': 'Python is a programming language.',
            'query': 'What is Python?'
        },
        # 混合格式
        {
            'document': 'Climate change affects global weather patterns.',
            'prompt': 'Explain the effects mentioned.',
            'note': 'This is an additional field'  # 额外字段也会被处理
        }
    ]

    # 使用示例模型
    tokenizer = AutoTokenizer.from_pretrained("../models/Llama-3.2-3B-Instruct")  # 仅作示例
    builder = create_prompt_builder(tokenizer)

    for i, data in enumerate(test_data):
        print(f"\n{'='*60}")
        print(f"Test case {i+1}:")
        print(f"Input data: {data}")
        print(f"Detected fields: {builder.detect_fields(data)}")

        try:
            prompt = builder.build_prompt(data)
            print(f"\nGenerated prompt:\n{prompt}")
        except Exception as e:
            print(f"Error: {e}")