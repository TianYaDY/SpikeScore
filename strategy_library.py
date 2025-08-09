"""
Chain-of-Thought 策略库
统一管理所有策略定义和选择逻辑
"""
import random
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

# ==================== 策略类型定义 ====================

class StrategyType(Enum):
    """策略类型"""
    ENCOURAGEMENT = "encouragement"
    ANALYTICAL = "analytical"
    STEP_BY_STEP = "step_by_step"
    EXPLORATORY = "exploratory"
    CRITICAL = "critical"
    CREATIVE = "creative"
    TECHNICAL = "technical"
    REFLECTIVE = "reflective"


# ==================== 策略数据结构 ====================

@dataclass
class Strategy:
    """策略数据类"""
    text: str                   # prompt文本
    type: StrategyType         # 策略类型
    intensity: int = 1         # 强度 1-5


# ==================== 策略定义 ====================

class StrategyDefinitions:
    """所有策略的定义"""

    STRATEGIES = {
        # 鼓励型
        "encouragement": [
            Strategy("Great! Continue.", StrategyType.ENCOURAGEMENT, 1),
            Strategy("Excellent! Please continue with more details.", StrategyType.ENCOURAGEMENT, 2),
            Strategy("Perfect! Keep going with your analysis.", StrategyType.ENCOURAGEMENT, 2),
            Strategy("That's a good start! Now, let's dive deeper into the specifics. What else can you tell me about this?", StrategyType.ENCOURAGEMENT, 3),
            Strategy("Wonderful analysis so far! I'm impressed by your reasoning. Can you expand on these points with more concrete examples?", StrategyType.ENCOURAGEMENT, 4),
            Strategy("Outstanding work! Your explanation is becoming clearer. Please continue developing these ideas with additional evidence or reasoning.", StrategyType.ENCOURAGEMENT, 4),
            Strategy("Your analysis demonstrates exceptional understanding! The depth and clarity of your reasoning is truly impressive. Now, I'd like you to push even further - explore the nuances, edge cases, and broader implications.", StrategyType.ENCOURAGEMENT, 5),
        ],

        # 分析型
        "analytical": [
            Strategy("Now let's analyze this systematically. Break it down further.", StrategyType.ANALYTICAL, 2),
            Strategy("Interesting perspective. Let's examine each component more carefully. What are the underlying principles at work here?", StrategyType.ANALYTICAL, 3),
            Strategy("Let's dissect this problem methodically. Identify the key variables, their interactions, and the governing principles.", StrategyType.ANALYTICAL, 4),
            Strategy("Your analysis is progressing well. Now, let's apply more rigorous analytical frameworks. Consider the causal relationships, dependencies, and potential edge cases in your reasoning.", StrategyType.ANALYTICAL, 5),
        ],

        # 步骤型
        "step_by_step": [
            Strategy("Good. What's the next step?", StrategyType.STEP_BY_STEP, 1),
            Strategy("Please continue with step-by-step reasoning.", StrategyType.STEP_BY_STEP, 2),
            Strategy("Excellent progress. Now, let's move to the next logical step in this process. What comes after what you just described?", StrategyType.STEP_BY_STEP, 3),
            Strategy("Very systematic approach! You've covered the initial steps beautifully. Now, let's continue with the subsequent steps, maintaining the same level of detail and clarity.", StrategyType.STEP_BY_STEP, 4),
        ],

        # 探索型
        "exploratory": [
            Strategy("Interesting! What other angles should we consider?", StrategyType.EXPLORATORY, 2),
            Strategy("That opens up new possibilities! Let's explore alternative perspectives or unconventional approaches to this problem.", StrategyType.EXPLORATORY, 3),
            Strategy("Let's think beyond the obvious. What hidden connections or non-intuitive insights might we discover?", StrategyType.EXPLORATORY, 4),
            Strategy("Fascinating direction! Now let's venture into unexplored territory. What if we challenged our initial assumptions? What alternative frameworks or paradigms could we apply here?", StrategyType.EXPLORATORY, 5),
        ],

        # 技术型
        "technical": [
            Strategy("Now provide the technical specifications.", StrategyType.TECHNICAL, 2),
            Strategy("Excellent! Now let's get into the technical details. Include specific parameters, formulas, or implementation details.", StrategyType.TECHNICAL, 3),
            Strategy("Provide precise technical measurements, calculations, and specifications. Be as quantitative as possible.", StrategyType.TECHNICAL, 4),
            Strategy("Impressive technical foundation! Now, delve into the advanced technical aspects. Include mathematical proofs, algorithmic complexity analysis, or system architecture details as relevant.", StrategyType.TECHNICAL, 5),
        ],

        # 批判型
        "critical": [
            Strategy("What are the potential flaws in this reasoning?", StrategyType.CRITICAL, 2),
            Strategy("Play devil's advocate. What would a skeptic say about this analysis?", StrategyType.CRITICAL, 3),
            Strategy("Let's apply critical thinking here. What assumptions are we making? What could go wrong? What are the counterarguments?", StrategyType.CRITICAL, 4),
            Strategy("Time for rigorous critical analysis. Identify potential logical fallacies, hidden biases, unstated assumptions, and edge cases. Challenge every aspect of the reasoning.", StrategyType.CRITICAL, 5),
        ],

        # 创造型
        "creative": [
            Strategy("Be more creative with your approach!", StrategyType.CREATIVE, 2),
            Strategy("Wonderful! Now let's think outside the box. What innovative or unconventional solutions can you propose?", StrategyType.CREATIVE, 3),
            Strategy("What if we approached this from a completely different angle? Think metaphorically, analogically, or even poetically.", StrategyType.CREATIVE, 4),
            Strategy("Push the boundaries! Imagine revolutionary approaches, combine disparate concepts, or invent entirely new frameworks. Let your imagination run wild while maintaining logical coherence.", StrategyType.CREATIVE, 5),
        ],

        # 反思型
        "reflective": [
            Strategy("Reflect on what you've said so far.", StrategyType.REFLECTIVE, 1),
            Strategy("Take a moment to reflect. How does everything connect? What patterns do you notice in your reasoning?", StrategyType.REFLECTIVE, 2),
            Strategy("Let's pause and examine the journey of your reasoning. What key insights have emerged? What remains unclear?", StrategyType.REFLECTIVE, 3),
            Strategy("Step back and reflect deeply. What meta-insights emerge from your analysis? How has your understanding evolved? What would you reconsider or emphasize differently?", StrategyType.REFLECTIVE, 4),
        ],
    }


# ==================== 策略选择器 ====================

class StrategySelector:
    """策略选择逻辑"""

    def __init__(self, definitions: StrategyDefinitions):
        self.definitions = definitions

    def select_progressive(self, step: int) -> Strategy:
        """渐进式策略选择"""
        if step <= 3:
            # 前3步：温和鼓励
            strategies = self.definitions.STRATEGIES["encouragement"]
            return strategies[min(step-1, len(strategies)-1)]
        elif step <= 6:
            # 4-6步：开始分析
            strategies = self.definitions.STRATEGIES["analytical"]
            return strategies[min((step-4), len(strategies)-1)]
        elif step <= 9:
            # 7-9步：深入探索
            strategies = self.definitions.STRATEGIES["exploratory"]
            return strategies[min((step-7), len(strategies)-1)]
        elif step <= 12:
            # 10-12步：技术细节
            strategies = self.definitions.STRATEGIES["technical"]
            return strategies[min((step-10), len(strategies)-1)]
        elif step <= 15:
            # 13-15步：批判思考
            strategies = self.definitions.STRATEGIES["critical"]
            return strategies[min((step-13), len(strategies)-1)]
        else:
            # 16步以上：深度反思
            strategies = self.definitions.STRATEGIES["reflective"]
            return strategies[-1]  # 使用最深度的反思

    def select_mixed(self, step: int) -> Strategy:
        """混合策略选择"""
        if step <= 3:
            pool = ["encouragement", "analytical"]
        elif step <= 10:
            pool = ["analytical", "exploratory", "creative"]
        else:
            pool = ["technical", "critical", "reflective"]

        strategy_type = random.choice(pool)
        strategies = self.definitions.STRATEGIES[strategy_type]
        return random.choice(strategies)

    def select_random(self, step: int) -> Strategy:
        """随机策略选择"""
        if step <= 2:
            # 前2步还是用基础的
            return self.definitions.STRATEGIES["encouragement"][0]

        # 从所有策略中随机选择
        all_strategies = []
        for strategies in self.definitions.STRATEGIES.values():
            all_strategies.extend(strategies)
        return random.choice(all_strategies)

    def select_adaptive(self, step: int, context: Dict[str, Any]) -> Strategy:
        """自适应策略选择"""
        last_response_length = context.get('last_response_length', 0)

        if last_response_length < 50:
            # 响应太短，使用更强烈的分析型prompt
            strategies = self.definitions.STRATEGIES["analytical"]
            # 选择高强度的
            high_intensity = [s for s in strategies if s.intensity >= 3]
            return random.choice(high_intensity) if high_intensity else strategies[-1]

        elif last_response_length > 500:
            # 响应太长，使用反思型
            strategies = self.definitions.STRATEGIES["reflective"]
            return strategies[min(2, len(strategies)-1)]

        else:
            # 正常长度，使用渐进式
            return self.select_progressive(step)

    def select_deep_analysis(self, step: int) -> Strategy:
        """深度分析策略"""
        cycle = ["analytical", "technical", "critical"]
        strategy_type = cycle[(step-1) % len(cycle)]
        strategies = self.definitions.STRATEGIES[strategy_type]

        # 随着步数增加，选择更高强度的
        intensity_target = min(5, 2 + (step // 3))
        suitable = [s for s in strategies if s.intensity >= intensity_target]
        return suitable[0] if suitable else strategies[-1]

    def select_creative_thinking(self, step: int) -> Strategy:
        """创造性思维策略"""
        if step % 3 == 1:
            strategy_type = "creative"
        elif step % 3 == 2:
            strategy_type = "exploratory"
        else:
            strategy_type = "reflective"

        strategies = self.definitions.STRATEGIES[strategy_type]
        return random.choice(strategies)


# ==================== 主策略库类 ====================

class StrategyLibrary:
    """策略库主类"""

    def __init__(self):
        self.definitions = StrategyDefinitions()
        self.selector = StrategySelector(self.definitions)

    def get_prompt_for_context(self, context: Dict[str, Any]) -> str:
        """
        根据上下文获取合适的prompt

        Args:
            context: 包含以下信息的字典
                - step: 当前步数
                - strategy_mode: 策略模式
                - last_response_length: 上一次响应的长度
                - 其他可选信息

        Returns:
            纯文本的prompt
        """
        step = context.get('step', 1)
        mode = context.get('strategy_mode', 'progressive')

        # 根据模式选择策略
        if mode == 'progressive':
            strategy = self.selector.select_progressive(step)
        elif mode == 'mixed':
            strategy = self.selector.select_mixed(step)
        elif mode == 'random':
            strategy = self.selector.select_random(step)
        elif mode == 'adaptive':
            strategy = self.selector.select_adaptive(step, context)
        elif mode == 'deep_analysis':
            strategy = self.selector.select_deep_analysis(step)
        elif mode == 'creative_thinking':
            strategy = self.selector.select_creative_thinking(step)
        elif mode == 'fixed':
            # 固定策略：始终使用同一个
            strategy = self.definitions.STRATEGIES["analytical"][0]
        else:
            # 默认使用渐进式
            strategy = self.selector.select_progressive(step)

        return strategy.text

    def get_strategy_info(self, mode: str) -> Dict[str, Any]:
        """获取策略模式的信息"""
        info = {
            'progressive': {
                'name': '渐进式增强',
                'description': '从简单鼓励逐步过渡到深度分析和反思'
            },
            'mixed': {
                'name': '混合策略',
                'description': '根据步数混合使用不同类型的策略'
            },
            'random': {
                'name': '随机策略',
                'description': '随机选择各种类型的策略'
            },
            'adaptive': {
                'name': '自适应策略',
                'description': '根据响应长度等因素动态调整策略'
            },
            'deep_analysis': {
                'name': '深度分析',
                'description': '循环使用分析、技术和批判型策略'
            },
            'creative_thinking': {
                'name': '创造性思维',
                'description': '强调创造性、探索性和反思性思考'
            },
            'fixed': {
                'name': '固定策略',
                'description': '始终使用同一个分析型提示'
            }
        }
        return info.get(mode, {'name': '未知', 'description': '未定义的策略模式'})

    def get_available_modes(self) -> List[str]:
        """获取所有可用的策略模式"""
        return ['progressive', 'mixed', 'random', 'adaptive',
                'deep_analysis', 'creative_thinking', 'fixed']


# ==================== 全局实例 ====================

_library_instance = None

def get_strategy_library() -> StrategyLibrary:
    """获取策略库单例"""
    global _library_instance
    if _library_instance is None:
        _library_instance = StrategyLibrary()
    return _library_instance


# ==================== 测试 ====================

if __name__ == "__main__":
    library = get_strategy_library()

    # 测试不同策略模式
    test_modes = ['progressive', 'mixed', 'adaptive']

    for mode in test_modes:
        print(f"\n{'='*50}")
        print(f"测试策略模式: {mode}")
        info = library.get_strategy_info(mode)
        print(f"描述: {info['description']}")
        print(f"{'='*50}")

        # 模拟多步
        for step in [1, 5, 10, 15, 20]:
            context = {
                'step': step,
                'strategy_mode': mode,
                'last_response_length': random.randint(20, 600)
            }
            prompt = library.get_prompt_for_context(context)
            print(f"\n步骤 {step}: {prompt}")