from openai import OpenAI
import threading
from typing import Optional

client = OpenAI(
    api_key="sk-your_key",
    base_url="https://xxxxx/v1"
)

def judge_answer(model_answer: str, reference_answer: str, question: str, timeout: float = 10.0) -> Optional[int]:
    """
    Judge if the model_answer matches the reference_answer for a specific question using OpenAI API (few-shot style).
    Returns:
        1 if semantically correct,
        0 if incorrect,
        None if timeout or error.
    """
    result = {'label': None}

    def _call():
        try:
            system_msg = {
                "role": "system",
                "content": (
                    "You are an impartial evaluator. "
                    "Given a question, a reference answer, and a model answer, judge if the model answer is semantically equivalent to the reference answer in the context of the question. "
                    "Reply ONLY with '1' (correct) or '0' (incorrect). No explanation."
                )
            }
            # Few-shot examples
            examples = [
                {
                    "role": "user",
                    "content": (
                        "Question: What is the capital of France?\n"
                        "Reference answer: The capital of France is Paris.\n"
                        "Model answer: Paris is the capital of France.\n"
                        "Your reply:"
                    )
                },
                {"role": "assistant", "content": "1"},
                {
                    "role": "user",
                    "content": (
                        "Question: What is the chemical formula for water?\n"
                        "Reference answer: The chemical formula for water is H2O.\n"
                        "Model answer: Water is composed of two hydrogen atoms and one oxygen atom.\n"
                        "Your reply:"
                    )
                },
                {"role": "assistant", "content": "1"},
                {
                    "role": "user",
                    "content": (
                        "Question: Which is the largest planet in the solar system?\n"
                        "Reference answer: The largest planet in the solar system is Jupiter.\n"
                        "Model answer: Saturn is the biggest planet in our solar system.\n"
                        "Your reply:"
                    )
                },
                {"role": "assistant", "content": "0"},
                {
                    "role": "user",
                    "content": (
                        "Question: How do plants make food?\n"
                        "Reference answer: The process by which plants make food is called photosynthesis.\n"
                        "Model answer: Plants produce food through respiration.\n"
                        "Your reply:"
                    )
                },
                {"role": "assistant", "content": "0"},
            ]
            # Your actual query
            query = {
                "role": "user",
                "content": (
                    f"Question: {question}\n"
                    f"Reference answer: {reference_answer}\n"
                    f"Model answer: {model_answer}\n"
                    "Your reply:"
                )
            }

            messages = [system_msg] + examples + [query]

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.0,
                max_tokens=1
            )
            reply = response.choices[0].message.content.strip()
            if reply == "1":
                result['label'] = 1
            elif reply == "0":
                result['label'] = 0
            else:
                result['label'] = None
        except Exception as e:
            result['label'] = None

    thread = threading.Thread(target=_call, daemon=True)
    thread.start()
    thread.join(timeout=timeout)
    if thread.is_alive():
        return None
    return result['label']


if __name__ == "__main__":

    # Example 1: Perfect match
    model_answer = "Paris is the capital of France."
    reference_answer = "The capital of France is Paris."
    question = "What is the capital of France?"
    print("Test 1:", judge_answer(model_answer, reference_answer, question))  # Should print 1

    # Example 2: Semantic equivalent
    model_answer = "Water consists of two hydrogen atoms and one oxygen atom."
    reference_answer = "The chemical formula for water is H2O."
    question = "What is the chemical formula for water?"
    print("Test 2:", judge_answer(model_answer, reference_answer, question))  # Should print 1

    # Example 3: Incorrect answer
    model_answer = "Saturn is the biggest planet in our solar system."
    reference_answer = "The largest planet in the solar system is Jupiter."
    question = "Which is the largest planet in the solar system?"
    print("Test 3:", judge_answer(model_answer, reference_answer, question))  # Should print 0

    # Example 4: Borderline case (different fact)
    model_answer = "Plants get energy from eating insects."
    reference_answer = "The process by which plants make food is called photosynthesis."
    question = "How do plants make food?"
    print("Test 4:", judge_answer(model_answer, reference_answer, question))  # Should print 0

    # Example 5: Completely irrelevant
    model_answer = "I like turtles."
    reference_answer = "The Earth revolves around the Sun."
    question = "What is the center of our solar system?"
    print("Test 5:", judge_answer(model_answer, reference_answer, question))  # Should print 0

    # Example 6: Empty or error (should handle gracefully)
    model_answer = ""
    reference_answer = ""
    question = ""
    print("Test 6:", judge_answer(model_answer, reference_answer, question))  # Should print 0 or None

    # Example 7: API timeout simulation (very short timeout)
    print("Test 7:", judge_answer("Paris", "Paris", "What is the capital of France?", timeout=0.01))  # Should print None

