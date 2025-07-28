from .base_llm import BaseLLM


class CoTModel(BaseLLM):
    def format_prompt(self, question: str) -> str:
        """
        Take a question and convert it into a chat template. The LLM will likely answer much
        better if you provide a chat template. self.tokenizer.apply_chat_template can help here
        """
        messages = [
            {"role": "system", "content": "You are an expert in calculating unit conversions with precision.Be concise and provide final answer in the form of <answer></answer>"},
            # {"role": "user", "content": "'Can you provide the conversion value from kg to g for 6 units?"},
            # {"role": "assistant", "content": "Use step by step. Identify given: 6 kg . Target: g. Use conversion factor 1 kg = 1000 g. Calculate 6*1000 = <answer>6000.0</answer>"},
            {"role": "user", "content": " How many ounces are 5 gallons?"},
            {"role": "assistant", "content": "Given: 5 gallon. Target: ounce. 1 gallon is 128 ounce. 5*128 = <answer>640.0</answer>"},
            {"role": "user", "content": "How many seconds are there in 10 hours?"},
            {"role": "assistant", "content": "Given: 10 hours. Target: second. 1 hour is 3600 second. 10*3600 = <answer>36000.0</answer>"},
            # {"role": "user", "content": "How much is 3 ft when converted to yd?"},
            # {"role": "assistant", "content": "3ft is 1 yd.<answer>1.0</answer>"},
            # {"role": "user", "content": "Can you provide the conversion value from kg to g for 6 units?"},
            # {"role": "assistant", "content": "6 kg is 6000 g.<answer>6000.0</answer>"},
            # {"role": "user", "content": "What is 1 GB in M?"},
            # {"role": "assistant", "content": "1 GB is 1024 M.<answer>1024.0</answer>"},
            # {"role": "user", "content": "Could you provide the value of 8 week in day?"},
            # {"role": "assistant", "content": "Use step by step. Identify given: 8 week . Target: day. Use conversion factor 1 week = 7 day. Calculate 8*7 = 56.0 <answer>56.0</answer>"},
#             {"role": "user", "content": "How much is 5 litre when converted to milliliter?"},
#             {"role": "assistant", "content": "1 litre = 1000 milliliter. 5*1000=5000 <answer>5000.0</answer>"},
#             # The actual question to be answered
#             {"role": "user", "content": "How does 4 years measure up in terms of week?"},
#             {"role": "assistant", "content": "Use step by step. Identify given: 4 years . Target: week. Use conversion factor 1 year = 52.14 week. Calculate 4*52.14 = 208.56 <answer>208.56</answer>"},
#             {"role": "user", "content": "'Convert the measurement of 7 meter into cm"},
#             {"role": "assistant", "content": "Use step by step. Identify given: 7 meter. Target: cm. Use conversion factor 7 meter = 100 cm. Calculate 7*100 = 700 <answer>700.0</answer>"},

            {"role": "user", "content": f"{question}."},
           ]

        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def load() -> CoTModel:
    return CoTModel()


def test_model():
    from .data import Dataset, benchmark

    testset = Dataset("valid")
    model = CoTModel()
    benchmark_result = benchmark(model, testset, 100)
    print("benchmark_result",benchmark_result)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"test": test_model, "load": load})
