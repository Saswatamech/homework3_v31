from typing import overload

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "HuggingFaceTB/SmolLM2-360M-Instruct"
#checkpoint = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


class BaseLLM:
    def __init__(self, checkpoint=checkpoint):
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
        self.device = device
        self.tokenizer.padding_side = "left"

    def format_prompt(self, question: str) -> str:
        """
        Take a question and convert it into an input to SmolLM2. The LLM will likely answer much
        better if you provide a chat template. self.tokenizer.apply_chat_template can help here
        You don't need to change this function for now.
        """
        return question

    def parse_answer(self, answer: str) -> float:
        """
        Parse the <answer></answer> tag and return a float.
        This function is somewhat robust to output errors (e.g. missing </answer> tags).
        """
        try:
            return float(answer.split("<answer>")[1].split("</answer>")[0])
        except (IndexError, ValueError):
            return float("nan")

    def generate(self, prompt: str) -> str:
        """
        (Optional) Implement this method first and then implement batched_generate below.
        It is much easier to implement generation without batching.

        The overall flow is the same:
        - tokenize the prompt with self.tokenizer
        - call self.model.generate
        - decode the outputs with self.tokenizer.decode

        """
        return self.batched_generate([prompt])[0]

    @overload
    def batched_generate(
        self, prompts: list[str], num_return_sequences: None = None, temperature: float = 0
    ) -> list[str]:
        """
        Batched version of `generate` method.
        This version returns a single generation for each prompt.
        """

    @overload
    def batched_generate(
        self, prompts: list[str], num_return_sequences: int, temperature: float = 0
    ) -> list[list[str]]:
        """
        Batched version of `generate` method.
        This version returns a list of generation for each prompt.
        """

    def batched_generate(
        self, prompts: list[str], num_return_sequences: int | None = None, temperature: float = 0
    ) -> list[str] | list[list[str]]:
        """
        Batched version of `generate` method.

        You will likely get an up to 10x speedup using batched decoding.

        To implement batch decoding you will need to:
        - tokenize the prompts self.tokenizer with padding=True and return_tensors="pt"
        - call self.model.generate
        - decode the outputs with self.tokenizer.batch_decode

        Tip: You need to set self.tokenizer.padding_side = "left" to get the correct padding behavior for generation.
             Left padding makes sure all sequences are aligned to the right (i.e. where tokens are generated).
        Tip: self.model.generate takes a lot of parameters. Here are some relevant ones:
            - max_new_tokens: The maximum number of tokens to generate. Set this to a reasonable value
                              (50 should suffice).
            - do_sample and temperature: For any temperature > 0, set do_sample=True.
                                         do_sample=False will use greedy decoding.
            - num_return_sequences: The number of sequences to return. Note that this will generate a flat
                                    list of len(prompts) * num_return_sequences entries.
            - eos_token_id: The end of sequence token id. This is used to stop generation. Set this
                            to self.tokenizer.eos_token_id.
        Pro Tip: Only batch_decode generated tokens by masking out the inputs with
                 outputs[:, len(inputs["input_ids"][0]) :]
        """
        from tqdm import tqdm  # Importing tqdm for progress bar
        # Preventing OOM
        # Depending on your GPU batched generation will use a lot of memory.
        # If you run out of memory, try to reduce the micro_batch_size.
        micro_batch_size = 2 # Reduced for potentially lower memory consumption with SmolLM2-360M
        if len(prompts) > micro_batch_size:
            all_generations = []
            for idx in tqdm(
                range(0, len(prompts), micro_batch_size), desc=f"LLM Running on Micro Batches {micro_batch_size}"
            ):
                micro_batch_prompts = prompts[idx : idx + micro_batch_size]
                all_generations.extend(self.batched_generate(micro_batch_prompts, num_return_sequences, temperature))
            return all_generations

        # Tokenize the prompts with padding=True and return_tensors="pt"
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Determine generation parameters
        generation_kwargs = {
            "max_new_tokens": 50,  # Set to a reasonable value
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id # Fallback
        }

        if temperature > 0:
            generation_kwargs["do_sample"] = True
            generation_kwargs["temperature"] = temperature
            #generation_kwargs["top_p"] = 0.9 # Common sampling parameter
        else:
            generation_kwargs["do_sample"] = False
            # For greedy/beam search, num_beams > 1 would be used with do_sample=False
            # If num_return_sequences is requested for greedy, it will return identical sequences
            # unless a different strategy like diverse beam search is employed,
            # but for this specific problem, it's implied num_return_sequences relates to sampling.

        if num_return_sequences is not None:
            generation_kwargs["num_return_sequences"] = num_return_sequences
            # If num_return_sequences > 1, often num_beams is also set.
            # However, if do_sample=True, num_return_sequences will return different samples.
            # If do_sample=False and num_beams=1, num_return_sequences > 1 will return identical sequences.
            # For the purpose of this problem, let's assume num_return_sequences is for sampling multiple outputs.


        # Call self.model.generate
        outputs = self.model.generate(
            **inputs,
            **generation_kwargs
        )

        # Calculate the length of the original input for each prompt in the batch
        # This is the number of tokens before generation
        # When padding_side="left", the original tokens are aligned to the right.
        # The generated tokens are appended after these.
        # Therefore, for each output sequence, the number of input tokens is the length of its corresponding original input.
        # If input_ids are all of the same length due to padding and truncation, `inputs["input_ids"].shape[1]` can be used.
        # However, for robustness, especially if truncation leads to varying original lengths,
        # it's safer to determine the start of generation for each output.
        # With padding_side="left", the original prompt tokens are at the end of the input_ids sequence,
        # and the generated tokens follow.
        # The total length of `outputs` will be `input_length + generated_length`.
        # We need to find the `input_length` for each sequence.

        # When `num_return_sequences > 1`, `model.generate` flattens the output.
        # So `outputs` will have `len(prompts) * num_return_sequences` rows.
        # Each row `outputs[k]` corresponds to `prompts[k // num_return_sequences]`.
        # The input length for that row is `len(inputs["input_ids"][k // num_return_sequences])`.

        decoded_generations = []
        if num_return_sequences is not None:
            for i in range(len(prompts)):
                prompt_generations = []
                for j in range(num_return_sequences):
                    output_idx = i * num_return_sequences + j
                    # Get the input length for the corresponding original prompt
                    input_len_for_this_output = len(inputs["input_ids"][i])
                    # Slice to get only the newly generated tokens
                    generated_token_ids = outputs[output_idx, input_len_for_this_output:]
                    decoded_text = self.tokenizer.decode(generated_token_ids, skip_special_tokens=True)
                    prompt_generations.append(decoded_text.strip())
                decoded_generations.append(prompt_generations)
        else:
            for i in range(len(prompts)):
                # For single return sequence, outputs[i] corresponds to prompts[i]
                input_len_for_this_output = len(inputs["input_ids"][i])
                generated_token_ids = outputs[i, input_len_for_this_output:]
                decoded_text = self.tokenizer.decode(generated_token_ids, skip_special_tokens=True)
                decoded_generations.append(decoded_text.strip())

        # Handle the return type based on num_return_sequences
        if num_return_sequences is None:
            return decoded_generations # list[str]
        else:
            return decoded_generations # list[list[str]]

    def answer(self, *questions) -> list[float]:
        """
        Answer questions given as individual string arguments.
        """
        # Convert each question
        prompts = [self.format_prompt(q) for q in questions]
        generations = self.batched_generate(prompts)
        return [self.parse_answer(g) for g in generations]


def test_model():
    # The following code simply tests of the BaseLLM is able to complete text.
    # It should produce garbage answers, but it should not crash.
    # In my case it talks about cats eating cats, and dogs being happy.
    testset = ["The cat went up", "The dog went down"]
    model = BaseLLM()
    for t in testset:
        print("testing generate function")
        print("input", t)
        answer = model.generate(t)
        print("output", answer)
    answers = model.batched_generate(testset)
    print(answers)


if __name__ == "__main__":
    from fire import Fire

    Fire({"test": test_model})
