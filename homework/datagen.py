import json
import re
from pathlib import Path
from .cot import CoTModel



from .base_llm import BaseLLM
from .data import Dataset # Assuming CoTModel is in .data

def generate_dataset(output_json: str="data/rft.json", oversample: int = 20, temperature: float = 0.6):
    #raise NotImplementedError()
# Initialize CoTModel with the specified Instruct model
    #output_json = "data/rft.json"
    # Load the training dataset
    train_dataset = Dataset("train")
    cot_llm = CoTModel()

    rft_data = []
    success_count = 0

    print(f"Generating RFT dataset with {oversample} oversamples per question...")

    for i, (question, ground_truth_answer) in enumerate(train_dataset):
        # Prompt for CoT generation
        cot_prompt = (
            f"Given the question: \"{question}\", "
             "provide a detailed thought process to arrive at the numerical answer. "
            "Then, state the final numerical answer in the format <answer>{answer}</answer>.\n"
            "Thought:"
        )

        # Generate multiple completions
        # Ensure that batched_generate returns a list of strings
        try:
            generated_texts = cot_llm.batched_generate(prompts=cot_llm.format_prompt(question), num_return_sequences=oversample, temperature=temperature)[0]
        except Exception as e:
            print(f"Error generating for question {i}: {e}")
            continue # Skip to the next question if generation fails


        best_reasoning = None

        # Regex to find the answer and reasoning
        # Matches content after "Thought:" and before "The final answer is <answer>...</answer>"
        # and then extracts the numerical answer.
        answer_regex = re.compile(r"<answer>(.*?)</answer>")

        for text in generated_texts:
            # The 'text' here starts from "Thought: "
            # We need to extract the reasoning and the final answer
            match = answer_regex.search(text)
            #print("text", text)
            if match:
                extracted_answer_str = match.group(1)
                try:
                    # Attempt to convert to float for comparison
                    extracted_answer = float(extracted_answer_str)

                    # Round both for robust comparison
                    rounded_extracted_answer = round(extracted_answer, 2)
                    rounded_ground_truth_answer = round(ground_truth_answer, 2)
                    #print("rounded_extracted_answer",rounded_extracted_answer)
                    if rounded_extracted_answer == rounded_ground_truth_answer:
                        # Found a correct answer, extract reasoning
                        # The reasoning is everything before "The final answer is <answer>...</answer>"
                        # after "Thought: "
                        reasoning_end_index = text.find("The final answer is <answer>")
                        if reasoning_end_index != -1:
                            reasoning = text[:reasoning_end_index].strip()
                        else:
                            reasoning = text.replace(f"<answer>{extracted_answer_str}</answer>", "").strip()

                        # Remove "Thought:" if it's at the beginning of the extracted reasoning
                        if reasoning.startswith("Thought:"):
                            reasoning = reasoning[len("Thought:"):].strip()

                        best_reasoning = {
                            "question": question,
                            "reasoning": reasoning,
                            "answer": ground_truth_answer # Keep original float for consistency
                        }
                        success_count += 1
                        break # Move to the next question if a correct reasoning is found

                except ValueError:
                    # Ignore if the extracted answer is not a valid float
                    continue

        if best_reasoning:
            rft_data.append(best_reasoning)
        # else:
        #     print(f"No correct answer found for question: {question}") # Optional: for debugging

        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1} questions. Current success rate: {success_count / (i + 1) * 100:.2f}%")

    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(rft_data, f, indent=4)

    print(f"\nGenerated {len(rft_data)} RFT examples out of {len(train_dataset)} total questions.")
    print(f"Overall success rate: {success_count / len(train_dataset) * 100:.2f}%")


if __name__ == "__main__":
    from fire import Fire

    Fire(generate_dataset)
