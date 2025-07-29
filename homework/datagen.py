import json
import re
from pathlib import Path
from .cot import CoTModel



from .base_llm import BaseLLM
from .data import Dataset # Assuming CoTModel is in .data

def generate_dataset(output_json: str="data/rft.json", oversample: int = 20, temperature: float = 0.6):
    """
    Generates a dataset for Reasoning Feedback Training (RFT).
    It uses a CoTModel to generate multiple completions for each question,
    selects the one with the correct answer, and stores it along with the reasoning.

    Args:
        output_json (str): Path to the output JSON file.
        oversample (int): Number of completions to generate for each question.
        temperature (float): Sampling temperature for generation.
    """
    # Initialize CoTModel with the specified Instruct model
    cot_llm = CoTModel()

    # Load the training dataset
    train_dataset = Dataset("train")

    rft_data = []
    success_count = 0

    print(f"Generating RFT dataset with {oversample} oversamples per question...")

    for i, (question, ground_truth_answer) in enumerate(train_dataset):
        # Format the question using the CoTModel's chat template
        formatted_prompt = cot_llm.format_prompt(question)

        try:
            # Generate multiple completions for the current question
            # batched_generate returns list[list[str]] when num_return_sequences is set
            generated_texts_for_question = cot_llm.batched_generate(
                prompts=[formatted_prompt], # Pass as a list as batched_generate expects a list
                num_return_sequences=oversample,
                temperature=temperature
            )[0] # Get the list of generations for the single prompt
        except Exception as e:
            print(f"Error generating for question {i} ('{question}'): {e}")
            continue # Skip to the next question if generation fails

        best_entry_found = None

        for generated_text in generated_texts_for_question:
            # Parse the answer from the generated text
            extracted_answer = cot_llm.parse_answer(generated_text)

            # Check if the extracted answer is a valid number and matches the ground truth
            if not isinstance(extracted_answer, float) or extracted_answer == float("nan"):
                continue # Skip if parsing failed or it's not a number

            # Round both for robust comparison, especially with floating point numbers
            rounded_extracted_answer = round(extracted_answer, 2)
            rounded_ground_truth_answer = round(ground_truth_answer, 2)

            if rounded_extracted_answer == rounded_ground_truth_answer:
                # If the answer is correct, store the full generated text as reasoning
                # The prompt asks for the format: [question, answer, reasoning_text_with_answer_tag]
                best_entry_found = [
                    question,
                    ground_truth_answer,
                    generated_text.strip() # Store the full generated text as the reasoning+answer
                ]
                success_count += 1
                break # Move to the next question as we found a correct completion

        if best_entry_found:
            rft_data.append(best_entry_found)
        # else:
        #     print(f"No correct answer found for question: {question}") # Optional: for debugging

        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1} questions. Current success rate: {success_count / (i + 1) * 100:.2f}%")

    # Ensure the output directory exists
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the generated RFT dataset to a JSON file
    with open(output_path, "w") as f:
        json.dump(rft_data, f, indent=4)

    print(f"\nGenerated {len(rft_data)} RFT examples out of {len(train_dataset)} total questions.")
    print(f"Overall success rate: {success_count / len(train_dataset) * 100:.2f}%")


if __name__ == "__main__":
    from fire import Fire

    Fire(generate_dataset)

