from .base_llm import BaseLLM
from .sft import test_model
from peft import LoraConfig, get_peft_model
import torch
from transformers import Trainer, TrainingArguments
from pathlib import Path
from .data import Dataset, benchmark


def format_example(prompt: str, reasoning: str,answer: str) -> dict[str, str]:
    """
    Construct a question / answer pair. Consider rounding the answer to make it easier for the LLM.
    """
    #rounded_answer = round(answer, 2)
    return {
            "question": prompt,
            "reasoning": reasoning,
            "answer": f"<answer>{answer}</answer>"
    }

class TokenizedDataset:
    def __init__(self, tokenizer, data: Dataset, format_fn):
        """
        Use the
        - BaseLLM.tokenizer
        - Dataset
        - format_fn which converts a data element into a dict with entries
          - question: str
          - answer: str
        """
        self.format_fn = format_fn
        self.tokenizer = tokenizer
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        formated_data = self.format_fn(*self.data[idx])
        return tokenize(self.tokenizer, **formated_data)

def tokenize(tokenizer, question: str, reasoning: str, answer: str):
    """
    Tokenize a data element.
    We first append the <EOS> token to the question / answer pair.
    Then we tokenize and construct the ground truth `labels`.
    `labels[i] == -100` for the question or masked out parts, since we only want to supervise
    the answer.
    """
    full_text = f"{question}{reasoning}{answer}{tokenizer.eos_token}"

    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    full = tokenizer(full_text, padding="max_length", truncation=True)

    input_ids = full["input_ids"]
    question_len = len(tokenizer(question)["input_ids"])

    # Create labels: mask out the prompt part
    labels = [-100] * question_len + input_ids[question_len:]

    for i in range(len(labels)):
        if full["attention_mask"][i] == 0:
            labels[i] = -100

    full["labels"] = labels
    return full

def load() -> BaseLLM:
    from pathlib import Path

    from peft import PeftModel

    model_name = "rft_model"
    model_path = Path(__file__).parent / model_name

    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()

    return llm


def train_model(
    output_dir: str="homework/rft_model",
    **kwargs,
):
    # Reuse much of the SFT code here
    #raise NotImplementedError()
        llm = BaseLLM()

        # Configure LoRA
        lora_config = LoraConfig(
            r=32,  # Rank, chosen to keep model size below 20MB
            lora_alpha=128, # 4-5 times the rank
            target_modules="all-linear",
            bias="none",
            task_type="CAUSAL_LM",
        )

        # Convert the base model to a LoRA adapted version
        llm.model = get_peft_model(llm.model, lora_config)
        llm.model.print_trainable_parameters()

        # Enable input_require_grads for gradient_checkpointing
        if torch.cuda.is_available():
            llm.model.enable_input_require_grads()

        # Define TrainingArguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            logging_dir=output_dir,
            report_to="tensorboard",
            gradient_checkpointing=True,
            learning_rate=1e-4,  # A reasonable learning rate
            num_train_epochs=5,
            per_device_train_batch_size=32,
            save_strategy="epoch", # Save checkpoint at the end of each epoch
            logging_steps=10,
        )

        # Prepare datasets
        train_dataset = Dataset("rft")
        tokenized_train_dataset = TokenizedDataset(llm.tokenizer, train_dataset, format_example)

        # Initialize Trainer
        trainer = Trainer(
            model=llm.model,
            args=training_args,
            train_dataset=tokenized_train_dataset,
        )

        # Train the model
        trainer.train()

        # Save the final model checkpoint to the specified directory
        model_save_path = Path("homework") / "rft_model"
        model_save_path.mkdir(parents=True, exist_ok=True)
        trainer.save_model(model_save_path)
        print("output_dir", output_dir)
        test_model(output_dir) # You might want to test the saved model instead of the trainer's output_dir if you prefer.


if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})
