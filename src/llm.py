import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)
import argparse
import json
import re
import platform
import warnings

# Suppress unnecessary warnings
warnings.filterwarnings("ignore", message="`resume_download` is deprecated")
warnings.filterwarnings("ignore", message="urllib3.*")

def select_device():
    """Select the appropriate device: CUDA GPU, Mac GPU (MPS), or CPU."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"  # Metal Performance Shaders (Mac GPU)
    return "cpu"

class CustomDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=512):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(file_path, 'r') as f:
            raw_data = json.load(f)
            for item in raw_data:
                question = item["question"]
                answer = item["answer_cot"]
                text = f"Question: {question}\nAnswer: {answer}"
                self.data.append(text)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.data[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(),
            "attention_mask": encoded["attention_mask"].squeeze(),
            "labels": encoded["input_ids"].squeeze()
        }

class LLM(nn.Module):
    def __init__(self, name: str):
        super().__init__()
        self.name = name
        self.device = select_device()
        self.model = None
        self.tokenizer = None

    def set_model(self, model="gpt2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model, trust_remote_code=True).to(self.device)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        special_tokens = {"additional_special_tokens": ["<r>", "</r>"]}
        self.tokenizer.add_special_tokens(special_tokens)

    def fine_tune(self, train_file, output_dir, epochs=3, batch_size=4):
        dataset = CustomDataset(train_file, self.tokenizer)
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            save_steps=500,
            save_total_limit=2,
            logging_dir=f"{output_dir}/logs",
            logging_steps=10,
            evaluation_strategy="no",
            learning_rate=5e-5,
            weight_decay=0.01,
            report_to="none",
            push_to_hub=False
        )
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=self.tokenizer
        )
        trainer.train()
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)

    def evaluate(self, test_file, max_new_tokens=100, temperature=0.7):
        with open(test_file, 'r') as f:
            test_data = json.load(f)

        correct_predictions, missing_predictions = 0, 0
        total_samples = len(test_data)

        for item in test_data:
            question = item["question"]
            ground_truth = float(item["answer_value"])
            inputs = self.tokenizer(question, return_tensors="pt", truncation=True).to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )
            generated_cot = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
            match = re.search(r"<(\d+)>", generated_cot)
            predicted_answer = float(match.group(1)) if match else None
            if predicted_answer is None:
                missing_predictions += 1
            elif predicted_answer == ground_truth:
                correct_predictions += 1

        accuracy = correct_predictions / total_samples
        print(f"Accuracy: {accuracy:.2%}, Missing: {missing_predictions}/{total_samples}")
        return accuracy

    def prompt(self, text, max_new_tokens=100, temperature=0.7, top_k=50):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified LLM Operations")
    parser.add_argument("--task", type=str, required=True, choices=["fine_tune", "evaluate", "prompt"], help="Task to perform")
    parser.add_argument("--model", type=str, default="gpt2", help="Model name or path")
    parser.add_argument("--train_file", type=str, help="Path to training data")
    parser.add_argument("--test_file", type=str, help="Path to testing data")
    parser.add_argument("--output_dir", type=str, default="./fine_tuned_model", help="Output directory for fine-tuning")
    parser.add_argument("--prompt_text", type=str, help="Input prompt text")
    args = parser.parse_args()

    llm = LLM(name=args.model)
    llm.set_model(model=args.model)

    if args.task == "fine_tune":
        if not args.train_file:
            raise ValueError("Training file required for fine-tuning")
        llm.fine_tune(args.train_file, args.output_dir)

    elif args.task == "evaluate":
        if not args.test_file:
            raise ValueError("Testing file required for evaluation")
        llm.evaluate(args.test_file)

    elif args.task == "prompt":
        if not args.prompt_text:
            raise ValueError("Prompt text required for generating response")
        response = llm.prompt(args.prompt_text)
        print(f"Response: {response}")