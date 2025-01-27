import torch
import torch.nn as nn
import platform
from transformers import AutoTokenizer, AutoModelForCausalLM
from modelscope import AutoModelForCausalLM as QwenModelForCausalLM, AutoTokenizer as QwenTokenizer
import argparse
import warnings

# Suppress unnecessary warnings
warnings.filterwarnings("ignore", message="`resume_download` is deprecated")
warnings.filterwarnings("ignore", message="urllib3.*")

class LLM(nn.Module):
    def __init__(self, name: str):
        super().__init__()
        self.name = name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if platform.system() == "Darwin" and torch.backends.mps.is_available() else "cpu")
        self.model = None
        self.tokenizer = None

    def set_model(self, model="gpt2"):
        if model == "gpt2":
            # Load GPT-2 model and tokenizer
            self.model = AutoModelForCausalLM.from_pretrained('gpt2', trust_remote_code=True).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained('gpt2', trust_remote_code=True)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.eval()
        elif model == "qwen":
            # Load Qwen model and tokenizer
            model_name = "qwen/Qwen2-Math-72B"
            self.model = QwenModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="auto"
            )
            self.tokenizer = QwenTokenizer.from_pretrained(model_name)
        elif model == "galactica":
            # Load Galactica model and tokenizer
            model_name = "facebook/galactica-6.7b"
            self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.eval()
        else:
            raise ValueError(f"Model {model} not supported")

    def prompt(self, text: str, max_new_tokens: int = 50, temperature: float = 0.7, top_k: int = 50):
        if self.name == "qwen":
            # Prepare Qwen inputs
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": text}
            ]
            formatted_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            inputs = self.tokenizer([formatted_text], return_tensors="pt").to(self.device)

            # Generate text
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_k=top_k
                )
                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
                ]
                response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return response
        else:
            # Default prompt handling for GPT-2 and other models
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    max_new_tokens=max_new_tokens,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    do_sample=True,
                    temperature=temperature,
                    top_k=top_k
                )
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def print_results(self, generated_text: str):
        print("\nGenerated Text:")
        print("---------------")
        print(generated_text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Text Generation")
    parser.add_argument("--model", type=str, default="gpt2", help="Model name to use ('gpt2', 'qwen', or 'galactica')")
    parser.add_argument("--text", type=str, default="Find the value of $x$ that satisfies the equation $4x+5 = 6x+7$.", help="Input text")
    args = parser.parse_args()

    model = LLM(name=args.model)
    model.set_model(model=args.model)
    generated_text = model.prompt(args.text, max_new_tokens=100, temperature=0.5, top_k=100)
    model.print_results(generated_text)