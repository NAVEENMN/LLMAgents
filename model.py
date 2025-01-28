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
        elif model == "llama":
            # Load LLaMA model and tokenizer
            model_name = "meta-llama/Llama-2-7b-hf"  # Replace with the specific LLaMA model you want
            self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.eval()
        else:
            raise ValueError(f"Model {model} not supported")

    def prompt(self, text: str, max_new_tokens: int = 50, temperature: float = 0.7, top_k: int = 50):
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