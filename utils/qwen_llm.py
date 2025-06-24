from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import psutil
import warnings

class QwenLLM:
    def __init__(self, model_name_or_path="Qwen/Qwen3-1.7B", device=None):
        # Kiểm tra RAM khả dụng
        total_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        if total_ram_gb < 8:
            warnings.warn(f"Warning: Your system has only {total_ram_gb:.1f}GB RAM. Qwen3-1.7B may require at least 6-8GB RAM for inference.")
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Tối ưu cho CPU: ép dtype float32, tắt fp16, không load quantized
        if self.device == "cpu":
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                trust_remote_code=True
            )
        self.model = self.model.to(self.device)
        self.model.eval()

    def generate(self, prompt, max_new_tokens=28, temperature=0.7):
        # Tối ưu cho CPU: giảm max_new_tokens mặc định, batch nhỏ
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        result = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return result[len(prompt):].strip() if result.startswith(prompt) else result.strip() 