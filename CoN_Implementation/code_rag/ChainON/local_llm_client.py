from __future__ import annotations
import torch
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer


class LocalLLM:
    """
    Client for running local models (Qwen, Llama, Mistral) on GPU.
    Supports both standard transformers and quantization for memory efficiency.
    """
    
    def __init__(
        self,
        model_path: str,
        max_output_tokens: int = 16000,
        temperature: float = 0.2,
        device: str = "cuda",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
    ) -> None:
        """
        Initialize local LLM.
        
        Args:
            model_path: Path to model on disk or HuggingFace model ID
                       Examples: 
                       - "/path/to/Qwen2.5-7B-Instruct"
                       - "Qwen/Qwen2.5-7B-Instruct"
                       - "meta-llama/Llama-3.1-8B-Instruct"
                       - "mistralai/Mistral-7B-Instruct-v0.3"
            max_output_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 = deterministic)
            device: Device to use ("cuda" or "cpu")
            load_in_8bit: Use 8-bit quantization (saves ~50% memory)
            load_in_4bit: Use 4-bit quantization (saves ~75% memory)
        """
        self.model_path = model_path
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.device = device
        
        print(f"Loading model from: {model_path}")
        print(f"Device: {device}")
        if load_in_8bit:
            print("Using 8-bit quantization")
        elif load_in_4bit:
            print("Using 4-bit quantization")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="left"
        )
        
        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Prepare model loading arguments
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16,
        }
        
        if load_in_8bit:
            model_kwargs["load_in_8bit"] = True
            model_kwargs["device_map"] = "auto"
        elif load_in_4bit:
            model_kwargs["load_in_4bit"] = True
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["device_map"] = "auto"
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **model_kwargs
        )
        
        print(f"Model loaded successfully!")
        print(f"Model max length: {self.tokenizer.model_max_length}")
    
    def generate(
        self,
        system: str,
        user: str,
        max_output_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        model: Optional[str] = None,  # Ignored, for API compatibility
        **kwargs
    ) -> str:
        """
        Generate response using the local model.
        
        Args:
            system: System prompt
            user: User message
            max_output_tokens: Override max tokens
            temperature: Override temperature
            model: Ignored (for compatibility with AnthropicLLM interface)
            
        Returns:
            Generated text
        """
        mot = max_output_tokens if max_output_tokens is not None else self.max_output_tokens
        temp = temperature if temperature is not None else self.temperature
        
        # Format prompt based on model type
        prompt = self._format_prompt(system, user)
        
        # Calculate max input length to leave room for generation
        max_input_length = self.tokenizer.model_max_length - mot
        if max_input_length < 100:
            print(f"Warning: Very little room for input tokens. Consider reducing max_output_tokens.")
            max_input_length = self.tokenizer.model_max_length - 100
            mot = 100
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_length
        ).to(self.device)
        
        input_length = inputs.input_ids.shape[1]
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=mot,
                temperature=temp,
                do_sample=temp > 0,
                top_p=0.9 if temp > 0 else None,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode only the generated portion (skip input)
        generated_ids = outputs[0][input_length:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        return generated_text.strip()
    
    def _format_prompt(self, system: str, user: str) -> str:
        """
        Format prompt according to model's chat template.
        Different models use different chat formats.
        """
        # Try to use model's built-in chat template first
        if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ]
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception as e:
                print(f"Warning: Could not apply chat template: {e}")
                print("Falling back to manual formatting...")
        
        # Fallback: Manual formatting for different model families
        model_name_lower = self.model_path.lower()
        
        if "qwen" in model_name_lower:
            # Qwen2 chat format
            return f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
        
        elif "llama-3" in model_name_lower or "llama3" in model_name_lower:
            # Llama 3.x format
            return f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        
        elif "llama" in model_name_lower:
            # Llama 2 format (older)
            combined = f"{system}\n\n{user}"
            return f"[INST] {combined} [/INST]"
        
        elif "mistral" in model_name_lower:
            # Mistral format (no separate system role)
            combined = f"{system}\n\n{user}"
            return f"[INST] {combined} [/INST]"
        
        else:
            # Generic fallback format
            print(f"Warning: Unknown model type, using generic format")
            return f"System: {system}\n\nUser: {user}\n\nAssistant:"


class VLLMClient:
    """
    Alternative client using vLLM for faster inference (2-4x speedup).
    Requires: pip install vllm --break-system-packages
    
    vLLM is optimized for throughput and uses PagedAttention for efficient memory.
    """
    
    def __init__(
        self,
        model_path: str,
        max_output_tokens: int = 16000,
        temperature: float = 0.2,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
    ) -> None:
        """
        Initialize vLLM client.
        
        Args:
            model_path: Path to model or HuggingFace ID
            max_output_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            tensor_parallel_size: Number of GPUs to use (1 for single GPU)
            gpu_memory_utilization: Fraction of GPU memory to use (0.0-1.0)
        """
        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            raise ImportError(
                "vLLM is not installed. Install with: "
                "pip install vllm --break-system-packages"
            )
        
        self.model_path = model_path
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        
        print(f"Loading model with vLLM from: {model_path}")
        print(f"Tensor parallel size: {tensor_parallel_size}")
        print(f"GPU memory utilization: {gpu_memory_utilization}")
        
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
        )
        
        self.tokenizer = self.llm.get_tokenizer()
        
        print("Model loaded successfully with vLLM!")
    
    def generate(
        self,
        system: str,
        user: str,
        max_output_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        model: Optional[str] = None,  # Ignored, for API compatibility
        **kwargs
    ) -> str:
        """
        Generate response using vLLM.
        
        Args:
            system: System prompt
            user: User message
            max_output_tokens: Override max tokens
            temperature: Override temperature
            model: Ignored (for compatibility)
            
        Returns:
            Generated text
        """
        from vllm import SamplingParams
        
        mot = max_output_tokens if max_output_tokens is not None else self.max_output_tokens
        temp = temperature if temperature is not None else self.temperature
        
        # Format prompt
        prompt = self._format_prompt(system, user)
        
        # Sampling parameters
        sampling_params = SamplingParams(
            temperature=temp,
            max_tokens=mot,
            top_p=0.9 if temp > 0 else 1.0,
        )
        
        # Generate (vLLM can handle batches, but we do single for simplicity)
        outputs = self.llm.generate([prompt], sampling_params)
        
        return outputs[0].outputs[0].text.strip()
    
    def _format_prompt(self, system: str, user: str) -> str:
        """
        Format prompt according to model's chat template.
        Same logic as LocalLLM._format_prompt.
        """
        # Try chat template first
        if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ]
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception:
                pass
        
        # Fallback to manual formatting
        model_name_lower = self.model_path.lower()
        
        if "qwen" in model_name_lower:
            return f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
        elif "llama-3" in model_name_lower or "llama3" in model_name_lower:
            return f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        elif "llama" in model_name_lower:
            combined = f"{system}\n\n{user}"
            return f"[INST] {combined} [/INST]"
        elif "mistral" in model_name_lower:
            combined = f"{system}\n\n{user}"
            return f"[INST] {combined} [/INST]"
        else:
            return f"System: {system}\n\nUser: {user}\n\nAssistant:"
