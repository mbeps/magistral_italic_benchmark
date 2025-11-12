"""
Configuration class for Magistral benchmarking with QLoRA support
"""

import os
from typing import Optional


class MagistralBenchmarkConfig:
    """Store and validate configuration for Magistral benchmarks.

    Holds model, tokenizer, evaluation, QLoRA and quantization options and
    performs basic validation of provided paths and parameter values.
    """

    def __init__(
        self,
        model_name: str = "mistralai/Magistral-Small-2506",
        tokenizer_name: str = "mistralai/Mistral-Nemo-Instruct-2407",
        batch_size: int = 8,
        min_vram_gb: int = 8,
        test_file: str = "./italic.jsonl",
        max_new_tokens: int = 350,
        max_eval_samples: int = None,
        system_message: str = "Sei un assistente utile e intelligente.",
        output_prefix: str = None,
        # QLoRA support parameters
        qlora_adapter_path: Optional[str] = None,
        merged_model_save_path: Optional[str] = None,
        # Quantization parameters
        use_quantization: bool = True,
        quantization_type: str = "nf4",
        quantization_compute_dtype: str = "float16",
        # Optimization parameters
        use_flash_attention: bool = True,
        use_torch_compile: bool = True,
        compile_mode: str = "max-autotune",
    ):
        """Create a benchmark config with sensible defaults and optional QLoRA.

        Purpose: initialize configuration fields and run validation when
        necessary (e.g., check test file and QLoRA adapter files).

        Args:
            model_name: model repo or identifier.
            tokenizer_name: tokenizer repo or identifier.
            batch_size: evaluation batch size.
            min_vram_gb: minimum VRAM required (GB).
            test_file: path to the JSONL test file.
            max_new_tokens: max tokens to generate per sample.
            max_eval_samples: maximum number of evaluation samples or None.
            system_message: assistant/system prompt.
            output_prefix: custom output prefix or None to auto-generate.
            qlora_adapter_path: path to a QLoRA adapter directory (optional).
            merged_model_save_path: path to save merged model (optional).
            use_quantization: whether to enable quantization.
            quantization_type: quantization scheme ('nf4'|'fp4').
            quantization_compute_dtype: dtype for quantized compute.
            use_flash_attention: enable Flash Attention.
            use_torch_compile: enable torch.compile.
            compile_mode: mode passed to torch.compile.
        """
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name
        self.batch_size = batch_size
        self.min_vram_gb = min_vram_gb
        self.test_file = test_file
        self.max_new_tokens = max_new_tokens
        self.max_eval_samples = max_eval_samples
        self.system_message = system_message

        # QLoRA parameters
        self.qlora_adapter_path = qlora_adapter_path
        self.merged_model_save_path = merged_model_save_path

        # Quantization parameters
        self.use_quantization = use_quantization
        self.quantization_type = quantization_type
        self.quantization_compute_dtype = quantization_compute_dtype

        # Optimization parameters
        self.use_flash_attention = use_flash_attention
        self.use_torch_compile = use_torch_compile
        self.compile_mode = compile_mode

        # Validate QLoRA configuration if specified
        if self.qlora_adapter_path:
            self._validate_qlora_config()

        # Auto-generate output prefix if not provided
        if output_prefix is None:
            model_size = "magistral_small"
            if self.qlora_adapter_path:
                adapter_name = os.path.basename(self.qlora_adapter_path.rstrip("/"))
                self.output_prefix = f"{model_size}_qlora_{adapter_name}"
            else:
                self.output_prefix = f"{model_size}_quantized"
        else:
            self.output_prefix = output_prefix

        # Validate configuration
        self._validate_config()

    def _validate_qlora_config(self):
        """Validate QLoRA adapter configuration"""
        if not os.path.exists(self.qlora_adapter_path):
            raise FileNotFoundError(
                f"QLoRA adapter path not found: {self.qlora_adapter_path}"
            )

        # Check for required QLoRA files
        required_files = ["adapter_config.json"]
        adapter_model_files = ["adapter_model.safetensors", "adapter_model.bin"]

        missing_files = []
        for file in required_files:
            if not os.path.exists(os.path.join(self.qlora_adapter_path, file)):
                missing_files.append(file)

        # Check for at least one adapter model file
        if not any(
            os.path.exists(os.path.join(self.qlora_adapter_path, f))
            for f in adapter_model_files
        ):
            missing_files.extend(adapter_model_files)

        if missing_files:
            raise FileNotFoundError(f"Missing QLoRA adapter files: {missing_files}")

        print(f"✅ QLoRA adapter found at: {self.qlora_adapter_path}")

    def _validate_config(self):
        """Validate configuration parameters"""
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")

        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")

        if not os.path.exists(self.test_file):
            raise FileNotFoundError(f"Test file not found: {self.test_file}")

        if self.quantization_type not in ["nf4", "fp4"]:
            raise ValueError("quantization_type must be 'nf4' or 'fp4'")

        if self.quantization_compute_dtype not in ["float16", "bfloat16"]:
            raise ValueError(
                "quantization_compute_dtype must be 'float16' or 'bfloat16'"
            )

    def print_config(self):
        """Print current configuration"""
        print(f"\n{'=' * 60}")
        print("MAGISTRAL BENCHMARK CONFIGURATION")
        if self.qlora_adapter_path:
            print("QLoRA ADAPTER SUPPORT ENABLED")
        print(f"{'=' * 60}")
        print(f"✓ Model: {self.model_name}")
        print(f"✓ Tokenizer: {self.tokenizer_name}")

        if self.qlora_adapter_path:
            print(f"✓ QLoRA Adapter: {self.qlora_adapter_path}")
            if self.merged_model_save_path:
                print(f"✓ Save merged model to: {self.merged_model_save_path}")

        print(f"✓ Test file: {self.test_file}")
        print(f"✓ Max samples: {self.max_eval_samples or 'All'}")
        print(f"✓ Batch size: {self.batch_size}")
        print(f"✓ Max new tokens: {self.max_new_tokens}")
        print(f"✓ Min VRAM required: {self.min_vram_gb}GB")

        if self.use_quantization:
            print(
                f"✓ Quantization: {self.quantization_type} ({self.quantization_compute_dtype})"
            )
        else:
            print("✓ Quantization: Disabled")

        print(
            f"✓ Flash Attention 2: {'Enabled' if self.use_flash_attention else 'Disabled'}"
        )
        print(f"✓ torch.compile: {'Enabled' if self.use_torch_compile else 'Disabled'}")
        print(f"✓ Output prefix: {self.output_prefix}")
        print(f"✓ System message: {self.system_message}")
        print(f"{'=' * 60}")
