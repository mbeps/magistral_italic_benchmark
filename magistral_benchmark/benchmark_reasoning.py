"""
Magistral Benchmark with Reasoning Support
Extension of base benchmark with thinking enabled and aggressive prompting
"""

import json
import re
import torch
import os
import gc
import warnings
import tempfile
import shutil
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import Optional, List, Dict, Any, Tuple

from .config import MagistralBenchmarkConfig


class MagistralReasoningBenchmark:
    """Magistral benchmark with reasoning/thinking support"""

    def __init__(self, config: MagistralBenchmarkConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.merged_model_path = None
        self.optimal_batch_size = None

        # Setup environment
        self._setup_environment()
        self._validate_hardware()

    def _setup_environment(self):
        """Setup environment and optimizations"""
        # Suppress warnings
        warnings.filterwarnings("ignore")
        os.environ["TRANSFORMERS_VERBOSITY"] = "error"

        # Load environment variables
        load_dotenv()
        hf_token = os.getenv("HF_TOKEN", "")
        if hf_token:
            os.environ["HF_TOKEN"] = hf_token

        # Set random seeds
        torch.manual_seed(42)
        np.random.seed(42)

        # CUDA optimizations
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            print("✅ CUDA optimizations enabled (TF32, cuDNN benchmark)")

    def _validate_hardware(self):
        """Validate hardware requirements"""
        if not torch.cuda.is_available():
            raise RuntimeError(
                "❌ CUDA not available! This benchmark requires GPU execution."
            )

        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")

        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"VRAM: {total_memory:.1f} GB")

        if total_memory < self.config.min_vram_gb:
            raise RuntimeError(
                f"❌ Insufficient GPU memory: {total_memory:.1f}GB < {self.config.min_vram_gb}GB required"
            )

        print(f"✅ GPU has sufficient VRAM for the model")

    def _merge_qlora_adapters(self) -> str:
        """Merge QLoRA adapters with base model (same as base implementation)"""
        print(f"\n🔄 Merging QLoRA adapters with base model...")
        print(f"Adapter path: {self.config.qlora_adapter_path}")
        print(f"Base model: {self.config.model_name}")

        try:
            from peft import AutoPeftModelForCausalLM
            from transformers import AutoTokenizer

            print("📦 Loading QLoRA adapter and base model...")

            peft_model = AutoPeftModelForCausalLM.from_pretrained(
                self.config.qlora_adapter_path,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
                token=os.environ.get("HF_TOKEN", ""),
            )

            print(f"✅ PEFT model loaded")
            print("🔄 Merging QLoRA adapters into base model...")
            merged_model = peft_model.merge_and_unload()
            print(f"✅ Adapters merged")

            if self.config.merged_model_save_path:
                merged_path = self.config.merged_model_save_path
                print(f"💾 Saving merged model to: {merged_path}")
            else:
                temp_dir = tempfile.mkdtemp(prefix="magistral_merged_")
                merged_path = temp_dir
                print(f"💾 Saving merged model to temporary directory: {merged_path}")

            os.makedirs(merged_path, exist_ok=True)
            merged_model.save_pretrained(merged_path, safe_serialization=True)

            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    self.config.tokenizer_name, 
                    trust_remote_code=True,
                    token=os.environ.get("HF_TOKEN", "")
                )
                tokenizer.save_pretrained(merged_path)
                print("✅ Tokenizer saved with merged model")
            except Exception as e:
                print(f"⚠️  Warning: Could not save tokenizer: {e}")

            del peft_model
            del merged_model
            self.clear_gpu_memory()

            print("✅ QLoRA adapters successfully merged!")
            return merged_path

        except ImportError as e:
            raise ImportError(
                "PEFT library is required for QLoRA adapter merging. "
                "Please install it with: pip install peft"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to merge QLoRA adapters: {str(e)}") from e

    @staticmethod
    def clear_gpu_memory():
        """Clear GPU memory and garbage collect"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()

    @staticmethod
    def load_jsonl(file_path: str) -> list:
        """Load JSONL file"""
        with open(file_path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]

    @staticmethod
    def create_uniform_subset(data: list, n_samples: int) -> list:
        """Create a subset with uniform distribution across categories"""
        import random
        random.seed(42)

        if n_samples >= len(data):
            return data

        category_groups = defaultdict(list)
        for question in data:
            category_groups[question["category"]].append(question)

        print(f"\nOriginal dataset category distribution:")
        for category, questions in sorted(category_groups.items()):
            print(f"  {category}: {len(questions)} questions")

        categories = list(category_groups.keys())
        n_categories = len(categories)
        base_samples_per_category = n_samples // n_categories
        remainder = n_samples % n_categories

        print(f"\nCreating uniform subset with {n_samples} samples:")

        selected_questions = []
        for i, category in enumerate(sorted(categories)):
            samples_for_category = base_samples_per_category + (
                1 if i < remainder else 0
            )
            available_questions = category_groups[category]
            actual_samples = min(samples_for_category, len(available_questions))

            selected = random.sample(available_questions, actual_samples)
            selected_questions.extend(selected)
            print(
                f"  {category}: selected {actual_samples} / {len(available_questions)} questions"
            )

        random.shuffle(selected_questions)
        print(f"\nFinal subset: {len(selected_questions)} questions")
        return selected_questions

    @staticmethod
    def extract_answer_with_reasoning(output):
        """Extract answer from reasoning output"""
        if not output or len(output.strip()) == 0:
            return ""

        output = output.strip()

        # Reasoning patterns for Italian output
        reasoning_patterns = [
            r"FINALE[:\s]*([ABCDE])",
            r"finale[:\s]*([ABCDE])",
            r"RISPOSTA FINALE[:\s]*([ABCDE])",
            r"risposta finale[:\s]*([ABCDE])",
            r"La risposta corretta è[:\s]*([ABCDE])",
            r"la risposta corretta è[:\s]*([ABCDE])",
            r"Quindi[,\s]*([ABCDE])",
            r"quindi[,\s]*([ABCDE])",
            r"Pertanto[,\s]*([ABCDE])",
            r"pertanto[,\s]*([ABCDE])",
            r"risposta[:\s]*([ABCDE])",
            r"Risposta[:\s]*([ABCDE])",
            r"lettera[:\s]*([ABCDE])",
            r"opzione[:\s]*([ABCDE])",
            r"scelta[:\s]*([ABCDE])",
        ]

        for pattern in reasoning_patterns:
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                return match.group(1).upper()

        # Check last few lines for answers
        lines = output.strip().split("\n")
        for line in reversed(lines[-5:]):
            line = line.strip()
            if line and len(line) <= 20:
                # Look for emphasized letters
                letter_patterns = [
                    r"\*\*([ABCDE])\*\*",  # **A**
                    r"\*([ABCDE])\*",      # *A*
                    r"^([ABCDE])$",        # Just the letter
                    r"\b([ABCDE])\b",      # Standalone letter
                ]
                for pattern in letter_patterns:
                    letter_match = re.search(pattern, line, re.IGNORECASE)
                    if letter_match:
                        return letter_match.group(1).upper()

        # Standard fallback patterns
        fallback_patterns = [
            r"\b([ABCDE])\)",  # A), B), C)
            r"\b([ABCDE])\.",  # A., B., C.
            r"\b([ABCDE])\b",  # Standalone letters
        ]

        for pattern in fallback_patterns:
            matches = list(re.finditer(pattern, output, re.IGNORECASE))
            if matches:
                return matches[-1].group(1).upper()

        return ""

    def configure_payload(self, question_data, system_message=None):
        """Configure message payload with aggressive reasoning prompt"""
        topic = question_data["category"]
        question = question_data["question"]
        options = question_data["options"]
        answer = question_data["answer"]

        # Format options
        formatted_options = "\n".join(
            [f"{list(item.keys())[0]}) {list(item.values())[0]}" for item in options]
        )
        letters = "".join([list(item.keys())[0] for item in options])

        # AGGRESSIVE prompt to minimize thinking time (adapted from Qwen3)
        user_content = f"""Rispondi alla seguente domanda a scelta multipla sull'argomento '{topic}'.

{question}

{formatted_options}

ISTRUZIONI CRITICHE:
1. Pensa VELOCEMENTE - massimo 3 frasi di ragionamento!
2. PIÙ PENSI, PIÙ SEI PENALIZZATO - sii RAPIDO!
3. NON sprecare tempo con lunghe riflessioni!
4. Concludi SEMPRE con "FINALE: X" dove X è una lettera tra {letters}
5. Se pensi più di 3 frasi, la risposta è SBAGLIATA!

Ragiona BREVEMENTE e rispondi ORA:"""

        messages = [
            {
                "role": "system",
                "content": system_message or self.config.system_message,
            },
            {
                "role": "user",
                "content": user_content,
            },
        ]

        return messages, answer

    def _determine_optimal_batch_size(self):
        """Determine optimal batch size for reasoning mode (slightly smaller)"""
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            available_memory = total_memory - memory_reserved

            print(f"GPU Memory - Allocated: {memory_allocated:.2f} GB, Available: {available_memory:.2f} GB")
            
            # Slightly smaller batch sizes for reasoning mode
            if available_memory > 10:
                self.optimal_batch_size = min(self.config.batch_size, 8)
            elif available_memory > 6:
                self.optimal_batch_size = min(self.config.batch_size, 6)
            elif available_memory > 3:
                self.optimal_batch_size = min(self.config.batch_size, 4)
            else:
                self.optimal_batch_size = 2
            
            print(f"🚀 Optimal batch size for reasoning mode: {self.optimal_batch_size}")
        else:
            self.optimal_batch_size = max(2, self.config.batch_size - 2)

    def load_model(self):
        """Load Magistral model with reasoning support"""
        # Determine which model to load
        if self.config.qlora_adapter_path:
            print(f"🔄 QLoRA adapter specified - will merge with base model first")
            model_path = self._merge_qlora_adapters()
            self.merged_model_path = model_path
            model_description = f"{self.config.model_name} + QLoRA adapter"
        else:
            model_path = self.config.model_name
            model_description = self.config.model_name

        print(f"🔄 Loading {model_description} with REASONING support...")

        # Load tokenizer
        print(f"Loading tokenizer: {self.config.tokenizer_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.tokenizer_name,
            token=os.environ.get("HF_TOKEN", ""),
            padding_side="left",
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Check Flash Attention availability
        flash_attn_available = False
        if self.config.use_flash_attention:
            try:
                import flash_attn
                flash_attn_available = True
                print("✅ Flash Attention 2 detected - will use for optimization")
            except ImportError:
                print("⚠️ Flash Attention 2 not available - using standard attention")

        # Configure quantization
        quantization_config = None
        if self.config.use_quantization:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=getattr(torch, self.config.quantization_compute_dtype),
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type=self.config.quantization_type
            )

        # Load model
        print("Loading model with optimizations...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16,
            token=os.environ.get("HF_TOKEN", ""),
            low_cpu_mem_usage=True,
            attn_implementation="flash_attention_2" if flash_attn_available else None,
            trust_remote_code=True,
        )

        # Apply torch.compile if enabled
        if self.config.use_torch_compile:
            try:
                print("Compiling model with torch.compile for optimization...")
                self.model = torch.compile(self.model, mode=self.config.compile_mode)
                print("✅ Model compiled successfully!")
            except Exception as e:
                print(f"⚠️ Model compilation failed: {e}")
                print("Continuing without compilation...")

        print(f"✅ Model loaded with REASONING support!")
        print(f"✅ Sampling params: temp=0.1, top_p=0.95, top_k=20 (reasoning mode)")
        if self.config.qlora_adapter_path:
            print(f"✅ QLoRA adapter merged and loaded")

        # Determine optimal batch size
        self._determine_optimal_batch_size()

    def generate_batch_responses(self, batch_messages, max_tokens=None):
        """Generate responses with reasoning-specific parameters"""
        if max_tokens is None:
            max_tokens = self.config.max_new_tokens + 150  # Extra tokens for reasoning
            
        # Prepare batch prompts
        batch_prompts = []
        for messages in batch_messages:
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            batch_prompts.append(formatted_prompt)

        # Tokenize batch with padding
        inputs = self.tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
        )

        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Generate with controlled parameters for reasoning mode
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.1,   # Lower temperature for focused reasoning
                top_p=0.95,        # Higher top_p for reasoning diversity
                top_k=20,          # Controlled top_k for efficiency
                repetition_penalty=1.05,  # Prevent repetition loops
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
                early_stopping=True,
            )

        # Decode batch responses
        batch_responses = []
        for i in range(len(batch_prompts)):
            input_length = inputs['input_ids'][i].shape[0]
            new_tokens = outputs[i][input_length:]
            response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            batch_responses.append(response.strip())

        return batch_responses

    def evaluate_model(self, data: list) -> tuple:
        """Evaluate model with reasoning on dataset"""
        model_description = f"{self.config.model_name}"
        if self.config.qlora_adapter_path:
            adapter_name = os.path.basename(self.config.qlora_adapter_path.rstrip("/"))
            model_description += f" + QLoRA ({adapter_name})"

        print(f"\n🤔 Evaluating {model_description} WITH REASONING on {len(data)} questions...")
        print("⚡ Using aggressive prompting to minimize thinking time")

        results = []
        correct = 0
        total = 0
        category_stats = defaultdict(lambda: {"correct": 0, "total": 0})

        # Process in batches
        for batch_start in tqdm(
            range(0, len(data), self.optimal_batch_size), desc="Evaluating (with reasoning)"
        ):
            batch_end = min(batch_start + self.optimal_batch_size, len(data))
            batch_data = data[batch_start:batch_end]

            try:
                batch_messages = []
                batch_answers = []

                for item in batch_data:
                    messages, correct_answer = self.configure_payload(item)
                    batch_messages.append(messages)
                    batch_answers.append(correct_answer)

                # Generate responses with reasoning
                responses = self.generate_batch_responses(batch_messages)

                # Process batch results
                for i, (item, correct_answer, response) in enumerate(
                    zip(batch_data, batch_answers, responses)
                ):
                    # Extract answer from reasoning output
                    predicted = self.extract_answer_with_reasoning(response)
                    is_correct = predicted == correct_answer

                    # Update statistics
                    category = item["category"]
                    category_stats[category]["total"] += 1
                    if is_correct:
                        correct += 1
                        category_stats[category]["correct"] += 1
                    total += 1

                    # Store result
                    result = {
                        "index": batch_start + i,
                        "question": item["question"],
                        "category": category,
                        "correct_answer": correct_answer,
                        "predicted_answer": predicted,
                        "raw_response": response,
                        "reasoning_length": len(response),
                        "is_correct": is_correct,
                    }
                    results.append(result)

            except Exception as e:
                print(f"Error processing batch starting at {batch_start}: {e}")
                for i, item in enumerate(batch_data):
                    result = {
                        "index": batch_start + i,
                        "question": item["question"],
                        "category": item["category"],
                        "correct_answer": item["answer"],
                        "predicted_answer": "",
                        "raw_response": f"ERROR: {str(e)}",
                        "reasoning_length": 0,
                        "is_correct": False,
                    }
                    results.append(result)
                    category_stats[item["category"]]["total"] += 1
                    total += 1

            # Memory cleanup
            if (batch_start // self.optimal_batch_size) % 10 == 0:
                self.clear_gpu_memory()

                if total > 0 and total % 50 == 0:
                    current_accuracy = correct / total
                    avg_reasoning = sum(r['reasoning_length'] for r in results[-50:]) / min(50, len(results))
                    print(
                        f"Progress: {total}/{len(data)} ({total / len(data) * 100:.1f}%), "
                        f"Accuracy: {current_accuracy:.4f} ({current_accuracy * 100:.2f}%), "
                        f"Avg reasoning: {avg_reasoning:.0f} chars"
                    )

        accuracy = correct / total if total > 0 else 0

        print(f"\n📊 FINAL RESULTS (WITH REASONING):")
        print(f"Total questions: {total}")
        print(f"Correct answers: {correct}")
        print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

        if results:
            valid_reasoning = [r['reasoning_length'] for r in results if r['reasoning_length'] > 0]
            if valid_reasoning:
                avg_reasoning_length = sum(valid_reasoning) / len(valid_reasoning)
                print(f"Average reasoning length: {avg_reasoning_length:.0f} characters")

        return results, accuracy, category_stats

    @staticmethod
    def analyse_results_by_category(category_stats: dict):
        """Analyse results by category"""
        print(f"\n📈 RESULTS BY CATEGORY (REASONING MODE):")
        print("-" * 60)
        print(f"{'Category':25s} {'Accuracy':>12s} {'Correct':>8s} {'Total':>8s}")
        print("-" * 60)

        for category, stats in sorted(category_stats.items()):
            accuracy = (
                stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
            )
            print(
                f"{category:25s} {accuracy:11.2f}% {stats['correct']:>8d} {stats['total']:>8d}"
            )

        print("-" * 60)

    @staticmethod
    def show_sample_predictions(results: list, n_samples: int = 5):
        """Show sample predictions with reasoning info"""
        print(f"\n🔍 SAMPLE PREDICTIONS (WITH REASONING):")

        correct_samples = [r for r in results if r["is_correct"]][: n_samples // 2]
        incorrect_samples = [r for r in results if not r["is_correct"]][
            : n_samples // 2
        ]

        samples = correct_samples + incorrect_samples

        for i, result in enumerate(samples[:n_samples]):
            status = "✅ CORRECT" if result["is_correct"] else "❌ INCORRECT"
            print(f"\nExample {i + 1} {status}:")
            print(f"  Category: {result['category']}")
            print(f"  Question: {result['question'][:100]}...")
            print(f"  Expected: {result['correct_answer']}")
            print(f"  Predicted: {result['predicted_answer']}")
            print(f"  Reasoning length: {result['reasoning_length']} chars")
            print(f"  Response excerpt: '{result['raw_response'][:150]}...'")

    def save_results(self, results: list, accuracy: float, category_stats: dict):
        """Save benchmark results with reasoning metadata"""
        print("\n💾 Saving reasoning benchmark results...")

        results_df = pd.DataFrame(results)
        results_file = f"{self.config.output_prefix}_reasoning_results.csv"
        results_df.to_csv(results_file, index=False)
        print(f"Detailed results saved to '{results_file}'")

        model_info = {
            "base_model": self.config.model_name,
            "tokenizer": self.config.tokenizer_name,
            "evaluation_type": "zero-shot-with-reasoning",
            "reasoning_enabled": True,
            "quantization": f"{self.config.quantization_type} ({self.config.quantization_compute_dtype})" if self.config.use_quantization else "none",
            "flash_attention": self.config.use_flash_attention,
            "torch_compile": self.config.use_torch_compile,
        }

        if self.config.qlora_adapter_path:
            adapter_name = os.path.basename(self.config.qlora_adapter_path.rstrip("/"))
            model_info.update(
                {
                    "fine_tuning": "QLoRA",
                    "qlora_adapter_path": self.config.qlora_adapter_path,
                    "qlora_adapter_name": adapter_name,
                    "merged_model_path": self.merged_model_path,
                    "model_description": f"{self.config.model_name} + QLoRA ({adapter_name})",
                }
            )
        else:
            model_info.update(
                {"fine_tuning": "none", "model_description": self.config.model_name}
            )

        summary = {
            "model_info": model_info,
            "dataset_info": {
                "test_file": self.config.test_file,
                "total_questions_tested": len(results),
                "max_eval_samples": self.config.max_eval_samples,
            },
            "generation_config": {
                "system_message": self.config.system_message,
                "max_tokens": self.config.max_new_tokens + 150,
                "batch_size": self.optimal_batch_size,
                "temperature": 0.1,
                "top_p": 0.95,
                "top_k": 20,
                "prompt_style": "aggressive_reasoning",
            },
            "results": {
                "total_questions": len(results),
                "correct_answers": sum(result["is_correct"] for result in results),
                "accuracy": accuracy,
                "accuracy_percent": accuracy * 100,
                "average_reasoning_length": sum(r['reasoning_length'] for r in results if r['reasoning_length'] > 0) / len([r for r in results if r['reasoning_length'] > 0]) if results else 0,
                "category_results": {
                    cat: {
                        "accuracy": stats["correct"] / stats["total"],
                        "correct": stats["correct"],
                        "total": stats["total"],
                    }
                    for cat, stats in category_stats.items()
                },
            },
        }

        summary_file = f"{self.config.output_prefix}_reasoning_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"Summary saved to '{summary_file}'")

    def run_benchmark(self):
        """Run the complete reasoning benchmark"""
        print(f"\n{'=' * 60}")
        print("MAGISTRAL REASONING BENCHMARK")
        print("🤔 THINKING MODE ENABLED")
        if self.config.qlora_adapter_path:
            print("🔄 WITH QLORA ADAPTER SUPPORT")
        print(f"{'=' * 60}")

        # Print configuration
        self.config.print_config()
        print(f"✓ Reasoning: ENABLED")
        print(f"✓ Aggressive prompting: YES (max 3 sentences)")

        # Load dataset
        print(f"\n📚 Loading ITALIC dataset...")
        data = self.load_jsonl(self.config.test_file)
        print(f"Loaded {len(data)} questions")

        # Create subset if specified
        if self.config.max_eval_samples and self.config.max_eval_samples < len(data):
            test_data = self.create_uniform_subset(data, self.config.max_eval_samples)
        else:
            test_data = data

        # Show dataset statistics
        categories = defaultdict(int)
        for item in test_data:
            categories[item["category"]] += 1
        print(f"\nUsing {len(test_data)} questions for evaluation")
        print(f"Categories: {dict(sorted(categories.items()))}")

        # Load model with reasoning support
        self.load_model()

        # Test inference with reasoning
        print(f"\n🧪 Testing inference with REASONING enabled...")
        test_messages, test_answer = self.configure_payload(test_data[0])
        test_responses = self.generate_batch_responses([test_messages])
        test_response = test_responses[0]
        test_extracted = self.extract_answer_with_reasoning(test_response)

        print(f"Question: {test_data[0]['question'][:100]}...")
        print(f"Test response length: {len(test_response)} chars")
        print(f"Expected answer: '{test_answer}'")
        print(f"Extracted answer: '{test_extracted}'")
        print(f"Correct: {test_extracted == test_answer}")

        # Run evaluation
        print(f"\n{'=' * 50}")
        print("STARTING REASONING EVALUATION")
        print(f"{'=' * 50}")

        results, accuracy, category_stats = self.evaluate_model(test_data)

        # Analyse and save results
        self.analyse_results_by_category(category_stats)
        self.show_sample_predictions(results, 5)
        self.save_results(results, accuracy, category_stats)

        # Final summary
        model_description = self.config.model_name
        if self.config.qlora_adapter_path:
            adapter_name = os.path.basename(self.config.qlora_adapter_path.rstrip("/"))
            model_description += f" + QLoRA ({adapter_name})"

        print(f"\n🎉 REASONING EVALUATION COMPLETED!")
        print("=" * 60)
        print(f"📊 Model: {model_description}")
        print(f"📊 Final accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
        print(f"📊 Total questions evaluated: {len(results)}")
        print(f"📊 Batch size used: {self.optimal_batch_size}")
        print(f"🤔 Reasoning mode: ENABLED")
        print(f"⚡ Aggressive prompting: YES")
        if self.config.qlora_adapter_path:
            print(f"📊 QLoRA adapter merged and evaluated")
        print("=" * 60)

        # Cleanup
        print(f"\n🧹 Final cleanup...")
        if hasattr(self, 'model') and self.model:
            del self.model
        if hasattr(self, 'tokenizer') and self.tokenizer:
            del self.tokenizer
        self.clear_gpu_memory()

        # Clean up temporary merged model if created
        if (
            self.merged_model_path
            and self.merged_model_path != self.config.merged_model_save_path
            and self.merged_model_path.startswith(tempfile.gettempdir())
        ):
            try:
                shutil.rmtree(self.merged_model_path)
                print(f"🗑️  Cleaned up temporary merged model: {self.merged_model_path}")
            except Exception as e:
                print(f"⚠️  Warning: Could not clean up temporary directory: {e}")

        final_memory = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
        print(f"Final GPU memory usage: {final_memory:.2f}GB")
        print(f"✅ {model_description} reasoning benchmark complete! 🚀")

        return results, accuracy, category_stats
