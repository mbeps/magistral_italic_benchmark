"""
Magistral Benchmark Package
A modular package for benchmarking Magistral-Small on the ITALIC dataset.
Supports both standard (non-thinking) and reasoning (thinking) evaluation modes.
"""

from .config import MagistralBenchmarkConfig
from .benchmark import MagistralBenchmark
from .benchmark_reasoning import MagistralReasoningBenchmark

__all__ = [
    "MagistralBenchmarkConfig",
    "MagistralBenchmark",
    "MagistralReasoningBenchmark",
]
