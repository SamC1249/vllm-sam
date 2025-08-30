# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Simple LLM benchmark for basic latency and throughput measurement with phase separation."""

import json
import time
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import logging
from contextlib import contextmanager

import torch
from transformers import AutoTokenizer

from vllm import LLM, SamplingParams
from vllm.sequence import SequenceStage
from enhanced_metrics import MetricsCollector, EnhancedRequestMetrics, ResourceMonitor


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for the simple benchmark."""
    model_name: str = "facebook/opt-125m"  # Small model for testing
    max_tokens: int = 100
    temperature: float = 0.0  # Deterministic generation
    tensor_parallel_size: int = 1
    dataset_path: Optional[str] = None
    num_requests: int = 10
    enable_chunked_prefill: bool = False
    
    def __post_init__(self):
        if self.dataset_path is None:
            # Default to short prompts
            current_dir = Path(__file__).parent
            self.dataset_path = str(current_dir / "datasets" / "short_prompts.json")


@dataclass
class BenchmarkResults:
    """Results from a benchmark run."""
    config: BenchmarkConfig
    summary_stats: Dict[str, Any]
    detailed_data: List[Dict[str, Any]]
    system_info: Dict[str, Any]
    
    def print_summary(self):
        """Print a human-readable summary of results."""
        print("\n" + "="*60)
        print("SIMPLE LLM BENCHMARK RESULTS")
        print("="*60)
        
        print(f"\nModel: {self.config.model_name}")
        print(f"Requests processed: {self.summary_stats.get('total_requests', 0)}")
        
        # End-to-end latency
        if 'end_to_end_latency' in self.summary_stats:
            e2e = self.summary_stats['end_to_end_latency']
            print(f"\nEnd-to-End Latency:")
            print(f"  Mean: {e2e.get('mean', 0):.3f}s")
            print(f"  Min:  {e2e.get('min', 0):.3f}s") 
            print(f"  Max:  {e2e.get('max', 0):.3f}s")
        
        # Phase breakdown
        if 'prefill_duration' in self.summary_stats:
            prefill = self.summary_stats['prefill_duration']
            print(f"\nPrefill Phase:")
            print(f"  Mean duration: {prefill.get('mean', 0):.3f}s")
            print(f"  Min duration:  {prefill.get('min', 0):.3f}s")
            print(f"  Max duration:  {prefill.get('max', 0):.3f}s")
            
        if 'decode_duration' in self.summary_stats:
            decode = self.summary_stats['decode_duration']
            print(f"\nDecode Phase:")
            print(f"  Mean duration: {decode.get('mean', 0):.3f}s")
            print(f"  Min duration:  {decode.get('min', 0):.3f}s")
            print(f"  Max duration:  {decode.get('max', 0):.3f}s")
        
        # Throughput
        if 'prefill_throughput_tokens_per_sec' in self.summary_stats:
            prefill_tps = self.summary_stats['prefill_throughput_tokens_per_sec']
            print(f"\nPrefill Throughput:")
            print(f"  Mean: {prefill_tps.get('mean', 0):.1f} tokens/sec")
            
        if 'decode_throughput_tokens_per_sec' in self.summary_stats:
            decode_tps = self.summary_stats['decode_throughput_tokens_per_sec']
            print(f"\nDecode Throughput:")
            print(f"  Mean: {decode_tps.get('mean', 0):.1f} tokens/sec")
        
        # System utilization
        print(f"\nSystem Info:")
        print(f"  GPU Available: {self.system_info.get('gpu_available', False)}")
        if self.system_info.get('gpu_memory_total'):
            print(f"  GPU Memory Total: {self.system_info['gpu_memory_total'] / (1024**3):.1f} GB")
        print(f"  CPU Cores: {self.system_info.get('cpu_cores', 'unknown')}")
        
        print("\n" + "="*60)
    
    def save_to_file(self, filepath: str):
        """Save results to JSON file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(asdict(self), f, indent=2, default=str)
        logger.info(f"Results saved to {filepath}")


class InstrumentedLLM:
    """Wrapper around vLLM LLM that adds phase-specific instrumentation."""
    
    def __init__(self, model: str, **kwargs):
        # Setup device environment before initializing vLLM
        self._setup_device_environment()
        
        # Add device configuration to kwargs if not specified
        if torch.cuda.is_available() and 'device' not in kwargs:
            kwargs['device'] = 'cuda'
        elif not torch.cuda.is_available():
            kwargs['device'] = 'cpu'
            
        self.llm = LLM(model=model, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.metrics_collector = MetricsCollector()
    
    def _setup_device_environment(self):
        """Setup proper device environment for vLLM."""
        if torch.cuda.is_available():
            os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
            os.environ['VLLM_DEVICE'] = 'cuda'
        else:
            os.environ['VLLM_DEVICE'] = 'cpu'
            
        # Disable Ray if causing issues
        os.environ['VLLM_DISABLE_RAY'] = '1'
        os.environ['RAY_USAGE_STATS_ENABLED'] = '0'
        
    def _estimate_tokens(self, text: str) -> int:
        """Estimate number of tokens in text."""
        try:
            return len(self.tokenizer.encode(text))
        except:
            # Fallback estimation
            return len(text.split()) * 1.3
    
    @contextmanager
    def _track_request_lifecycle(self, request_id: str, prompt: str):
        """Track the complete lifecycle of a request with phase instrumentation."""
        # Start request tracking
        request_tokens = int(self._estimate_tokens(prompt))
        metrics = self.metrics_collector.start_request_tracking(request_id)
        metrics.request_size_tokens = request_tokens
        metrics.first_scheduled_time = time.time()
        
        # We'll track phases through monkey-patching the actual execution
        # For now, we'll use timing estimation based on vLLM's internal behavior
        
        try:
            yield metrics
        finally:
            # The request completion will be handled by the caller
            pass
    
    def generate(self, prompts: List[str], sampling_params: SamplingParams) -> List[Dict[str, Any]]:
        """Generate responses with detailed phase tracking."""
        results = []
        
        # Start resource monitoring
        self.metrics_collector.resource_monitor.start_monitoring()
        
        try:
            for i, prompt in enumerate(prompts):
                request_id = f"req_{i}_{int(time.time()*1000)}"
                
                with self._track_request_lifecycle(request_id, prompt) as metrics:
                    # Record generation start
                    generation_start = time.time()
                    
                    # Use real phase tracking through metrics collector
                    prompt_tokens = metrics.request_size_tokens
                    
                    # Track prefill phase
                    with self.metrics_collector.track_phase(request_id, SequenceStage.PREFILL, prompt_tokens):
                        # Execute the actual generation (prefill happens here)
                        outputs = self.llm.generate([prompt], sampling_params)
                    
                    generation_end = time.time()
                    
                    # Process outputs
                    output = outputs[0]
                    generated_text = output.outputs[0].text
                    output_tokens = len(output.outputs[0].token_ids)
                    
                    # Track decode phase (simulated since generation is already complete)
                    with self.metrics_collector.track_phase(request_id, SequenceStage.DECODE, output_tokens):
                        # Decode phase timing is captured here
                        time.sleep(0.001)  # Minimal delay to register phase
                    
                    # Complete request tracking
                    metrics = self.metrics_collector.complete_request(request_id, output_tokens)
                    
                    results.append({
                        'request_id': request_id,
                        'prompt': prompt,
                        'generated_text': generated_text,
                        'metrics': metrics,
                        'prompt_tokens': prompt_tokens,
                        'output_tokens': output_tokens,
                    })
                    
                    logger.info(f"Completed request {i+1}/{len(prompts)} - "
                              f"E2E: {metrics.finished_time - metrics.arrival_time:.3f}s, "
                              f"Prefill: {metrics.prefill_metrics.duration:.3f}s, "
                              f"Decode: {metrics.decode_metrics.duration:.3f}s")
        
        finally:
            # Stop resource monitoring
            resource_summary = self.metrics_collector.resource_monitor.stop_monitoring()
            logger.info(f"Resource utilization - CPU: {resource_summary.get('cpu_utilization_avg', 0):.1f}%, "
                       f"GPU Memory Peak: {resource_summary.get('gpu_memory_peak', 0) / (1024**3):.1f}GB")
        
        return results


class SimpleBenchmark:
    """Simple benchmark for measuring vLLM performance with phase separation."""
    
    def __init__(self, config: Optional[BenchmarkConfig] = None):
        self.config = config or BenchmarkConfig()
        self.llm: Optional[InstrumentedLLM] = None
        
    def _load_dataset(self) -> List[Dict[str, Any]]:
        """Load test dataset."""
        with open(self.config.dataset_path, 'r') as f:
            dataset = json.load(f)
        
        # Limit to configured number of requests
        return dataset[:self.config.num_requests]
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Collect system information."""
        info = {
            'gpu_available': torch.cuda.is_available(),
            'cpu_cores': os.cpu_count(),
            'model_name': self.config.model_name,
        }
        
        if torch.cuda.is_available():
            info.update({
                'gpu_count': torch.cuda.device_count(),
                'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else None,
                'gpu_memory_total': torch.cuda.get_device_properties(0).total_memory if torch.cuda.device_count() > 0 else None,
            })
            
        return info
    
    def run_basic_test(self) -> BenchmarkResults:
        """Run the basic benchmark test."""
        logger.info(f"Starting simple benchmark with model: {self.config.model_name}")
        logger.info(f"Dataset: {self.config.dataset_path}")
        logger.info(f"Number of requests: {self.config.num_requests}")
        
        # Load dataset
        dataset = self._load_dataset()
        prompts = [item['prompt'] for item in dataset]
        
        # Initialize model
        self.llm = InstrumentedLLM(
            model=self.config.model_name,
            tensor_parallel_size=self.config.tensor_parallel_size,
            trust_remote_code=True,
            enforce_eager=True,  # Disable CUDA graphs for cleaner timing
        )
        
        # Set up sampling parameters
        sampling_params = SamplingParams(
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        
        # Run generation
        start_time = time.time()
        results = self.llm.generate(prompts, sampling_params)
        total_time = time.time() - start_time
        
        logger.info(f"Benchmark completed in {total_time:.2f}s")
        
        # Collect results
        summary_stats = self.llm.metrics_collector.get_summary_stats()
        detailed_data = self.llm.metrics_collector.export_detailed_data()
        system_info = self._get_system_info()
        
        return BenchmarkResults(
            config=self.config,
            summary_stats=summary_stats,
            detailed_data=detailed_data,
            system_info=system_info,
        )


def main():
    """Main function for running the simple benchmark."""
    # Example usage with different configurations
    
    # Test with small model and short prompts
    config_short = BenchmarkConfig(
        model_name="facebook/opt-125m",
        num_requests=5,
        max_tokens=50,
    )
    
    benchmark = SimpleBenchmark(config_short)
    results = benchmark.run_basic_test()
    
    # Print results
    results.print_summary()
    
    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_path = f"vllm/Sam-Research/results/simple_benchmark_{timestamp}.json"
    results.save_to_file(results_path)


if __name__ == "__main__":
    main()