# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Comprehensive benchmark for multi-factor vLLM performance analysis."""

import json
import time
import os
import asyncio
import concurrent.futures
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

import torch
from vllm import LLM, SamplingParams

from .simple_llm_benchmark import InstrumentedLLM, BenchmarkConfig
from .enhanced_metrics import MetricsCollector


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ComprehensiveConfig:
    """Configuration for comprehensive multi-factor analysis."""
    
    # Model configurations
    models: List[str] = field(default_factory=lambda: [
        "facebook/opt-125m",
        "facebook/opt-350m",
    ])
    
    # Dataset configurations
    datasets: List[str] = field(default_factory=lambda: [
        "short_prompts.json",
        "long_prompts.json",
    ])
    
    # Load testing parameters
    concurrent_requests: List[int] = field(default_factory=lambda: [1, 2, 4])
    arrival_rates: List[float] = field(default_factory=lambda: [0.5, 1.0, 2.0])  # requests per second
    
    # System configurations
    tensor_parallel_sizes: List[int] = field(default_factory=lambda: [1])
    gpu_counts: List[int] = field(default_factory=lambda: [1])
    
    # Generation parameters
    max_tokens: int = 100
    temperature: float = 0.0
    
    # Test parameters
    requests_per_test: int = 10
    warmup_requests: int = 3


@dataclass
class TestConfiguration:
    """Single test configuration."""
    model: str
    dataset: str
    concurrent_requests: int
    arrival_rate: float
    tensor_parallel_size: int
    gpu_count: int
    test_id: str = ""
    
    def __post_init__(self):
        if not self.test_id:
            self.test_id = f"{Path(self.model).name}_{Path(self.dataset).stem}_c{self.concurrent_requests}_r{self.arrival_rate}_tp{self.tensor_parallel_size}_gpu{self.gpu_count}"


@dataclass
class TestResults:
    """Results from a single test configuration."""
    config: TestConfiguration
    summary_stats: Dict[str, Any]
    detailed_data: List[Dict[str, Any]]
    system_metrics: Dict[str, Any]
    test_duration: float
    success_rate: float


class LoadTester:
    """Handle different load testing scenarios."""
    
    def __init__(self, llm: InstrumentedLLM, sampling_params: SamplingParams):
        self.llm = llm
        self.sampling_params = sampling_params
    
    async def run_concurrent_requests(self, prompts: List[str], concurrent_count: int) -> List[Dict[str, Any]]:
        """Run requests with specified concurrency level."""
        if concurrent_count == 1:
            # Sequential execution
            return self.llm.generate(prompts, self.sampling_params)
        
        # For now, simulate concurrent execution by batching
        # In a full implementation, this would use async vLLM or multiple processes
        logger.info(f"Running {len(prompts)} requests with concurrency {concurrent_count}")
        
        batches = [prompts[i:i + concurrent_count] for i in range(0, len(prompts), concurrent_count)]
        all_results = []
        
        for batch in batches:
            batch_results = self.llm.generate(batch, self.sampling_params)
            all_results.extend(batch_results)
            
        return all_results
    
    async def run_with_arrival_rate(self, prompts: List[str], arrival_rate: float) -> List[Dict[str, Any]]:
        """Run requests with controlled arrival rate."""
        if arrival_rate <= 0:
            return self.llm.generate(prompts, self.sampling_params)
        
        results = []
        interval = 1.0 / arrival_rate
        
        logger.info(f"Running {len(prompts)} requests at {arrival_rate} req/s (interval: {interval:.3f}s)")
        
        for i, prompt in enumerate(prompts):
            if i > 0:
                # Wait for the specified interval
                await asyncio.sleep(interval)
            
            # Generate single request
            result = self.llm.generate([prompt], self.sampling_params)
            results.extend(result)
            
        return results


class ComprehensiveBenchmark:
    """Comprehensive multi-factor benchmark system."""
    
    def __init__(self, config: Optional[ComprehensiveConfig] = None):
        self.config = config or ComprehensiveConfig()
        self.results: Dict[str, TestResults] = {}
        
    def _load_dataset(self, dataset_name: str) -> List[Dict[str, Any]]:
        """Load specified dataset."""
        current_dir = Path(__file__).parent
        dataset_path = current_dir / "datasets" / dataset_name
        
        with open(dataset_path, 'r') as f:
            return json.load(f)
    
    def _generate_test_configurations(self) -> List[TestConfiguration]:
        """Generate all test configuration combinations."""
        configs = []
        
        for model in self.config.models:
            for dataset in self.config.datasets:
                for concurrent in self.config.concurrent_requests:
                    for arrival_rate in self.config.arrival_rates:
                        for tp_size in self.config.tensor_parallel_sizes:
                            for gpu_count in self.config.gpu_counts:
                                config = TestConfiguration(
                                    model=model,
                                    dataset=dataset,
                                    concurrent_requests=concurrent,
                                    arrival_rate=arrival_rate,
                                    tensor_parallel_size=tp_size,
                                    gpu_count=gpu_count,
                                )
                                configs.append(config)
        
        logger.info(f"Generated {len(configs)} test configurations")
        return configs
    
    async def _run_single_test(self, test_config: TestConfiguration) -> TestResults:
        """Run a single test configuration."""
        logger.info(f"Running test: {test_config.test_id}")
        
        # Load dataset
        dataset = self._load_dataset(test_config.dataset)
        prompts = [item['prompt'] for item in dataset[:self.config.requests_per_test]]
        
        # Initialize model for this test
        llm = InstrumentedLLM(
            model=test_config.model,
            tensor_parallel_size=test_config.tensor_parallel_size,
            trust_remote_code=True,
            enforce_eager=True,
        )
        
        sampling_params = SamplingParams(
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        
        # Run warmup
        if self.config.warmup_requests > 0:
            warmup_prompts = prompts[:self.config.warmup_requests]
            logger.info(f"Running {len(warmup_prompts)} warmup requests")
            _ = llm.generate(warmup_prompts, sampling_params)
            
            # Reset metrics after warmup
            llm.metrics_collector = MetricsCollector()
        
        # Set up load tester
        load_tester = LoadTester(llm, sampling_params)
        
        # Run the actual test
        start_time = time.time()
        
        try:
            if test_config.concurrent_requests > 1:
                results = await load_tester.run_concurrent_requests(prompts, test_config.concurrent_requests)
            else:
                results = await load_tester.run_with_arrival_rate(prompts, test_config.arrival_rate)
            
            test_duration = time.time() - start_time
            success_rate = 1.0  # All requests succeeded
            
        except Exception as e:
            logger.error(f"Test failed: {test_config.test_id}, Error: {e}")
            test_duration = time.time() - start_time
            success_rate = 0.0
            results = []
        
        # Collect metrics
        summary_stats = llm.metrics_collector.get_summary_stats()
        detailed_data = llm.metrics_collector.export_detailed_data()
        
        # System metrics
        system_metrics = {
            'gpu_available': torch.cuda.is_available(),
            'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'tensor_parallel_size': test_config.tensor_parallel_size,
            'test_duration': test_duration,
        }
        
        return TestResults(
            config=test_config,
            summary_stats=summary_stats,
            detailed_data=detailed_data,
            system_metrics=system_metrics,
            test_duration=test_duration,
            success_rate=success_rate,
        )
    
    async def run_full_analysis(self) -> Dict[str, TestResults]:
        """Run the complete comprehensive analysis."""
        logger.info("Starting comprehensive benchmark analysis")
        
        test_configs = self._generate_test_configurations()
        
        # Run all tests
        for i, config in enumerate(test_configs):
            logger.info(f"Progress: {i+1}/{len(test_configs)}")
            
            try:
                result = await self._run_single_test(config)
                self.results[config.test_id] = result
                logger.info(f"Completed test {config.test_id} - Success rate: {result.success_rate:.1%}")
                
            except Exception as e:
                logger.error(f"Failed to run test {config.test_id}: {e}")
                continue
            
            # Add a brief pause between tests to avoid resource contention
            await asyncio.sleep(1.0)
        
        logger.info(f"Comprehensive analysis completed. {len(self.results)} tests successful.")
        return self.results
    
    def generate_figures(self, output_dir: str = "vllm/Sam-Research/results/figures"):
        """Generate comprehensive performance analysis figures."""
        os.makedirs(output_dir, exist_ok=True)
        
        if not self.results:
            logger.warning("No results available for figure generation")
            return
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Figure 1: Latency breakdown by model and dataset
        self._plot_latency_breakdown(output_dir)
        
        # Figure 2: Throughput comparison
        self._plot_throughput_comparison(output_dir)
        
        # Figure 3: Concurrency scaling
        self._plot_concurrency_scaling(output_dir)
        
        # Figure 4: Phase timing analysis
        self._plot_phase_analysis(output_dir)
        
        # Figure 5: Resource utilization heatmap
        self._plot_resource_utilization(output_dir)
        
        logger.info(f"Figures generated in {output_dir}")
    
    def _plot_latency_breakdown(self, output_dir: str):
        """Plot end-to-end latency breakdown by phase."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Latency Breakdown Analysis', fontsize=16)
        
        # Collect data
        data = []
        for result in self.results.values():
            if result.success_rate < 1.0:
                continue
                
            config = result.config
            stats = result.summary_stats
            
            e2e_latency = stats.get('end_to_end_latency', {})
            prefill_latency = stats.get('prefill_duration', {})
            decode_latency = stats.get('decode_duration', {})
            
            data.append({
                'model': Path(config.model).name,
                'dataset': Path(config.dataset).stem,
                'e2e_mean': e2e_latency.get('mean', 0),
                'prefill_mean': prefill_latency.get('mean', 0),
                'decode_mean': decode_latency.get('mean', 0),
                'concurrent': config.concurrent_requests,
            })
        
        if not data:
            logger.warning("No data available for latency breakdown plot")
            return
        
        # Plot 1: E2E latency by model
        models = list(set(d['model'] for d in data))
        datasets = list(set(d['dataset'] for d in data))
        
        e2e_by_model = defaultdict(list)
        for d in data:
            e2e_by_model[d['model']].append(d['e2e_mean'])
        
        axes[0, 0].bar(e2e_by_model.keys(), [np.mean(v) for v in e2e_by_model.values()])
        axes[0, 0].set_title('Mean End-to-End Latency by Model')
        axes[0, 0].set_ylabel('Latency (seconds)')
        
        # Plot 2: Phase breakdown
        prefill_means = [d['prefill_mean'] for d in data]
        decode_means = [d['decode_mean'] for d in data]
        
        axes[0, 1].hist([prefill_means, decode_means], bins=20, alpha=0.7, 
                       label=['Prefill', 'Decode'])
        axes[0, 1].set_title('Phase Duration Distribution')
        axes[0, 1].set_xlabel('Duration (seconds)')
        axes[0, 1].legend()
        
        # Plot 3: Dataset comparison
        e2e_by_dataset = defaultdict(list)
        for d in data:
            e2e_by_dataset[d['dataset']].append(d['e2e_mean'])
        
        axes[1, 0].bar(e2e_by_dataset.keys(), [np.mean(v) for v in e2e_by_dataset.values()])
        axes[1, 0].set_title('Mean End-to-End Latency by Dataset')
        axes[1, 0].set_ylabel('Latency (seconds)')
        
        # Plot 4: Concurrency impact
        concurrent_data = defaultdict(list)
        for d in data:
            concurrent_data[d['concurrent']].append(d['e2e_mean'])
        
        concurrent_levels = sorted(concurrent_data.keys())
        concurrent_means = [np.mean(concurrent_data[c]) for c in concurrent_levels]
        
        axes[1, 1].plot(concurrent_levels, concurrent_means, 'o-')
        axes[1, 1].set_title('Latency vs Concurrency')
        axes[1, 1].set_xlabel('Concurrent Requests')
        axes[1, 1].set_ylabel('Mean Latency (seconds)')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/latency_breakdown.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_throughput_comparison(self, output_dir: str):
        """Plot throughput comparison across configurations."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Throughput Analysis', fontsize=16)
        
        # Collect throughput data
        data = []
        for result in self.results.values():
            if result.success_rate < 1.0:
                continue
                
            config = result.config
            stats = result.summary_stats
            
            prefill_tps = stats.get('prefill_throughput_tokens_per_sec', {})
            decode_tps = stats.get('decode_throughput_tokens_per_sec', {})
            
            data.append({
                'model': Path(config.model).name,
                'dataset': Path(config.dataset).stem,
                'prefill_tps': prefill_tps.get('mean', 0),
                'decode_tps': decode_tps.get('mean', 0),
                'config_id': config.test_id,
            })
        
        if not data:
            return
        
        # Plot prefill throughput
        models = list(set(d['model'] for d in data))
        prefill_by_model = defaultdict(list)
        for d in data:
            prefill_by_model[d['model']].append(d['prefill_tps'])
        
        axes[0].bar(prefill_by_model.keys(), [np.mean(v) for v in prefill_by_model.values()])
        axes[0].set_title('Prefill Throughput by Model')
        axes[0].set_ylabel('Tokens/sec')
        
        # Plot decode throughput
        decode_by_model = defaultdict(list)
        for d in data:
            decode_by_model[d['model']].append(d['decode_tps'])
        
        axes[1].bar(decode_by_model.keys(), [np.mean(v) for v in decode_by_model.values()])
        axes[1].set_title('Decode Throughput by Model')
        axes[1].set_ylabel('Tokens/sec')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/throughput_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_concurrency_scaling(self, output_dir: str):
        """Plot performance scaling with concurrency."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Concurrency Scaling Analysis', fontsize=16)
        
        # Collect scaling data
        scaling_data = defaultdict(lambda: defaultdict(list))
        
        for result in self.results.values():
            if result.success_rate < 1.0:
                continue
                
            config = result.config
            stats = result.summary_stats
            
            model_name = Path(config.model).name
            concurrent = config.concurrent_requests
            
            e2e_latency = stats.get('end_to_end_latency', {}).get('mean', 0)
            requests_per_sec = stats.get('requests_per_second', 0)
            
            scaling_data[model_name]['concurrency'].append(concurrent)
            scaling_data[model_name]['latency'].append(e2e_latency)
            scaling_data[model_name]['throughput'].append(requests_per_sec)
        
        # Plot scaling curves for each model
        for i, (model, data) in enumerate(scaling_data.items()):
            if not data['concurrency']:
                continue
                
            # Sort by concurrency level
            sorted_data = sorted(zip(data['concurrency'], data['latency'], data['throughput']))
            concurrency, latency, throughput = zip(*sorted_data)
            
            # Latency scaling
            if i < 2:
                axes[0, i].plot(concurrency, latency, 'o-')
                axes[0, i].set_title(f'{model} - Latency vs Concurrency')
                axes[0, i].set_xlabel('Concurrent Requests')
                axes[0, i].set_ylabel('Mean Latency (seconds)')
                
                # Throughput scaling
                axes[1, i].plot(concurrency, throughput, 'o-', color='orange')
                axes[1, i].set_title(f'{model} - Throughput vs Concurrency')
                axes[1, i].set_xlabel('Concurrent Requests')
                axes[1, i].set_ylabel('Requests/sec')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/concurrency_scaling.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_phase_analysis(self, output_dir: str):
        """Plot detailed phase timing analysis."""
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Collect phase data
        phase_data = []
        for result in self.results.values():
            if result.success_rate < 1.0 or not result.detailed_data:
                continue
                
            for record in result.detailed_data:
                phase_data.append({
                    'model': Path(result.config.model).name,
                    'dataset': Path(result.config.dataset).stem,
                    'prefill_duration': record.get('prefill_duration', 0),
                    'decode_duration': record.get('decode_duration', 0),
                    'total_duration': record.get('end_to_end_latency', 0),
                })
        
        if not phase_data:
            return
        
        # Create scatter plot showing phase relationship
        prefill_times = [d['prefill_duration'] for d in phase_data if d['prefill_duration']]
        decode_times = [d['decode_duration'] for d in phase_data if d['decode_duration']]
        
        ax.scatter(prefill_times, decode_times, alpha=0.6)
        ax.set_xlabel('Prefill Duration (seconds)')
        ax.set_ylabel('Decode Duration (seconds)')
        ax.set_title('Prefill vs Decode Phase Timing')
        
        # Add diagonal line for reference
        if prefill_times and decode_times:
            max_time = max(max(prefill_times), max(decode_times))
            ax.plot([0, max_time], [0, max_time], 'r--', alpha=0.5, label='Equal Time Line')
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/phase_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_resource_utilization(self, output_dir: str):
        """Plot resource utilization patterns."""
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Collect resource data
        resource_data = []
        for result in self.results.values():
            if result.success_rate < 1.0:
                continue
                
            config = result.config
            model_name = Path(config.model).name
            dataset_name = Path(config.dataset).stem
            
            # Extract resource metrics (these would come from detailed monitoring)
            resource_data.append({
                'model': model_name,
                'dataset': dataset_name,
                'concurrent': config.concurrent_requests,
                'test_duration': result.test_duration,
                'success_rate': result.success_rate,
            })
        
        if not resource_data:
            return
        
        # Create heatmap of test duration by model and dataset
        models = list(set(d['model'] for d in resource_data))
        datasets = list(set(d['dataset'] for d in resource_data))
        
        heatmap_data = np.zeros((len(models), len(datasets)))
        
        for i, model in enumerate(models):
            for j, dataset in enumerate(datasets):
                durations = [d['test_duration'] for d in resource_data 
                           if d['model'] == model and d['dataset'] == dataset]
                if durations:
                    heatmap_data[i, j] = np.mean(durations)
        
        sns.heatmap(heatmap_data, xticklabels=datasets, yticklabels=models, 
                   annot=True, fmt='.2f', ax=ax, cmap='YlOrRd')
        ax.set_title('Mean Test Duration (seconds)')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/resource_utilization.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def export_results(self, filepath: str):
        """Export all results to JSON file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        export_data = {
            'config': asdict(self.config),
            'results': {test_id: asdict(result) for test_id, result in self.results.items()},
            'summary': {
                'total_tests': len(self.results),
                'successful_tests': sum(1 for r in self.results.values() if r.success_rate >= 1.0),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"Results exported to {filepath}")


async def main():
    """Main function for running comprehensive benchmark."""
    # Create benchmark with default configuration
    benchmark = ComprehensiveBenchmark()
    
    # Run full analysis
    results = await benchmark.run_full_analysis()
    
    # Generate figures
    benchmark.generate_figures()
    
    # Export results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_path = f"vllm/Sam-Research/results/comprehensive_benchmark_{timestamp}.json"
    benchmark.export_results(results_path)
    
    # Print summary
    print(f"\nComprehensive Benchmark Complete!")
    print(f"Total tests run: {len(results)}")
    print(f"Successful tests: {sum(1 for r in results.values() if r.success_rate >= 1.0)}")
    print(f"Results saved to: {results_path}")
    print(f"Figures generated in: vllm/Sam-Research/results/figures/")


if __name__ == "__main__":
    asyncio.run(main())