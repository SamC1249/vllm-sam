#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Utility script to run Sam's research benchmarks."""

import argparse
import asyncio
import sys
from pathlib import Path

# Add the vLLM directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from simple_llm_benchmark import SimpleBenchmark, BenchmarkConfig
from comprehensive_benchmark import ComprehensiveBenchmark, ComprehensiveConfig


def setup_environment():
    """Setup environment variables for benchmark execution."""
    import torch
    import os
    
    if torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
        os.environ['VLLM_DEVICE'] = 'cuda'
    else:
        os.environ['VLLM_DEVICE'] = 'cpu'
        
    # Disable Ray if causing issues
    os.environ['VLLM_DISABLE_RAY'] = '1'
    os.environ['RAY_USAGE_STATS_ENABLED'] = '0'


def run_simple_benchmark(args):
    """Run the simple benchmark."""
    print("Starting Simple LLM Benchmark...")
    
    config = BenchmarkConfig(
        model_name=args.model,
        num_requests=args.requests,
        max_tokens=args.max_tokens,
        tensor_parallel_size=args.tensor_parallel,
    )
    
    if args.dataset:
        config.dataset_path = args.dataset
    
    benchmark = SimpleBenchmark(config)
    results = benchmark.run_basic_test()
    
    # Print results
    results.print_summary()
    
    # Save results if requested
    if args.output:
        results.save_to_file(args.output)
    
    return results


async def run_comprehensive_benchmark(args):
    """Run the comprehensive benchmark."""
    print("Starting Comprehensive Benchmark Analysis...")
    
    config = ComprehensiveConfig()
    
    if args.models:
        config.models = args.models.split(',')
    
    if args.requests:
        config.requests_per_test = args.requests
    
    if args.tensor_parallel:
        config.tensor_parallel_sizes = [args.tensor_parallel]
    
    # Override concurrent requests if specified
    if hasattr(args, 'concurrent') and args.concurrent:
        config.concurrent_requests = [int(x) for x in args.concurrent.split(',')]
    
    benchmark = ComprehensiveBenchmark(config)
    
    # Run analysis
    results = await benchmark.run_full_analysis()
    
    # Generate figures
    benchmark.generate_figures()
    
    # Export results
    if args.output:
        benchmark.export_results(args.output)
    
    print(f"\nBenchmark completed with {len(results)} tests")
    return results


def main():
    parser = argparse.ArgumentParser(description="Run vLLM performance benchmarks")
    subparsers = parser.add_subparsers(dest='benchmark_type', help='Benchmark type')
    
    # Simple benchmark
    simple_parser = subparsers.add_parser('simple', help='Run simple benchmark')
    simple_parser.add_argument('--model', default='facebook/opt-125m', 
                              help='Model name to test')
    simple_parser.add_argument('--dataset', help='Path to dataset JSON file')
    simple_parser.add_argument('--requests', type=int, default=10,
                              help='Number of requests to run')
    simple_parser.add_argument('--max-tokens', type=int, default=100,
                              help='Maximum tokens to generate')
    simple_parser.add_argument('--tensor-parallel', type=int, default=1,
                              help='Tensor parallel size')
    simple_parser.add_argument('--output', help='Output file for results')
    
    # Comprehensive benchmark
    comp_parser = subparsers.add_parser('comprehensive', help='Run comprehensive benchmark')
    comp_parser.add_argument('--models', help='Comma-separated list of models')
    comp_parser.add_argument('--requests', type=int, default=10,
                            help='Number of requests per test')
    comp_parser.add_argument('--tensor-parallel', type=int, default=1,
                            help='Tensor parallel size')
    comp_parser.add_argument('--concurrent', help='Comma-separated concurrent levels')
    comp_parser.add_argument('--output', help='Output file for results')
    
    # Both benchmarks
    both_parser = subparsers.add_parser('both', help='Run both benchmarks')
    both_parser.add_argument('--model', default='facebook/opt-125m',
                            help='Model for simple benchmark')
    both_parser.add_argument('--models', help='Models for comprehensive benchmark')
    both_parser.add_argument('--requests', type=int, default=5,
                            help='Number of requests')
    both_parser.add_argument('--tensor-parallel', type=int, default=1,
                            help='Tensor parallel size')
    
    args = parser.parse_args()
    
    if not args.benchmark_type:
        parser.print_help()
        return
    
    # Setup environment before running any benchmarks
    setup_environment()
    
    try:
        if args.benchmark_type == 'simple':
            run_simple_benchmark(args)
            
        elif args.benchmark_type == 'comprehensive':
            asyncio.run(run_comprehensive_benchmark(args))
            
        elif args.benchmark_type == 'both':
            print("Running Simple Benchmark first...")
            simple_args = argparse.Namespace(
                model=args.model,
                dataset=None,
                requests=args.requests,
                max_tokens=100,
                tensor_parallel=args.tensor_parallel,
                output=None
            )
            run_simple_benchmark(simple_args)
            
            print("\nNow running Comprehensive Benchmark...")
            comp_args = argparse.Namespace(
                models=args.models,
                requests=args.requests,
                tensor_parallel=args.tensor_parallel,
                concurrent=None,
                output=None
            )
            asyncio.run(run_comprehensive_benchmark(comp_args))
    
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
    except Exception as e:
        print(f"Error running benchmark: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()