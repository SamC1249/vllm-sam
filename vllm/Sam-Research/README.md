# Sam's Research Framework for vLLM Performance Analysis

This directory contains research tools for measuring vLLM performance across different phases and configurations.

## Structure

- `enhanced_metrics.py` - Extended metrics collection with phase-specific timing
- `simple_llm_benchmark.py` - Basic latency and throughput measurement 
- `comprehensive_benchmark.py` - Multi-factor performance analysis
- `datasets/` - Test datasets for realistic workloads
- `results/` - Generated performance data and figures

## Key Features

### Phase-Specific Metrics
- Prefill phase timing and resource usage
- Decode phase timing and resource usage  
- End-to-end latency breakdown

### Multi-Factor Analysis
- Dataset variations (different input lengths, content types)
- Request arrival rate simulation
- Multiple model comparisons
- GPU scaling analysis
- Parallelism configuration testing

## Usage

### Simple Benchmark
```python
from vllm.Sam_Research.simple_llm_benchmark import SimpleBenchmark

benchmark = SimpleBenchmark(model="meta-llama/Llama-2-7b-hf")
results = benchmark.run_basic_test()
print(results.summary())
```

### Comprehensive Analysis
```python  
from vllm.Sam_Research.comprehensive_benchmark import ComprehensiveBenchmark

benchmark = ComprehensiveBenchmark()
results = benchmark.run_full_analysis()
benchmark.generate_figures(results)
```