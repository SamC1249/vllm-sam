# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## About vLLM

vLLM is a high-throughput and memory-efficient inference and serving engine for LLMs. It's designed for fast LLM serving with features like PagedAttention, continuous batching, and support for various quantization methods.

## Development Commands

### Installation & Setup
- `pip install -e .` - Install vLLM in development mode
- `pip install -r requirements/dev.txt` - Install development dependencies 
- `pip install -r requirements/test.txt` - Install test dependencies

### Testing
- `python -m pytest tests/` - Run all tests
- `python -m pytest tests/test_file.py` - Run specific test file
- `python -m pytest tests/ -k test_name` - Run tests matching pattern
- `python -m pytest tests/ --forked` - Run tests in separate processes (recommended for GPU tests)
- `python -m pytest tests/ -x` - Stop on first failure

### Linting & Code Quality  
- `pre-commit install` - Install pre-commit hooks (run once)
- `pre-commit run --all-files` - Run all linters manually
- `./tools/mypy.sh` - Run mypy type checking
- Note: The old `format.sh` has been replaced by pre-commit hooks

### Build Commands
- `python setup.py build_ext --inplace` - Build C++/CUDA extensions in-place
- `CMAKE_BUILD_TYPE=Debug python setup.py build_ext --inplace` - Debug build
- `VLLM_TARGET_DEVICE=cpu python setup.py build_ext --inplace` - CPU-only build

### Environment Variables
- `VLLM_TARGET_DEVICE` - Target device (cuda, cpu, tpu, neuron, xpu)
- `MAX_JOBS` - Number of parallel compilation jobs
- `NVCC_THREADS` - Number of NVCC threads for CUDA compilation
- `CMAKE_BUILD_TYPE` - Build type (Release, Debug, RelWithDebInfo)
- `VLLM_USE_PRECOMPILED` - Use precompiled binaries

## Architecture Overview

### Core Components

**Engine & Execution**
- `vllm/engine/` - LLM engine implementations (sync/async)
- `vllm/executor/` - Multi-process and distributed execution
- `vllm/worker/` - Worker processes for model execution
- `vllm/v1/` - Next-generation vLLM architecture (alpha)

**Model Support**
- `vllm/model_executor/models/` - Model implementations (LLaMA, GPT, etc.)
- `vllm/model_executor/layers/` - Custom layers (attention, MoE, quantization)
- `vllm/transformers_utils/` - HuggingFace integration utilities

**Memory & Attention**
- `vllm/attention/` - Attention implementations (PagedAttention, FlashAttention)
- `vllm/core/` - Core scheduling and block management
- `vllm/core/block_manager.py` - Memory block management

**Serving & APIs**
- `vllm/entrypoints/` - API servers and CLI interfaces
- `vllm/entrypoints/openai/` - OpenAI-compatible API server
- `vllm/engine/async_llm_engine.py` - Async serving engine

**Extensions & Plugins**
- `vllm/lora/` - LoRA adapter support
- `vllm/multimodal/` - Multi-modal input processing
- `vllm/compilation/` - torch.compile integration
- `vllm/distributed/` - Multi-GPU/multi-node support

### Key Files to Understand

- `vllm/engine/llm_engine.py` - Main engine class
- `vllm/core/scheduler.py` - Request scheduling logic
- `vllm/attention/selector.py` - Attention backend selection
- `vllm/model_executor/model_loader/weight_utils.py` - Model loading utilities
- `vllm/worker/worker.py` - Core worker implementation

### Build System

vLLM uses a hybrid build system:
- **Python packaging**: `pyproject.toml` and `setup.py`
- **C++/CUDA compilation**: Custom CMake integration via `cmake_build_ext`
- **Extensions**: Built as Python C extensions with CMake backend

Key build files:
- `CMakeLists.txt` - Main CMake configuration
- `setup.py` - Python build script with CMake integration
- `cmake/` - CMake utilities and external dependencies

### Testing Strategy

- **Unit tests**: `tests/` - Component-level testing
- **Integration tests**: End-to-end model serving tests
- **Distributed tests**: Multi-GPU testing with `pytest-forked`
- **Model tests**: Correctness tests for specific model implementations
- **Benchmark tests**: Performance regression testing

Use `pytest-forked` for GPU tests to avoid CUDA context issues between tests.

### Multi-Modal Support

vLLM supports vision and audio models through:
- `vllm/multimodal/` - Input preprocessing and registry
- `vllm/assets/` - Asset handling (images, audio, video)
- Model-specific implementations in `vllm/model_executor/models/`

### Quantization Support

Extensive quantization support through:
- `vllm/model_executor/layers/quantization/` - Quantization implementations
- Supports GPTQ, AWQ, FP8, INT8, and more
- Backend-specific optimizations (CUTLASS, Marlin, etc.)