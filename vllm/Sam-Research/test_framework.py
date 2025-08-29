#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Lightweight test to verify the research framework works on RunPod."""

import os
import sys
import time
import traceback
from pathlib import Path

# Minimal imports to test basic functionality
try:
    import torch
    print(f"✅ PyTorch available: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.version.cuda}")
        print(f"✅ GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"   - GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("⚠️  CUDA not available - will test CPU mode")
except ImportError as e:
    print(f"❌ PyTorch import failed: {e}")
    sys.exit(1)

try:
    from vllm import LLM, SamplingParams
    print("✅ vLLM imported successfully")
except ImportError as e:
    print(f"❌ vLLM import failed: {e}")
    print("Make sure vLLM is installed: pip install vllm")
    sys.exit(1)

# Test our framework imports
try:
    # Add current directory to path for imports
    current_dir = Path(__file__).parent
    sys.path.insert(0, str(current_dir.parent))
    
    from enhanced_metrics import MetricsCollector, EnhancedRequestMetrics
    print("✅ Enhanced metrics imported successfully")
    
    # Don't import the full benchmarks yet - just test basic functionality
    print("✅ Framework imports working")
except ImportError as e:
    print(f"❌ Framework import failed: {e}")
    traceback.print_exc()
    sys.exit(1)


def test_basic_vllm():
    """Test basic vLLM functionality with smallest possible model."""
    print("\n🧪 Testing basic vLLM functionality...")
    
    try:
        # Use the smallest model possible for testing
        model_name = "facebook/opt-125m"
        print(f"Loading model: {model_name}")
        
        # Initialize with minimal resources
        llm = LLM(
            model=model_name,
            tensor_parallel_size=1,
            trust_remote_code=True,
            enforce_eager=True,  # Disable CUDA graphs for simpler testing
            max_model_len=512,   # Reduce memory usage
        )
        
        print("✅ Model loaded successfully")
        
        # Test simple generation
        prompts = ["Hello, my name is"]
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=10,  # Very short generation for quick test
        )
        
        print("Generating response...")
        start_time = time.time()
        outputs = llm.generate(prompts, sampling_params)
        generation_time = time.time() - start_time
        
        # Check output
        output = outputs[0]
        generated_text = output.outputs[0].text
        
        print(f"✅ Generation successful in {generation_time:.2f}s")
        print(f"   Input: '{prompts[0]}'")
        print(f"   Output: '{generated_text}'")
        
        return True
        
    except Exception as e:
        print(f"❌ vLLM test failed: {e}")
        traceback.print_exc()
        return False


def test_metrics_collection():
    """Test our enhanced metrics collection."""
    print("\n🧪 Testing metrics collection...")
    
    try:
        # Test metrics collector
        collector = MetricsCollector()
        
        # Test request tracking
        request_id = "test_request"
        metrics = collector.start_request_tracking(request_id)
        
        print("✅ Request tracking started")
        
        # Simulate some processing time
        time.sleep(0.1)
        
        # Complete the request
        final_metrics = collector.complete_request(request_id, output_tokens=5)
        
        print("✅ Request tracking completed")
        
        # Get summary stats
        stats = collector.get_summary_stats()
        print(f"✅ Summary stats generated: {len(stats)} metrics")
        
        # Export detailed data
        detailed = collector.export_detailed_data()
        print(f"✅ Detailed data exported: {len(detailed)} records")
        
        return True
        
    except Exception as e:
        print(f"❌ Metrics test failed: {e}")
        traceback.print_exc()
        return False


def test_dataset_loading():
    """Test loading our test datasets."""
    print("\n🧪 Testing dataset loading...")
    
    try:
        import json
        
        # Test short prompts dataset
        short_path = current_dir / "datasets" / "short_prompts.json"
        if short_path.exists():
            with open(short_path, 'r') as f:
                short_data = json.load(f)
            print(f"✅ Short prompts loaded: {len(short_data)} items")
        else:
            print(f"❌ Short prompts file not found: {short_path}")
            return False
        
        # Test long prompts dataset  
        long_path = current_dir / "datasets" / "long_prompts.json"
        if long_path.exists():
            with open(long_path, 'r') as f:
                long_data = json.load(f)
            print(f"✅ Long prompts loaded: {len(long_data)} items")
        else:
            print(f"❌ Long prompts file not found: {long_path}")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        traceback.print_exc()
        return False


def test_integrated_workflow():
    """Test a minimal integrated workflow."""
    print("\n🧪 Testing integrated workflow...")
    
    try:
        # Initialize metrics collector
        collector = MetricsCollector()
        
        # Load tiny model
        model_name = "facebook/opt-125m"
        llm = LLM(
            model=model_name,
            tensor_parallel_size=1,
            trust_remote_code=True,
            enforce_eager=True,
            max_model_len=256,
        )
        
        # Simple prompt
        prompts = ["The quick brown fox"]
        sampling_params = SamplingParams(temperature=0.0, max_tokens=5)
        
        # Track request
        request_id = "integrated_test"
        metrics = collector.start_request_tracking(request_id)
        
        # Generate
        start_time = time.time()
        outputs = llm.generate(prompts, sampling_params)
        end_time = time.time()
        
        # Complete tracking
        output_tokens = len(outputs[0].outputs[0].token_ids)
        final_metrics = collector.complete_request(request_id, output_tokens)
        
        # Verify results
        duration = end_time - start_time
        generated_text = outputs[0].outputs[0].text
        
        print(f"✅ Integrated test successful:")
        print(f"   Duration: {duration:.3f}s")
        print(f"   Output tokens: {output_tokens}")
        print(f"   Generated: '{generated_text}'")
        print(f"   Metrics collected: {len(collector.completed_requests)} requests")
        
        return True
        
    except Exception as e:
        print(f"❌ Integrated test failed: {e}")
        traceback.print_exc()
        return False


def check_system_resources():
    """Check available system resources."""
    print("\n💻 System Resource Check:")
    
    # CPU info
    cpu_count = os.cpu_count()
    print(f"CPU cores: {cpu_count}")
    
    # Memory info (if psutil available)
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"RAM: {memory.total / (1024**3):.1f} GB total, {memory.available / (1024**3):.1f} GB available")
    except ImportError:
        print("RAM: psutil not available for memory check")
    
    # GPU info
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / (1024**3)
            print(f"GPU {i}: {props.name} - {memory_gb:.1f} GB")
    else:
        print("GPU: No CUDA devices available")
    
    # Disk space
    try:
        import shutil
        disk_usage = shutil.disk_usage(".")
        free_gb = disk_usage.free / (1024**3)
        print(f"Disk space: {free_gb:.1f} GB free")
    except:
        print("Disk space: Unable to check")


def main():
    """Main test function."""
    print("🚀 vLLM Research Framework Lightweight Test")
    print("=" * 50)
    
    # System check
    check_system_resources()
    
    # Run tests
    tests = [
        ("Dataset Loading", test_dataset_loading),
        ("Metrics Collection", test_metrics_collection), 
        ("Basic vLLM", test_basic_vllm),
        ("Integrated Workflow", test_integrated_workflow),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            success = test_func()
            results.append((test_name, success))
            if success:
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} CRASHED: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*50)
    print("🎯 TEST SUMMARY")
    print("="*50)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:20} : {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Framework is ready for RunPod deployment.")
        print("\nNext steps:")
        print("1. Deploy this code to RunPod")
        print("2. Run: python vllm/Sam-Research/test_framework.py")
        print("3. If successful, run the full benchmarks")
    else:
        print(f"\n⚠️  {total - passed} tests failed. Check the errors above before proceeding.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)