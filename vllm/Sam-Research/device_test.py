#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Focused device configuration test to diagnose the 'Device string must not be empty' error."""

import os
import sys
import time
import traceback
from pathlib import Path

print("Device Configuration Diagnostic Test")
print("=" * 50)

# Step 1: Check environment before any imports
print("1. INITIAL ENVIRONMENT CHECK:")
print(f"   CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'NOT SET')}")
print(f"   VLLM_DEVICE: {os.environ.get('VLLM_DEVICE', 'NOT SET')}")
print(f"   VLLM_DISABLE_RAY: {os.environ.get('VLLM_DISABLE_RAY', 'NOT SET')}")
print(f"   RAY_USAGE_STATS_ENABLED: {os.environ.get('RAY_USAGE_STATS_ENABLED', 'NOT SET')}")

# Step 2: Import torch and check CUDA
print("\n2. TORCH & CUDA CHECK:")
try:
    import torch
    print(f"   OK PyTorch version: {torch.__version__}")
    print(f"   OK CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   OK CUDA version: {torch.version.cuda}")
        print(f"   OK GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"      - GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("   WARNING: CUDA not available - will use CPU mode")
except Exception as e:
    print(f"   ERROR: Torch import failed: {e}")
    sys.exit(1)

# Step 3: Manual device environment setup
print("\n3. MANUAL DEVICE ENVIRONMENT SETUP:")

def setup_device_environment():
    """Setup device environment manually."""
    print("   Setting up environment variables...")
    
    if torch.cuda.is_available():
        # Set CUDA device
        if 'CUDA_VISIBLE_DEVICES' not in os.environ:
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        os.environ['VLLM_DEVICE'] = 'cuda'
        print(f"      CUDA_VISIBLE_DEVICES = {os.environ['CUDA_VISIBLE_DEVICES']}")
        print(f"      VLLM_DEVICE = {os.environ['VLLM_DEVICE']}")
    else:
        os.environ['VLLM_DEVICE'] = 'cpu'
        print(f"      VLLM_DEVICE = {os.environ['VLLM_DEVICE']}")
    
    # Disable Ray to avoid conflicts
    os.environ['VLLM_DISABLE_RAY'] = '1'
    os.environ['RAY_USAGE_STATS_ENABLED'] = '0'
    print(f"      VLLM_DISABLE_RAY = {os.environ['VLLM_DISABLE_RAY']}")
    print(f"      RAY_USAGE_STATS_ENABLED = {os.environ['RAY_USAGE_STATS_ENABLED']}")

setup_device_environment()

# Step 4: Test vLLM import
print("\n4. VLLM IMPORT TEST:")
try:
    from vllm import LLM, SamplingParams
    print("   OK: vLLM imported successfully")
except Exception as e:
    print(f"   ERROR: vLLM import failed: {e}")
    traceback.print_exc()
    print("\n   SYSTEM ISSUE: vLLM is not installed or not accessible.")
    print("   This needs to be fixed before continuing.")
    sys.exit(1)

# Step 5: Test basic vLLM initialization with explicit device configuration
print("\n5. VLLM INITIALIZATION TEST:")

def test_vllm_initialization():
    """Test vLLM initialization with different device configurations."""
    
    model_name = "facebook/opt-125m"
    
    # Test case 1: Default initialization
    print(f"   Test 1: Default LLM initialization with {model_name}")
    try:
        llm = LLM(
            model=model_name,
            tensor_parallel_size=1,
            trust_remote_code=True,
            enforce_eager=True,
            max_model_len=256,
        )
        print("   OK: Default initialization succeeded")
        del llm  # Clean up
        return True
    except Exception as e:
        print(f"   ERROR: Default initialization failed: {e}")
        print("   Full error:")
        traceback.print_exc()
    
    # Test case 2: Explicit device specification
    print(f"   Test 2: Explicit device specification")
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        llm = LLM(
            model=model_name,
            device=device,
            tensor_parallel_size=1,
            trust_remote_code=True,
            enforce_eager=True,
            max_model_len=256,
        )
        print(f"   OK: Explicit device={device} initialization succeeded")
        del llm  # Clean up
        return True
    except Exception as e:
        print(f"   ERROR: Explicit device initialization failed: {e}")
        print("   Full error:")
        traceback.print_exc()
    
    # Test case 3: Force CPU mode
    print(f"   Test 3: Force CPU mode")
    try:
        # Temporarily override environment for CPU test
        old_device = os.environ.get('VLLM_DEVICE')
        os.environ['VLLM_DEVICE'] = 'cpu'
        
        llm = LLM(
            model=model_name,
            device='cpu',
            tensor_parallel_size=1,
            trust_remote_code=True,
            enforce_eager=True,
            max_model_len=256,
        )
        print("   OK: CPU mode initialization succeeded")
        del llm  # Clean up
        
        # Restore original device setting
        if old_device:
            os.environ['VLLM_DEVICE'] = old_device
        
        return True
    except Exception as e:
        print(f"   ERROR: CPU mode initialization failed: {e}")
        print("   Full error:")
        traceback.print_exc()
    
    return False

success = test_vllm_initialization()

# Step 6: Test our enhanced metrics
print("\n6. ENHANCED METRICS TEST:")
if success:
    try:
        # Add current directory to path for imports
        current_dir = Path(__file__).parent
        if str(current_dir) not in sys.path:
            sys.path.insert(0, str(current_dir))
        
        # Try to import enhanced_metrics
        from enhanced_metrics import MetricsCollector
        collector = MetricsCollector()
        print("   OK: Enhanced metrics imported and initialized")
        
        # Test request tracking
        request_id = "device_test"
        metrics = collector.start_request_tracking(request_id)
        time.sleep(0.01)
        final_metrics = collector.complete_request(request_id, output_tokens=1)
        print("   OK: Basic metrics tracking works")
        
    except Exception as e:
        print(f"   ERROR: Enhanced metrics test failed: {e}")
        traceback.print_exc()
        success = False

# Step 7: Final diagnosis
print("\n" + "=" * 50)
print("7. DIAGNOSTIC SUMMARY:")
print("=" * 50)

if success:
    print("SUCCESS: ALL TESTS PASSED!")
    print("\nThe device configuration appears to be working correctly.")
    print("If you're still getting the 'Device string must not be empty' error,")
    print("it might be occurring in a different part of the code.")
    print("\nNext steps:")
    print("1. Run this test in your environment: python vllm/Sam-Research/device_test.py")
    print("2. If this passes, try the lightweight test: python vllm/Sam-Research/test_framework.py")
    print("3. Compare the environment variables and initialization patterns")
else:
    print("FAILURE: TESTS FAILED!")
    print("\nThe device configuration needs additional fixes.")
    print("Common issues:")
    print("- VLLM_DEVICE environment variable not properly set")
    print("- CUDA device visibility issues")
    print("- Ray initialization conflicts")
    print("- vLLM version compatibility issues")

# Step 8: Environment summary for reference
print(f"\nFINAL ENVIRONMENT STATE:")
print(f"   CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'NOT SET')}")
print(f"   VLLM_DEVICE: {os.environ.get('VLLM_DEVICE', 'NOT SET')}")
print(f"   VLLM_DISABLE_RAY: {os.environ.get('VLLM_DISABLE_RAY', 'NOT SET')}")
print(f"   RAY_USAGE_STATS_ENABLED: {os.environ.get('RAY_USAGE_STATS_ENABLED', 'NOT SET')}")

exit_code = 0 if success else 1
sys.exit(exit_code)