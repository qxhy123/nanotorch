#!/usr/bin/env python3
"""
Test script for Transformer API endpoints
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_forward_endpoint():
    """Test the /forward endpoint"""
    print("Testing /forward endpoint...")

    payload = {
        "config": {
            "d_model": 512,
            "nhead": 8,
            "num_encoder_layers": 2,
            "num_decoder_layers": 0,
            "dim_feedforward": 2048,
            "dropout": 0.1,
            "activation": "relu",
            "max_seq_len": 128,
            "vocab_size": 10000,
            "layer_norm_eps": 1e-5,
            "batch_first": True,
            "norm_first": False
        },
        "input_data": {
            "text": "Hello world"
        },
        "options": {
            "return_attention": True,
            "return_all_layers": True,
            "return_embeddings": True
        }
    }

    try:
        response = requests.post(f"{BASE_URL}/api/v1/transformer/forward", json=payload)
        response.raise_for_status()
        result = response.json()

        print(f"  Status: {response.status_code}")
        print(f"  Success: {result.get('success')}")
        if result.get('success'):
            data = result.get('data', {})
            print(f"  Final output shape: {data.get('final_output', {}).get('shape')}")
            print(f"  Embeddings: {'✓' if data.get('embeddings') else '✗'}")
            print(f"  Attention weights: {len(data.get('attention_weights', []))} layers")
        else:
            print(f"  Error: {result.get('error')}")

        return result.get('success', False)
    except Exception as e:
        print(f"  Error: {e}")
        return False

def test_attention_endpoint():
    """Test the /attention endpoint"""
    print("\nTesting /attention endpoint...")

    payload = {
        "config": {
            "d_model": 512,
            "nhead": 8,
            "num_encoder_layers": 2,
            "num_decoder_layers": 0,
            "dim_feedforward": 2048,
            "dropout": 0.1,
            "activation": "relu",
            "max_seq_len": 128,
            "vocab_size": 10000,
            "layer_norm_eps": 1e-5,
            "batch_first": True,
            "norm_first": False
        },
        "input_data": {
            "text": "Test"
        },
        "layer_index": 0
    }

    try:
        response = requests.post(f"{BASE_URL}/api/v1/transformer/attention", json=payload)
        response.raise_for_status()
        result = response.json()

        print(f"  Status: {response.status_code}")
        print(f"  Success: {result.get('success')}")

        return result.get('success', False)
    except Exception as e:
        print(f"  Error: {e}")
        return False

def test_embeddings_endpoint():
    """Test the /embeddings endpoint"""
    print("\nTesting /embeddings endpoint...")

    payload = {
        "config": {
            "d_model": 512,
            "nhead": 8,
            "num_encoder_layers": 2,
            "num_decoder_layers": 0,
            "dim_feedforward": 2048,
            "dropout": 0.1,
            "activation": "relu",
            "max_seq_len": 128,
            "vocab_size": 10000,
            "layer_norm_eps": 1e-5,
            "batch_first": True,
            "norm_first": False
        },
        "input_data": {
            "text": "Hello"
        }
    }

    try:
        response = requests.post(f"{BASE_URL}/api/v1/transformer/embeddings", json=payload)
        response.raise_for_status()
        result = response.json()

        print(f"  Status: {response.status_code}")
        print(f"  Success: {result.get('success')}")

        return result.get('success', False)
    except Exception as e:
        print(f"  Error: {e}")
        return False

def test_health_endpoint():
    """Test the health endpoint"""
    print("\nTesting /health endpoint...")

    try:
        response = requests.get(f"{BASE_URL}/health")
        response.raise_for_status()
        result = response.json()

        print(f"  Status: {result.get('status')}")
        print(f"  Version: {result.get('version')}")
        print(f"  nanotorch available: {result.get('nanotorch_available')}")

        return result.get('status') == 'healthy'
    except Exception as e:
        print(f"  Error: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("Transformer API Test Suite")
    print("=" * 50)

    # Check if server is running
    try:
        requests.get(BASE_URL, timeout=2)
    except Exception:
        print("\n❌ Error: Backend server is not running!")
        print("\nPlease start the backend first:")
        print("  cd /Users/yangyang/ai_projs/nanotorch/backend")
        print("  PYTHONPATH=/Users/yangyang/ai_projs/nanotorch:/Users/yangyang/ai_projs/nanotorch/backend python -m uvicorn app.main:app --reload")
        exit(1)

    # Run tests
    tests = [
        ("Health Check", test_health_endpoint),
        ("Forward Pass", test_forward_endpoint),
        ("Attention Weights", test_attention_endpoint),
        ("Embeddings", test_embeddings_endpoint),
    ]

    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"  ❌ {name} failed with exception: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "=" * 50)
    print("Test Results Summary")
    print("=" * 50)

    for name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"  {name}: {status}")

    all_passed = all(success for _, success in results)

    if all_passed:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed. Please check the logs above.")
