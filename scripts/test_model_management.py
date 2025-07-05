#!/usr/bin/env python3
"""
Test script for model management functionality.
This script tests the model management endpoints to ensure they work correctly.
"""

import asyncio
import httpx
import json
import sys
from typing import List, Dict, Any

# Configuration
BASE_URL = "http://localhost:8000"
MODELS_TO_TEST = [
    "phi3:mini",
    "gemma3:2b", 
    "gemma3:7b",
    "llama4:3b",
    "llama4:7b",
    "llama4:14b"
]

async def test_model_endpoints():
    """Test all model management endpoints."""
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        print("🧪 Testing Model Management Endpoints")
        print("=" * 50)
        
        # Test 1: Health check
        print("\n1. Testing health check...")
        try:
            response = await client.get(f"{BASE_URL}/api/models/health/status")
            if response.status_code == 200:
                health_data = response.json()
                print(f"✅ Health check passed: {health_data['status']}")
            else:
                print(f"❌ Health check failed: {response.status_code}")
        except Exception as e:
            print(f"❌ Health check error: {e}")
        
        # Test 2: List models
        print("\n2. Testing list models...")
        try:
            response = await client.get(f"{BASE_URL}/api/models/")
            if response.status_code == 200:
                models_data = response.json()
                available_models = [m['name'] for m in models_data['models']]
                print(f"✅ Found {len(available_models)} available models:")
                for model in available_models:
                    print(f"   - {model}")
            else:
                print(f"❌ List models failed: {response.status_code}")
        except Exception as e:
            print(f"❌ List models error: {e}")
        
        # Test 3: Check specific model status
        print("\n3. Testing model status checks...")
        for model in MODELS_TO_TEST:
            try:
                response = await client.get(f"{BASE_URL}/api/models/{model}")
                if response.status_code == 200:
                    model_info = response.json()
                    print(f"✅ {model}: Available (Size: {model_info.get('size', 'Unknown')})")
                elif response.status_code == 404:
                    print(f"❌ {model}: Not available")
                else:
                    print(f"⚠️ {model}: Status unknown ({response.status_code})")
            except Exception as e:
                print(f"❌ {model}: Error checking status - {e}")
        
        # Test 4: Test model download (only for one model to avoid long waits)
        print("\n4. Testing model download...")
        test_model = "phi3:mini"
        try:
            print(f"   Downloading {test_model}...")
            response = await client.post(f"{BASE_URL}/api/models/{test_model}/pull/stream")
            
            if response.status_code == 200:
                print(f"✅ Download stream started for {test_model}")
                # Read the stream to see progress
                async for line in response.aiter_lines():
                    if line.startswith('data: '):
                        try:
                            data = json.loads(line[6:])
                            print(f"   Progress: {data.get('progress', 0)}% - {data.get('status', 'unknown')}")
                            if data.get('status') == 'completed':
                                print(f"✅ {test_model} download completed!")
                                break
                            elif data.get('status') == 'failed':
                                print(f"❌ {test_model} download failed: {data.get('error', 'Unknown error')}")
                                break
                        except json.JSONDecodeError:
                            continue
            else:
                print(f"❌ Download failed: {response.status_code}")
        except Exception as e:
            print(f"❌ Download error: {e}")
        
        print("\n" + "=" * 50)
        print("🏁 Model management tests completed!")

async def test_pipeline_with_models():
    """Test the pipeline with different models."""
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        print("\n🧪 Testing Pipeline with Different Models")
        print("=" * 50)
        
        test_query = "Show me all users"
        
        for model in ["phi3:mini", "gemma3:2b"]:  # Test with smaller models
            print(f"\nTesting pipeline with {model}...")
            try:
                response = await client.post(
                    f"{BASE_URL}/api/multiagent/process/stream",
                    json={
                        "query": test_query,
                        "pipeline_strategy": "fast",
                        "translator_model": model,
                        "pre_model": model,
                        "review_model": model
                    }
                )
                
                if response.status_code == 200:
                    print(f"✅ Pipeline started with {model}")
                    # Read a few events to see if it's working
                    event_count = 0
                    async for line in response.aiter_lines():
                        if line.startswith('data: '):
                            try:
                                data = json.loads(line[6:])
                                event_type = data.get('event', 'unknown')
                                print(f"   Event: {event_type}")
                                event_count += 1
                                if event_count >= 5:  # Just check first few events
                                    break
                            except json.JSONDecodeError:
                                continue
                else:
                    print(f"❌ Pipeline failed with {model}: {response.status_code}")
            except Exception as e:
                print(f"❌ Pipeline error with {model}: {e}")

async def main():
    """Main test function."""
    print("🚀 Starting Model Management Tests")
    print("Make sure the backend is running on http://localhost:8000")
    print("=" * 60)
    
    try:
        await test_model_endpoints()
        await test_pipeline_with_models()
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 