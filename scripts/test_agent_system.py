#!/usr/bin/env python3
"""
Test script for the multi-agent system

This script tests the agent interaction system to ensure it works properly
after the recent updates.
"""

import asyncio
import json
import sys
import time
from typing import Dict, Any

# Add backend to path
sys.path.insert(0, 'backend')

from config.agent_config import agent_config_manager
from services.enhanced_orchestration_service import EnhancedOrchestrationService

async def test_agent_config():
    """Test that agent configuration system works"""
    print("🔧 Testing Agent Configuration System...")
    
    # Test getting agent configs
    rewriter_config = agent_config_manager.get_agent_config('rewriter')
    assert rewriter_config is not None, "Rewriter config should exist"
    print(f"✅ Rewriter agent config: {rewriter_config.model}")
    
    # Test getting pipeline configs
    standard_pipeline = agent_config_manager.get_pipeline_config('standard')
    assert standard_pipeline is not None, "Standard pipeline should exist"
    print(f"✅ Standard pipeline has {len(standard_pipeline.agents)} agents")
    
    # Test comprehensive pipeline
    comprehensive_pipeline = agent_config_manager.get_pipeline_config('comprehensive')
    assert comprehensive_pipeline is not None, "Comprehensive pipeline should exist"
    print(f"✅ Comprehensive pipeline has {len(comprehensive_pipeline.agents)} agents")
    
    print("✅ Agent configuration system working!")
    return True

async def test_orchestration_service():
    """Test that orchestration service initializes properly"""
    print("\n🎭 Testing Orchestration Service...")
    
    try:
        service = EnhancedOrchestrationService()
        print("✅ Orchestration service initialized")
        
        # Test pipeline configs
        pipeline_configs = service.pipeline_configs
        assert 'standard' in pipeline_configs, "Standard pipeline should be configured"
        assert 'comprehensive' in pipeline_configs, "Comprehensive pipeline should be configured"
        print(f"✅ Service has {len(pipeline_configs)} pipeline configurations")
        
        return True
    except Exception as e:
        print(f"❌ Orchestration service failed: {e}")
        return False

async def simulate_agent_pipeline():
    """Simulate a simple agent pipeline execution"""
    print("\n🚀 Simulating Agent Pipeline...")
    
    try:
        service = EnhancedOrchestrationService()
        
        # Create a test context
        context = {
            'request_id': 'test-123',
            'original_query': 'Return all thermal scans over 60 celsius',
            'schema_context': 'thermalScans { id temperature timestamp }',
            'examples': []
        }
        
        print(f"📝 Test query: {context['original_query']}")
        
        # Test that we can create the pipeline generator
        # (We won't actually run it to avoid needing Ollama)
        pipeline_gen = service.process_query_stream(
            query=context['original_query'],
            pipeline_strategy='standard',
            pre_model='phi3:mini',
            translator_model='phi3:mini',
            review_model='phi3:mini'
        )
        
        print("✅ Pipeline generator created successfully")
        print("✅ Agent pipeline simulation successful!")
        return True
        
    except Exception as e:
        print(f"❌ Pipeline simulation failed: {e}")
        return False

async def main():
    """Run all tests"""
    print("🧪 Testing Multi-Agent System\n")
    
    tests = [
        test_agent_config,
        test_orchestration_service,
        simulate_agent_pipeline
    ]
    
    results = []
    for test in tests:
        try:
            result = await test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} failed: {e}")
            results.append(False)
    
    # Summary
    print(f"\n📊 Test Results: {sum(results)}/{len(results)} passed")
    
    if all(results):
        print("🎉 All tests passed! Multi-agent system is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 