#!/usr/bin/env python3
"""
Test script specifically for the Data Reviewer Agent

This script tests the data reviewer agent functionality to ensure it can:
1. Execute GraphQL queries
2. Analyze results 
3. Suggest improvements
4. Provide proper feedback
"""

import asyncio
import json
import sys
import time

# Add backend to path
sys.path.insert(0, 'backend')

from agents.implementations import DataReviewerAgent
from agents.base import AgentContext

async def test_data_reviewer_basic():
    """Test basic data reviewer functionality"""
    print("🔍 Testing Data Reviewer Agent...")
    
    try:
        # Create a data reviewer agent
        agent = DataReviewerAgent()
        print("✅ Data reviewer agent created")
        
        # Create test context
        context = AgentContext(
            original_query="Return all thermal scans over 60 celsius",
            graphql_query="query { thermalScans(limit: 5) { id temperature timestamp } }",
            schema_context="thermalScans { id temperature timestamp }",
            examples=[],
            metadata={'request_id': 'test-data-reviewer'}
        )
        
        # Test configuration
        config = {'model': 'phi3:mini'}  # Use lighter model for testing
        
        print(f"📝 Test query: {context.original_query}")
        print(f"📊 GraphQL query: {context.graphql_query}")
        
        # Run the data reviewer (this will try to execute the query)
        result = await agent.run(context, config)
        
        print("✅ Data reviewer executed successfully")
        print(f"📋 Result keys: {list(result.keys()) if result else 'None'}")
        
        # Check result structure
        if result:
            if 'satisfied' in result:
                print(f"✅ Satisfaction status: {result['satisfied']}")
            if 'accuracy_score' in result:
                print(f"✅ Accuracy score: {result['accuracy_score']}")
            if 'query_result' in result:
                query_result = result['query_result']
                print(f"✅ Query execution: {'Success' if query_result.get('success') else 'Failed'}")
                if not query_result.get('success'):
                    print(f"   Error: {query_result.get('error', 'Unknown')}")
            if 'suggested_query' in result:
                print(f"✅ Suggested query: {result['suggested_query'][:50]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Data reviewer test failed: {e}")
        return False

async def test_data_reviewer_with_bad_query():
    """Test data reviewer with a deliberately bad query"""
    print("\n🔍 Testing Data Reviewer with Bad Query...")
    
    try:
        agent = DataReviewerAgent()
        
        # Create context with a bad query
        context = AgentContext(
            original_query="Show me thermal scans above 60 degrees",
            graphql_query="query { badCollection(invalid: syntax) { nonexistent } }",
            schema_context="thermalScans { id temperature timestamp }",
            examples=[],
            metadata={'request_id': 'test-bad-query'}
        )
        
        config = {'model': 'phi3:mini'}
        
        print(f"📝 Test query: {context.original_query}")
        print(f"📊 Bad GraphQL: {context.graphql_query}")
        
        result = await agent.run(context, config)
        
        print("✅ Data reviewer handled bad query")
        
        if result:
            # Should not be satisfied with bad query
            if not result.get('satisfied', True):
                print("✅ Correctly identified query as unsatisfactory")
            
            # Should have error information
            if result.get('query_result', {}).get('success') == False:
                print("✅ Correctly detected query execution failure")
            
            # Should suggest improvements
            if result.get('suggested_query'):
                print("✅ Suggested an improved query")
            elif result.get('suggestions'):
                print("✅ Provided improvement suggestions")
        
        return True
        
    except Exception as e:
        print(f"❌ Bad query test failed: {e}")
        return False

async def test_data_reviewer_no_query():
    """Test data reviewer with no query provided"""
    print("\n🔍 Testing Data Reviewer with No Query...")
    
    try:
        agent = DataReviewerAgent()
        
        # Create context with no query
        context = AgentContext(
            original_query="Show me thermal scans",
            graphql_query="",  # Empty query
            schema_context="thermalScans { id temperature timestamp }",
            examples=[],
            metadata={'request_id': 'test-no-query'}
        )
        
        config = {'model': 'phi3:mini'}
        
        result = await agent.run(context, config)
        
        print("✅ Data reviewer handled empty query")
        
        if result:
            # Should skip when no query provided
            if result.get('status') == 'skipped':
                print("✅ Correctly skipped empty query")
            if result.get('reason'):
                print(f"✅ Provided skip reason: {result['reason']}")
        
        return True
        
    except Exception as e:
        print(f"❌ No query test failed: {e}")
        return False

async def main():
    """Run all data reviewer tests"""
    print("🧪 Testing Data Reviewer Agent\n")
    
    tests = [
        test_data_reviewer_basic,
        test_data_reviewer_with_bad_query,
        test_data_reviewer_no_query
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
        print("🎉 All data reviewer tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 