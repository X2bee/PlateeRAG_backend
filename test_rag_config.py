#!/usr/bin/env python3
"""
Test script for RAG service configuration and debug functionality
"""

import asyncio
import json
from pathlib import Path

# Add the project root to the Python path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from config.config_composer import ConfigComposer
from service.retrieval.rag_service import RAGService


async def test_rag_config():
    """Test the RAG service configuration and debug functionality"""
    print("🧪 Testing RAG Service Configuration and Debug Functionality")
    print("=" * 60)
    
    try:
        # Initialize config
        config_composer = ConfigComposer()
        config = config_composer.get_config()
        
        # Initialize RAG service
        rag_service = RAGService(config)
        
        # Test 1: Get all configuration
        print("\n1️⃣ Testing get_config() method:")
        print("-" * 40)
        
        config_data = rag_service.get_config()
        print(f"Configuration retrieved successfully!")
        print(f"Config keys: {list(config_data.keys())}")
        
        # Pretty print the config
        print("\n📋 Current Configuration:")
        for key, value in config_data.items():
            print(f"  {key}: {type(value).__name__}")
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    print(f"    {sub_key}: {sub_value}")
            else:
                print(f"    {value}")
        
        # Test 2: Test generate_query_embedding without debug
        print("\n2️⃣ Testing generate_query_embedding() without debug:")
        print("-" * 40)
        
        test_query = "What is machine learning?"
        embedding = await rag_service.generate_query_embedding(test_query, debug=False)
        print(f"Embedding generated successfully!")
        print(f"Embedding type: {type(embedding)}")
        print(f"Embedding dimension: {len(embedding)}")
        print(f"First 5 values: {embedding[:5]}")
        
        # Test 3: Test generate_query_embedding with debug
        print("\n3️⃣ Testing generate_query_embedding() with debug:")
        print("-" * 40)
        
        debug_result = await rag_service.generate_query_embedding(test_query, debug=True)
        print(f"Debug result generated successfully!")
        print(f"Debug result type: {type(debug_result)}")
        print(f"Debug result keys: {list(debug_result.keys())}")
        
        # Pretty print the debug result
        print("\n🔍 Debug Result:")
        for key, value in debug_result.items():
            if key == "embedding_vector":
                print(f"  {key}: List of {len(value)} floats")
                print(f"    First 5 values: {value[:5]}")
                print(f"    Last 5 values: {value[-5:]}")
            else:
                print(f"  {key}: {value}")
        
        # Test 4: Test embedding status
        print("\n4️⃣ Testing embedding status:")
        print("-" * 40)
        
        embedding_status = rag_service.get_embedding_status()
        print(f"Embedding status retrieved successfully!")
        print(f"Status: {embedding_status}")
        
        print("\n✅ All tests passed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_rag_config())
