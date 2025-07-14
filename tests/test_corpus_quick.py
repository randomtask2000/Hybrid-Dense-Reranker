#!/usr/bin/env python3
"""
Quick validation test for corpus configuration functionality.
This script performs basic smoke tests to ensure the corpus configuration is working.
"""

import os
import sys
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_default_corpus():
    """Test loading default corpus"""
    print("Testing default corpus loading...")
    
    # Set environment for default corpus
    os.environ['CORPUS_SOURCE'] = 'default'
    
    try:
        from app import load_corpus, app
        
        corpus = load_corpus()
        
        if not corpus:
            print("  ✗ Failed to load default corpus")
            return False
        
        print(f"  ✓ Loaded {len(corpus)} documents from default corpus")
        
        # Check structure
        first_doc = corpus[0]
        if 'title' not in first_doc or 'content' not in first_doc:
            print("  ✗ Invalid document structure")
            return False
        
        print(f"  ✓ Document structure valid")
        
        # Check for legal content
        legal_keywords = ['legal', 'contract', 'liability', 'risk', 'compliance']
        found_legal = any(keyword in doc['content'].lower() for doc in corpus for keyword in legal_keywords)
        
        if found_legal:
            print("  ✓ Contains expected legal content")
        else:
            print("  ⚠ No legal content found (unexpected for default corpus)")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def test_mormon_corpus():
    """Test loading Mormon corpus"""
    print("Testing Mormon corpus loading...")
    
    # Check if Mormon text file exists
    if not os.path.exists('data/mormon13short.txt'):
        print("  ⚠ Mormon text file not found, skipping test")
        return True
    
    # Set environment for Mormon corpus
    os.environ['CORPUS_SOURCE'] = 'mormon'
    os.environ['CHUNK_SIZE'] = '500'
    os.environ['CHUNK_OVERLAP'] = '50'
    
    try:
        # Clear module cache to force reload
        if 'app' in sys.modules:
            del sys.modules['app']
        
        from app import load_corpus
        
        corpus = load_corpus()
        
        if not corpus:
            print("  ✗ Failed to load Mormon corpus")
            return False
        
        print(f"  ✓ Loaded {len(corpus)} documents from Mormon corpus")
        
        # Check structure
        first_doc = corpus[0]
        if 'title' not in first_doc or 'content' not in first_doc:
            print("  ✗ Invalid document structure")
            return False
        
        print(f"  ✓ Document structure valid")
        
        # Check for Mormon content
        mormon_keywords = ['Nephi', 'Jacob', 'Lord', 'wilderness', 'record']
        found_mormon = any(keyword in doc['content'] for doc in corpus for keyword in mormon_keywords)
        
        if found_mormon:
            print("  ✓ Contains expected Mormon content")
        else:
            print("  ⚠ No Mormon content found")
        
        # Check chunk sizes
        oversized_chunks = [doc for doc in corpus if len(doc['content']) > 500]
        if oversized_chunks:
            print(f"  ⚠ Found {len(oversized_chunks)} chunks exceeding size limit")
        else:
            print("  ✓ All chunks within size limit")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def test_api_endpoint():
    """Test RAG query API endpoint"""
    print("Testing RAG query API endpoint...")
    
    try:
        from app import app
        
        client = app.test_client()
        
        # Test with a simple query
        query = {"query": "test query"}
        response = client.post('/rag-query',
                             json=query,
                             content_type='application/json')
        
        if response.status_code != 200:
            print(f"  ✗ API returned status {response.status_code}")
            return False
        
        data = response.get_json()
        
        if not isinstance(data, list):
            print("  ✗ API response is not a list")
            return False
        
        if len(data) == 0:
            print("  ⚠ API returned no results")
            return True
        
        print(f"  ✓ API returned {len(data)} results")
        
        # Check result structure
        first_result = data[0]
        required_fields = ['title', 'content', 'tfidf_score', 'claude_score', 'combined_score']
        missing_fields = [field for field in required_fields if field not in first_result]
        
        if missing_fields:
            print(f"  ✗ Missing fields in result: {missing_fields}")
            return False
        
        print("  ✓ Result structure valid")
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def test_environment_variables():
    """Test environment variable handling"""
    print("Testing environment variable handling...")
    
    # Test with missing variables
    original_vars = {}
    for var in ['CORPUS_SOURCE', 'CHUNK_SIZE', 'CHUNK_OVERLAP']:
        original_vars[var] = os.getenv(var)
        if var in os.environ:
            del os.environ[var]
    
    try:
        # Clear module cache
        if 'app' in sys.modules:
            del sys.modules['app']
        
        from app import load_corpus
        
        corpus = load_corpus()
        
        if not corpus:
            print("  ✗ Failed to load corpus with missing environment variables")
            return False
        
        print("  ✓ Successfully loaded corpus with default values")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False
    
    finally:
        # Restore environment variables
        for var, value in original_vars.items():
            if value is not None:
                os.environ[var] = value

def main():
    """Run quick validation tests"""
    print("CORPUS CONFIGURATION QUICK TESTS")
    print("=" * 50)
    print(f"Current environment:")
    print(f"  CORPUS_SOURCE: {os.getenv('CORPUS_SOURCE', 'Not set')}")
    print(f"  CHUNK_SIZE: {os.getenv('CHUNK_SIZE', 'Not set')}")
    print(f"  CHUNK_OVERLAP: {os.getenv('CHUNK_OVERLAP', 'Not set')}")
    print(f"  ANTHROPIC_API_KEY: {'Set' if os.getenv('ANTHROPIC_API_KEY') else 'Not set'}")
    print("=" * 50)
    
    tests = [
        ("Default Corpus", test_default_corpus),
        ("Mormon Corpus", test_mormon_corpus),
        ("API Endpoint", test_api_endpoint),
        ("Environment Variables", test_environment_variables),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"  ✗ Unexpected error: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("QUICK TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All quick tests passed! Corpus configuration appears to be working.")
        return 0
    else:
        print("❌ Some tests failed. Check the output above for details.")
        return 1

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)