#!/usr/bin/env python3
"""
Test Setup Validator

This script validates that the test environment is properly configured
and can run basic functionality tests before running the full integration suite.
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv

def main():
    """Validate test setup"""
    print("🔧 Validating Test Setup for Hybrid Dense Reranker")
    print("=" * 55)
    
    # Load environment variables
    load_dotenv()
    
    # Check .env file
    if not os.path.exists('.env'):
        print("❌ .env file not found")
        print("   Please copy .env.example to .env and configure it")
        return False
    
    print("✅ .env file found")
    
    # Check API key
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key or api_key == 'your-anthropic-api-key-here':
        print("⚠️  ANTHROPIC_API_KEY not configured")
        print("   Tests will use fallback behavior")
    else:
        print("✅ ANTHROPIC_API_KEY configured")
    
    # Check virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Running in virtual environment")
    else:
        print("⚠️  Not in virtual environment")
    
    # Test imports
    print("\n📦 Testing package imports...")
    
    try:
        import flask
        print("✅ Flask imported successfully")
    except ImportError:
        print("❌ Flask import failed")
        return False
    
    try:
        import numpy
        print("✅ NumPy imported successfully")
    except ImportError:
        print("❌ NumPy import failed")
        return False
    
    try:
        import faiss
        print("✅ FAISS imported successfully")
    except ImportError:
        print("❌ FAISS import failed")
        return False
    
    try:
        import sklearn
        print("✅ Scikit-learn imported successfully")
    except ImportError:
        print("❌ Scikit-learn import failed")
        return False
    
    try:
        import anthropic
        print("✅ Anthropic imported successfully")
    except ImportError:
        print("❌ Anthropic import failed")
        return False
    
    # Test app import
    print("\n🚀 Testing app import...")
    try:
        from app import app, get_embedding, analyze_with_claude, corpus, index
        print("✅ App imported successfully")
        print(f"✅ Corpus loaded with {len(corpus)} documents")
        print(f"✅ FAISS index initialized with {index.ntotal} vectors")
    except Exception as e:
        print(f"❌ App import failed: {e}")
        return False
    
    # Test basic functionality
    print("\n🧪 Testing basic functionality...")
    
    try:
        # Test embedding generation
        test_text = "test document"
        embedding = get_embedding(test_text)
        print(f"✅ Embedding generation works (dimension: {len(embedding)})")
    except Exception as e:
        print(f"❌ Embedding generation failed: {e}")
        return False
    
    try:
        # Test Flask app creation
        test_client = app.test_client()
        print("✅ Flask test client created successfully")
    except Exception as e:
        print(f"❌ Flask test client creation failed: {e}")
        return False
    
    print("\n" + "=" * 55)
    print("✅ All validation checks passed!")
    print("🚀 Ready to run integration tests")
    print("\nNext steps:")
    print("   python test_integration.py")
    print("   or")
    print("   python run_integration_tests.py")
    
    return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)