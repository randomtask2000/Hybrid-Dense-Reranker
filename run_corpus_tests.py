#!/usr/bin/env python3
"""
Comprehensive test runner for corpus configuration functionality.
This script runs all tests related to the new configurable corpus feature.
"""

import os
import sys
import unittest
import subprocess
from pathlib import Path

def setup_test_environment():
    """Set up the test environment"""
    print("Setting up test environment...")
    
    # Ensure we're in the correct directory
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    # Check if virtual environment is activated
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("Warning: Virtual environment not detected. Consider activating your virtual environment.")
    
    # Check for required dependencies
    try:
        import flask
        import anthropic
        import faiss
        import numpy
        import sklearn
        import dotenv
        print("✓ All required dependencies found")
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("Please install requirements: pip install -r requirements.txt")
        return False
    
    # Check for .env file
    if not os.path.exists('.env'):
        print("Warning: .env file not found. Some tests may fail.")
        print("Please copy .env.example to .env and configure your API keys.")
    
    return True

def run_unit_tests():
    """Run unit tests for corpus configuration"""
    print("\n" + "="*60)
    print("RUNNING UNIT TESTS - CORPUS CONFIGURATION")
    print("="*60)
    
    # Discover and run unit tests
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Load tests from test_corpus_config.py
    try:
        from test_corpus_config import (
            TestCorpusConfiguration,
            TestCorpusConfigurationIntegration,
            TestCorpusChunking
        )
        
        suite.addTests(loader.loadTestsFromTestCase(TestCorpusConfiguration))
        suite.addTests(loader.loadTestsFromTestCase(TestCorpusConfigurationIntegration))
        suite.addTests(loader.loadTestsFromTestCase(TestCorpusChunking))
        
        print(f"Loaded {suite.countTestCases()} unit tests")
        
    except ImportError as e:
        print(f"Error loading unit tests: {e}")
        return False
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(suite)
    
    return result.wasSuccessful()

def run_integration_tests():
    """Run integration tests for corpus configuration"""
    print("\n" + "="*60)
    print("RUNNING INTEGRATION TESTS - CORPUS CONFIGURATION")
    print("="*60)
    
    # Discover and run integration tests
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Load tests from test_corpus_integration.py
    try:
        from test_corpus_integration import (
            TestCorpusIntegrationWorkflow,
            TestCorpusConfigurationEdgeCases
        )
        
        suite.addTests(loader.loadTestsFromTestCase(TestCorpusIntegrationWorkflow))
        suite.addTests(loader.loadTestsFromTestCase(TestCorpusConfigurationEdgeCases))
        
        print(f"Loaded {suite.countTestCases()} integration tests")
        
    except ImportError as e:
        print(f"Error loading integration tests: {e}")
        return False
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(suite)
    
    return result.wasSuccessful()

def run_existing_tests():
    """Run existing tests to ensure no regression"""
    print("\n" + "="*60)
    print("RUNNING EXISTING TESTS - REGRESSION CHECK")
    print("="*60)
    
    # Run existing integration tests
    try:
        result = subprocess.run([
            sys.executable, 'test_integration.py'
        ], capture_output=True, text=True, timeout=300)
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("✗ Existing tests timed out")
        return False
    except Exception as e:
        print(f"✗ Error running existing tests: {e}")
        return False

def test_corpus_loading_scenarios():
    """Test specific corpus loading scenarios"""
    print("\n" + "="*60)
    print("TESTING CORPUS LOADING SCENARIOS")
    print("="*60)
    
    scenarios = [
        {
            'name': 'Default Corpus',
            'env': {'CORPUS_SOURCE': 'default'},
            'expected_keywords': ['legal', 'contract', 'liability']
        },
        {
            'name': 'Mormon Corpus (if file exists)',
            'env': {
                'CORPUS_SOURCE': 'mormon',
                'CHUNK_SIZE': '500',
                'CHUNK_OVERLAP': '50'
            },
            'expected_keywords': ['Nephi', 'Jacob'] if os.path.exists('data/mormon13short.txt') else ['legal']
        },
        {
            'name': 'Invalid Source (fallback to default)',
            'env': {'CORPUS_SOURCE': 'invalid'},
            'expected_keywords': ['legal', 'contract']
        }
    ]
    
    all_passed = True
    
    for scenario in scenarios:
        print(f"\nTesting: {scenario['name']}")
        
        # Set environment variables
        original_env = {}
        for key, value in scenario['env'].items():
            original_env[key] = os.getenv(key)
            os.environ[key] = value
        
        try:
            # Import app with new environment
            if 'app' in sys.modules:
                del sys.modules['app']
            
            from app import load_corpus
            
            corpus = load_corpus()
            
            # Verify corpus loaded
            if not corpus:
                print(f"  ✗ No corpus loaded")
                all_passed = False
                continue
            
            print(f"  ✓ Loaded {len(corpus)} documents")
            
            # Check for expected keywords
            found_keywords = []
            for doc in corpus:
                for keyword in scenario['expected_keywords']:
                    if keyword.lower() in doc['content'].lower():
                        found_keywords.append(keyword)
                        break
            
            if found_keywords:
                print(f"  ✓ Found expected content: {found_keywords}")
            else:
                print(f"  ⚠ Expected keywords not found: {scenario['expected_keywords']}")
            
            # Verify document structure
            for i, doc in enumerate(corpus[:3]):  # Check first 3 docs
                if not isinstance(doc, dict) or 'title' not in doc or 'content' not in doc:
                    print(f"  ✗ Document {i} has invalid structure")
                    all_passed = False
                    break
            else:
                print(f"  ✓ Document structure valid")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            all_passed = False
        
        finally:
            # Restore environment variables
            for key, value in original_env.items():
                if value is not None:
                    os.environ[key] = value
                elif key in os.environ:
                    del os.environ[key]
    
    return all_passed

def test_api_endpoints():
    """Test API endpoints with different corpus configurations"""
    print("\n" + "="*60)
    print("TESTING API ENDPOINTS")
    print("="*60)
    
    try:
        from app import app
        client = app.test_client()
        
        test_queries = [
            {"query": "legal risks and liability"},
            {"query": "Nephi and his teachings"},
            {"query": "contract compliance"},
            {"query": ""},  # Empty query
        ]
        
        all_passed = True
        
        for i, query in enumerate(test_queries):
            print(f"\nTesting query {i+1}: '{query['query']}'")
            
            try:
                response = client.post('/rag-query',
                                     json=query,
                                     content_type='application/json')
                
                if response.status_code == 200:
                    data = response.get_json()
                    if isinstance(data, list) and len(data) > 0:
                        print(f"  ✓ Returned {len(data)} results")
                        
                        # Check result structure
                        first_result = data[0]
                        required_fields = ['title', 'content', 'tfidf_score', 'claude_score', 'combined_score']
                        missing_fields = [field for field in required_fields if field not in first_result]
                        
                        if missing_fields:
                            print(f"  ✗ Missing fields: {missing_fields}")
                            all_passed = False
                        else:
                            print(f"  ✓ Result structure valid")
                    else:
                        print(f"  ⚠ No results returned")
                else:
                    print(f"  ✗ HTTP {response.status_code}: {response.get_data(as_text=True)}")
                    all_passed = False
                    
            except Exception as e:
                print(f"  ✗ Error: {e}")
                all_passed = False
        
        return all_passed
        
    except Exception as e:
        print(f"✗ Error setting up API tests: {e}")
        return False

def main():
    """Main test runner"""
    print("CORPUS CONFIGURATION TEST SUITE")
    print("="*60)
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    print(f"Environment variables:")
    print(f"  CORPUS_SOURCE: {os.getenv('CORPUS_SOURCE', 'Not set')}")
    print(f"  CHUNK_SIZE: {os.getenv('CHUNK_SIZE', 'Not set')}")
    print(f"  CHUNK_OVERLAP: {os.getenv('CHUNK_OVERLAP', 'Not set')}")
    print(f"  ANTHROPIC_API_KEY: {'Set' if os.getenv('ANTHROPIC_API_KEY') else 'Not set'}")
    
    # Setup test environment
    if not setup_test_environment():
        print("✗ Test environment setup failed")
        return 1
    
    # Track test results
    results = {}
    
    # Run unit tests
    print("\n" + "="*60)
    print("PHASE 1: UNIT TESTS")
    print("="*60)
    results['unit_tests'] = run_unit_tests()
    
    # Run integration tests
    print("\n" + "="*60)
    print("PHASE 2: INTEGRATION TESTS")
    print("="*60)
    results['integration_tests'] = run_integration_tests()
    
    # Test corpus loading scenarios
    print("\n" + "="*60)
    print("PHASE 3: CORPUS LOADING SCENARIOS")
    print("="*60)
    results['corpus_scenarios'] = test_corpus_loading_scenarios()
    
    # Test API endpoints
    print("\n" + "="*60)
    print("PHASE 4: API ENDPOINT TESTS")
    print("="*60)
    results['api_tests'] = test_api_endpoints()
    
    # Run existing tests for regression check
    print("\n" + "="*60)
    print("PHASE 5: REGRESSION TESTS")
    print("="*60)
    results['regression_tests'] = run_existing_tests()
    
    # Print final summary
    print("\n" + "="*60)
    print("FINAL TEST SUMMARY")
    print("="*60)
    
    total_phases = len(results)
    passed_phases = sum(1 for result in results.values() if result)
    
    for phase, result in results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{phase.replace('_', ' ').title()}: {status}")
    
    print(f"\nOverall: {passed_phases}/{total_phases} phases passed")
    
    if passed_phases == total_phases:
        print("🎉 ALL TESTS PASSED! Corpus configuration is working correctly.")
        return 0
    else:
        print("❌ SOME TESTS FAILED. Please review the output above.")
        return 1

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)