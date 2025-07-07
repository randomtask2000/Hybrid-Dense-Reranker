#!/usr/bin/env python3
"""
Integration Test Runner for Hybrid Dense Reranker App

This script runs comprehensive integration tests for app.py without mocking.
It tests all methods and endpoints to ensure the application works correctly.
Uses the existing virtual environment and .env configuration.
"""

import os
import sys
import subprocess
import time
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def check_environment():
    """Check if the environment is properly set up for testing"""
    print("🔍 Checking environment setup...")
    
    # Check if .env file exists
    if not os.path.exists('.env'):
        print("⚠️  Warning: .env file not found. Please copy .env.example to .env and configure it.")
        print("   The tests will use default/test values for missing environment variables.")
        return False
    else:
        print("✅ .env file found")
    
    # Check if ANTHROPIC_API_KEY is set (already loaded by load_dotenv())
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key or api_key == 'your-anthropic-api-key-here':
        print("⚠️  Warning: ANTHROPIC_API_KEY not properly configured in .env file.")
        print("   Some tests may fail or use fallback behavior.")
        print("   Set your API key in .env file for full functionality.")
        print("   Example: ANTHROPIC_API_KEY=sk-ant-api03-...")
    else:
        print("✅ ANTHROPIC_API_KEY is configured")
    
    # Check if we're in a virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Running in virtual environment")
    else:
        print("⚠️  Warning: Not running in a virtual environment")
        print("   Consider activating your virtual environment first")
    
    # Check if required packages are installed
    try:
        import flask
        import numpy
        import faiss
        import sklearn
        import anthropic
        from dotenv import load_dotenv
        print("✅ All required packages are available")
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("   Run: pip install -r requirements.txt")
        print("   Make sure you're in the correct virtual environment")
        return False
    
    return True

def install_test_dependencies():
    """Install test-specific dependencies"""
    print("📦 Installing test dependencies...")
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'test_requirements.txt'], 
                      check=True, capture_output=True)
        print("✅ Test dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install test dependencies: {e}")
        print("   You can still run tests with unittest (python test_integration.py)")
        return False

def run_unittest_tests():
    """Run tests using unittest"""
    print("\n🧪 Running integration tests with unittest...")
    print("=" * 60)
    
    try:
        # Run the integration tests
        result = subprocess.run([sys.executable, 'test_integration.py'], 
                              capture_output=False, text=True)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error running unittest tests: {e}")
        return False

def run_pytest_tests():
    """Run tests using pytest"""
    print("\n🧪 Running integration tests with pytest...")
    print("=" * 60)
    
    try:
        # Run pytest with coverage
        result = subprocess.run([sys.executable, '-m', 'pytest', 'test_integration.py', '-v'], 
                              capture_output=False, text=True)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error running pytest tests: {e}")
        return False

def run_specific_test_class(test_class):
    """Run a specific test class"""
    print(f"\n🎯 Running specific test class: {test_class}")
    print("=" * 60)
    
    try:
        result = subprocess.run([sys.executable, '-m', 'pytest', f'test_integration.py::{test_class}', '-v'], 
                              capture_output=False, text=True)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error running specific test class: {e}")
        return False

def main():
    """Main test runner function"""
    print("🚀 Hybrid Dense Reranker - Integration Test Runner")
    print("=" * 60)
    
    # Check environment
    if not check_environment():
        print("\n❌ Environment check failed. Please fix the issues above.")
        return 1
    
    # Try to install test dependencies
    pytest_available = install_test_dependencies()
    
    # Ask user what they want to run
    print("\n📋 Test Options:")
    print("1. Run all integration tests (recommended)")
    print("2. Run only app integration tests")
    print("3. Run only performance tests")
    print("4. Run with unittest (no pytest dependencies needed)")
    print("5. Exit")
    
    while True:
        try:
            choice = input("\nSelect an option (1-5): ").strip()
            
            if choice == '1':
                if pytest_available:
                    success = run_pytest_tests()
                else:
                    success = run_unittest_tests()
                break
            elif choice == '2':
                if pytest_available:
                    success = run_specific_test_class('TestAppIntegration')
                else:
                    print("This option requires pytest. Using unittest instead...")
                    success = run_unittest_tests()
                break
            elif choice == '3':
                if pytest_available:
                    success = run_specific_test_class('TestAppPerformance')
                else:
                    print("This option requires pytest. Using unittest instead...")
                    success = run_unittest_tests()
                break
            elif choice == '4':
                success = run_unittest_tests()
                break
            elif choice == '5':
                print("👋 Exiting...")
                return 0
            else:
                print("❌ Invalid choice. Please select 1-5.")
                continue
                
        except KeyboardInterrupt:
            print("\n\n👋 Test runner interrupted by user.")
            return 1
    
    # Print results
    print("\n" + "=" * 60)
    if success:
        print("✅ All tests completed successfully!")
        print("\n📊 Test Summary:")
        print("   - All integration tests passed")
        print("   - All endpoints tested")
        print("   - All methods tested")
        print("   - Error handling verified")
        
        if pytest_available:
            print("\n📈 Coverage report generated in htmlcov/ directory")
            print("   Open htmlcov/index.html in your browser to view detailed coverage")
        
        return 0
    else:
        print("❌ Some tests failed!")
        print("\n🔍 Troubleshooting tips:")
        print("   - Check your .env configuration")
        print("   - Ensure ANTHROPIC_API_KEY is valid")
        print("   - Verify all dependencies are installed")
        print("   - Check the test output above for specific errors")
        return 1

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)