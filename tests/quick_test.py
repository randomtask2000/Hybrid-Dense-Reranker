#!/usr/bin/env python3
"""
Quick Test Runner - Run a subset of tests to verify fixes
"""

import unittest
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import test classes
from test_integration import TestAppIntegration

class QuickTestSuite(unittest.TestCase):
    """Quick test suite with the most important tests"""
    
    def setUp(self):
        from app import app
        self.app = app.test_client()
        self.app.testing = True
    
    def test_basic_functionality(self):
        """Test basic app functionality"""
        from app import get_embedding, analyze_with_claude
        
        # Test embedding generation
        embedding = get_embedding("test text")
        self.assertIsNotNone(embedding)
        
        # Test Claude analysis
        score = analyze_with_claude("legal contract", "legal risks")
        self.assertIsInstance(score, (int, float))
        self.assertGreaterEqual(float(score), 0.0)
        self.assertLessEqual(float(score), 1.0)
    
    def test_endpoint_basic(self):
        """Test basic endpoint functionality"""
        import json
        
        response = self.app.post('/rag-query',
                               data=json.dumps({"query": "legal risks"}),
                               content_type='application/json')
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)

if __name__ == '__main__':
    print("🚀 Running Quick Integration Tests")
    print("=" * 40)
    
    # Run quick tests
    suite = unittest.TestLoader().loadTestsFromTestCase(QuickTestSuite)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    if result.wasSuccessful():
        print("\n✅ Quick tests passed! Ready to run full test suite.")
        print("Run: python test_integration.py")
    else:
        print("\n❌ Quick tests failed. Check the errors above.")
        sys.exit(1)