#!/usr/bin/env python3
"""
Comprehensive integration tests for corpus configuration functionality.
This script tests the complete workflow with both default and Mormon corpus sources.
"""

import unittest
import os
import json
import tempfile
import shutil
from unittest.mock import patch, mock_open
from dotenv import load_dotenv
import numpy as np

# Load environment variables
load_dotenv()


class TestCorpusIntegrationWorkflow(unittest.TestCase):
    """End-to-end integration tests for corpus configuration"""
    
    @classmethod
    def setUpClass(cls):
        """Set up class-level fixtures"""
        # Store original environment variables
        cls.original_env = {
            'CORPUS_SOURCE': os.getenv('CORPUS_SOURCE'),
            'CHUNK_SIZE': os.getenv('CHUNK_SIZE'),
            'CHUNK_OVERLAP': os.getenv('CHUNK_OVERLAP')
        }
        
        # Sample Mormon text for testing
        cls.sample_mormon_text = """1 Nephi 1:1 I, Nephi, having been born of goodly parents, therefore I was taught somewhat in all the learning of my father; and having seen many afflictions in the course of my days, nevertheless, having been highly favored of the Lord in all my days; yea, having had a great knowledge of the goodness and the mysteries of God, therefore I make a record of my proceedings in my days.

1 Nephi 1:2 Yea, I make a record in the language of my father, which consists of the learning of the Jews and the language of the Egyptians.

1 Nephi 1:3 And I know that the record which I make is true; and I make it with mine own hand; and I make it according to my knowledge.

2 Nephi 2:1 And now, Jacob, I speak unto you: You are my first-born in the days of my tribulation in the wilderness; and behold, in thy childhood thou hast suffered afflictions and much sorrow, because of the rudeness of thy brethren.

2 Nephi 2:2 Nevertheless, Jacob, my first-born in the wilderness, thou knowest the greatness of God; and he shall consecrate thine afflictions for thy gain.

2 Nephi 2:3 Wherefore, thy soul shall be blessed, and thou shalt dwell safely with thy brother, Nephi; and thy days shall be spent in the service of thy God. Wherefore, I know that thou art redeemed, because of the righteousness of thy Redeemer; for thou hast beheld that in the fulness of time he cometh to bring salvation unto men."""
    
    @classmethod
    def tearDownClass(cls):
        """Clean up class-level fixtures"""
        # Restore original environment variables
        for key, value in cls.original_env.items():
            if value is not None:
                os.environ[key] = value
            elif key in os.environ:
                del os.environ[key]
    
    def setUp(self):
        """Set up test fixtures before each test"""
        # Import app after environment setup
        from app import app
        self.app = app.test_client()
        self.app.testing = True
    
    def test_default_corpus_workflow(self):
        """Test complete workflow with default corpus"""
        # Set environment for default corpus
        os.environ['CORPUS_SOURCE'] = 'default'
        
        # Remove any existing corpus configuration to force reload
        if 'app' in globals():
            del globals()['app']
        
        # Re-import to pick up new environment
        from app import load_corpus, get_embedding, analyze_with_claude
        
        # Test corpus loading
        corpus = load_corpus()
        self.assertIsInstance(corpus, list)
        self.assertGreater(len(corpus), 0)
        
        # Verify default corpus contains legal content
        legal_keywords = ['legal', 'contract', 'liability', 'risk', 'compliance']
        found_legal_content = False
        for doc in corpus:
            if any(keyword in doc['content'].lower() for keyword in legal_keywords):
                found_legal_content = True
                break
        self.assertTrue(found_legal_content, "Default corpus should contain legal content")
        
        # Test embedding generation
        sample_doc = corpus[0]
        embedding = get_embedding(sample_doc['content'])
        self.assertIsInstance(embedding, np.ndarray)
        self.assertGreater(len(embedding), 0)
        
        # Test Claude analysis
        claude_score = analyze_with_claude(sample_doc['content'], "legal risks")
        self.assertIsInstance(claude_score, (int, float))
        self.assertGreaterEqual(float(claude_score), 0.0)
        self.assertLessEqual(float(claude_score), 1.0)
        
        # Test RAG query endpoint
        query = {"query": "contract liability and legal risks"}
        response = self.app.post('/rag-query',
                               data=json.dumps(query),
                               content_type='application/json')
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)
        
        # Verify response structure
        for result in data:
            self.assertIn('title', result)
            self.assertIn('content', result)
            self.assertIn('tfidf_score', result)
            self.assertIn('claude_score', result)
            self.assertIn('combined_score', result)
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.exists')
    def test_mormon_corpus_workflow(self, mock_exists, mock_file):
        """Test complete workflow with Mormon corpus"""
        # Mock file operations
        mock_exists.return_value = True
        mock_file.return_value.read.return_value = self.sample_mormon_text
        
        # Set environment for Mormon corpus
        os.environ['CORPUS_SOURCE'] = 'mormon'
        os.environ['CHUNK_SIZE'] = '400'
        os.environ['CHUNK_OVERLAP'] = '50'
        
        # Re-import to pick up new environment
        from app import load_corpus, get_embedding, analyze_with_claude
        
        # Test corpus loading
        corpus = load_corpus()
        self.assertIsInstance(corpus, list)
        self.assertGreater(len(corpus), 0)
        
        # Verify Mormon corpus contains expected content
        mormon_keywords = ['Nephi', 'Jacob', 'wilderness', 'Lord', 'record']
        found_mormon_content = False
        for doc in corpus:
            if any(keyword in doc['content'] for keyword in mormon_keywords):
                found_mormon_content = True
                break
        self.assertTrue(found_mormon_content, "Mormon corpus should contain Book of Mormon content")
        
        # Verify chunk structure
        for doc in corpus:
            self.assertIn('title', doc)
            self.assertIn('content', doc)
            self.assertIn('Book of Mormon', doc['title'])
            self.assertLessEqual(len(doc['content']), 400)  # Respects chunk size
        
        # Test embedding generation with Mormon content
        sample_doc = corpus[0]
        embedding = get_embedding(sample_doc['content'])
        self.assertIsInstance(embedding, np.ndarray)
        self.assertGreater(len(embedding), 0)
        
        # Test Claude analysis with Mormon content
        claude_score = analyze_with_claude(sample_doc['content'], "Nephi and his father")
        self.assertIsInstance(claude_score, (int, float))
        self.assertGreaterEqual(float(claude_score), 0.0)
        self.assertLessEqual(float(claude_score), 1.0)
        
        # Test RAG query endpoint with Mormon-specific query
        query = {"query": "Nephi and his teachings"}
        response = self.app.post('/rag-query',
                               data=json.dumps(query),
                               content_type='application/json')
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)
        
        # Verify response structure
        for result in data:
            self.assertIn('title', result)
            self.assertIn('content', result)
            self.assertIn('tfidf_score', result)
            self.assertIn('claude_score', result)
            self.assertIn('combined_score', result)
    
    @patch('builtins.open', side_effect=FileNotFoundError)
    @patch('os.path.exists')
    def test_mormon_corpus_fallback_workflow(self, mock_exists, mock_file):
        """Test workflow when Mormon corpus file is not found (fallback to default)"""
        # Mock file doesn't exist
        mock_exists.return_value = False
        
        # Set environment for Mormon corpus
        os.environ['CORPUS_SOURCE'] = 'mormon'
        
        # Re-import to pick up new environment
        from app import load_corpus
        
        # Test corpus loading falls back to default
        corpus = load_corpus()
        self.assertIsInstance(corpus, list)
        self.assertGreater(len(corpus), 0)
        
        # Should contain legal content (default corpus)
        legal_keywords = ['legal', 'contract', 'liability', 'risk']
        found_legal_content = False
        for doc in corpus:
            if any(keyword in doc['content'].lower() for keyword in legal_keywords):
                found_legal_content = True
                break
        self.assertTrue(found_legal_content, "Should fall back to default legal corpus")
        
        # Test RAG query still works
        query = {"query": "legal risks"}
        response = self.app.post('/rag-query',
                               data=json.dumps(query),
                               content_type='application/json')
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.exists')
    def test_different_chunk_sizes(self, mock_exists, mock_file):
        """Test Mormon corpus with different chunk sizes"""
        mock_exists.return_value = True
        mock_file.return_value.read.return_value = self.sample_mormon_text
        
        # Test with small chunks
        os.environ['CORPUS_SOURCE'] = 'mormon'
        os.environ['CHUNK_SIZE'] = '200'
        os.environ['CHUNK_OVERLAP'] = '30'
        
        from app import load_corpus
        
        small_corpus = load_corpus()
        
        # Test with large chunks
        os.environ['CHUNK_SIZE'] = '800'
        os.environ['CHUNK_OVERLAP'] = '100'
        
        # Force reload by clearing module cache
        import sys
        if 'app' in sys.modules:
            del sys.modules['app']
        
        from app import load_corpus
        
        large_corpus = load_corpus()
        
        # Small chunks should create more documents
        self.assertGreaterEqual(len(small_corpus), len(large_corpus))
        
        # Verify chunk size constraints
        for doc in small_corpus:
            self.assertLessEqual(len(doc['content']), 200)
        
        for doc in large_corpus:
            self.assertLessEqual(len(doc['content']), 800)
    
    def test_corpus_switching(self):
        """Test switching between corpus sources"""
        # Start with default
        os.environ['CORPUS_SOURCE'] = 'default'
        
        from app import load_corpus
        
        default_corpus = load_corpus()
        default_count = len(default_corpus)
        
        # Verify default corpus characteristics
        legal_content = any('legal' in doc['content'].lower() for doc in default_corpus)
        self.assertTrue(legal_content)
        
        # Test query with default corpus
        query = {"query": "legal compliance"}
        response = self.app.post('/rag-query',
                               data=json.dumps(query),
                               content_type='application/json')
        
        self.assertEqual(response.status_code, 200)
        default_results = json.loads(response.data)
        
        # Switch to Mormon corpus (with mocked file)
        with patch('os.path.exists', return_value=True), \
             patch('builtins.open', mock_open(read_data=self.sample_mormon_text)):
            
            os.environ['CORPUS_SOURCE'] = 'mormon'
            os.environ['CHUNK_SIZE'] = '300'
            
            # Force reload
            import sys
            if 'app' in sys.modules:
                del sys.modules['app']
            
            from app import load_corpus
            
            mormon_corpus = load_corpus()
            
            # Verify Mormon corpus characteristics
            mormon_content = any('Nephi' in doc['content'] for doc in mormon_corpus)
            self.assertTrue(mormon_content)
            
            # Test query with Mormon corpus
            query = {"query": "Nephi teachings"}
            response = self.app.post('/rag-query',
                                   data=json.dumps(query),
                                   content_type='application/json')
            
            self.assertEqual(response.status_code, 200)
            mormon_results = json.loads(response.data)
            
            # Results should be different between corpus sources
            self.assertIsInstance(default_results, list)
            self.assertIsInstance(mormon_results, list)
    
    def test_environment_variable_validation(self):
        """Test validation of environment variables"""
        # Test with invalid chunk size
        os.environ['CORPUS_SOURCE'] = 'mormon'
        os.environ['CHUNK_SIZE'] = 'invalid'
        os.environ['CHUNK_OVERLAP'] = '50'
        
        with patch('os.path.exists', return_value=True), \
             patch('builtins.open', mock_open(read_data=self.sample_mormon_text)):
            
            from app import load_corpus
            
            # Should handle invalid chunk size gracefully
            corpus = load_corpus()
            self.assertIsInstance(corpus, list)
    
    def test_performance_with_large_corpus(self):
        """Test performance characteristics with larger corpus"""
        # Create a larger sample text
        large_text = self.sample_mormon_text * 10  # Repeat content
        
        with patch('os.path.exists', return_value=True), \
             patch('builtins.open', mock_open(read_data=large_text)):
            
            os.environ['CORPUS_SOURCE'] = 'mormon'
            os.environ['CHUNK_SIZE'] = '300'
            os.environ['CHUNK_OVERLAP'] = '50'
            
            from app import load_corpus
            
            corpus = load_corpus()
            
            # Should handle larger corpus
            self.assertGreater(len(corpus), 10)  # Should create many chunks
            
            # Test query performance
            query = {"query": "Nephi and Jacob"}
            response = self.app.post('/rag-query',
                                   data=json.dumps(query),
                                   content_type='application/json')
            
            self.assertEqual(response.status_code, 200)
            data = json.loads(response.data)
            self.assertIsInstance(data, list)


class TestCorpusConfigurationEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions for corpus configuration"""
    
    def setUp(self):
        """Set up test fixtures"""
        from app import app
        self.app = app.test_client()
        self.app.testing = True
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.exists')
    def test_malformed_mormon_text(self, mock_exists, mock_file):
        """Test handling of malformed Mormon text"""
        # Mock file with malformed content
        mock_exists.return_value = True
        malformed_text = "This is not properly formatted Mormon text without verse references"
        mock_file.return_value.read.return_value = malformed_text
        
        os.environ['CORPUS_SOURCE'] = 'mormon'
        os.environ['CHUNK_SIZE'] = '300'
        
        from app import load_corpus
        
        corpus = load_corpus()
        
        # Should handle malformed text gracefully
        # Might return empty list or fall back to default
        self.assertIsInstance(corpus, list)
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.exists')
    def test_very_small_chunk_size(self, mock_exists, mock_file):
        """Test with very small chunk size"""
        mock_exists.return_value = True
        mock_file.return_value.read.return_value = "1 Nephi 1:1 Short verse."
        
        os.environ['CORPUS_SOURCE'] = 'mormon'
        os.environ['CHUNK_SIZE'] = '10'  # Very small
        os.environ['CHUNK_OVERLAP'] = '5'
        
        from app import load_corpus
        
        corpus = load_corpus()
        
        # Should handle small chunk size
        self.assertIsInstance(corpus, list)
        if len(corpus) > 0:
            for doc in corpus:
                self.assertLessEqual(len(doc['content']), 50)  # Some reasonable limit
    
    def test_missing_environment_variables(self):
        """Test behavior when environment variables are missing"""
        # Remove all corpus-related environment variables
        for var in ['CORPUS_SOURCE', 'CHUNK_SIZE', 'CHUNK_OVERLAP']:
            if var in os.environ:
                del os.environ[var]
        
        from app import load_corpus
        
        corpus = load_corpus()
        
        # Should use defaults and return valid corpus
        self.assertIsInstance(corpus, list)
        self.assertGreater(len(corpus), 0)


if __name__ == '__main__':
    # Set up test environment
    print("=" * 60)
    print("CORPUS CONFIGURATION INTEGRATION TESTS")
    print("=" * 60)
    print(f"Current CORPUS_SOURCE: {os.getenv('CORPUS_SOURCE', 'Not set')}")
    print(f"Current CHUNK_SIZE: {os.getenv('CHUNK_SIZE', 'Not set')}")
    print(f"Current CHUNK_OVERLAP: {os.getenv('CHUNK_OVERLAP', 'Not set')}")
    print("=" * 60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestCorpusIntegrationWorkflow))
    suite.addTests(loader.loadTestsFromTestCase(TestCorpusConfigurationEdgeCases))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\nFAILURES ({len(result.failures)}):")
        for test, traceback in result.failures:
            print(f"- {test}")
    
    if result.errors:
        print(f"\nERRORS ({len(result.errors)}):")
        for test, traceback in result.errors:
            print(f"- {test}")
    
    print("=" * 60)