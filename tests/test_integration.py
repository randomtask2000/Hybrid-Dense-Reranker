import unittest
import json
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv
import numpy as np

# Load environment variables
load_dotenv()

from src.hybrid_dense_reranker.app import create_app
from src.hybrid_dense_reranker.search import HybridReranker
from src.hybrid_dense_reranker.claude_client import ClaudeClient
from src.hybrid_dense_reranker.embeddings import EmbeddingManager


class TestAppIntegration(unittest.TestCase):
    """Integration tests for the restructured application."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create Flask app for testing
        self.flask_app = create_app()
        self.app = self.flask_app.test_client()
        self.flask_app.testing = True
        
        # Initialize components
        self.reranker = HybridReranker()
        self.claude_client = ClaudeClient()
        self.embedding_manager = EmbeddingManager()
        
        # Fit embedding manager with corpus
        texts = [doc["content"] for doc in self.reranker.corpus]
        self.embedding_manager.fit(texts)
        
        # Check API key
        self.anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
        if not self.anthropic_api_key or self.anthropic_api_key == 'your-anthropic-api-key-here':
            print("Warning: ANTHROPIC_API_KEY not configured")
    
    def test_health_endpoint(self):
        """Test the health check endpoint."""
        response = self.app.get('/health')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertIn('status', data)
        self.assertEqual(data['status'], 'healthy')
        self.assertIn('corpus_size', data)
        self.assertGreater(data['corpus_size'], 0)
    
    def test_rag_query_endpoint_valid(self):
        """Test the RAG query endpoint with valid input."""
        query_data = {"query": "legal risks and liability"}
        response = self.app.post('/rag-query',
                                data=json.dumps(query_data),
                                content_type='application/json')
        
        self.assertEqual(response.status_code, 200)
        
        # Parse response
        results = json.loads(response.data)
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        
        # Check result structure
        for result in results:
            self.assertIn('title', result)
            self.assertIn('content', result)
            self.assertIn('tfidf_score', result)
            self.assertIn('claude_score', result)
            self.assertIn('combined_score', result)
    
    def test_rag_query_endpoint_missing_query(self):
        """Test the RAG query endpoint with missing query field."""
        response = self.app.post('/rag-query',
                                data=json.dumps({}),
                                content_type='application/json')
        
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)
    
    def test_rag_query_endpoint_empty_query(self):
        """Test the RAG query endpoint with empty query."""
        query_data = {"query": ""}
        response = self.app.post('/rag-query',
                                data=json.dumps(query_data),
                                content_type='application/json')
        
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)
    
    def test_embedding_generation(self):
        """Test embedding generation functionality."""
        text = "This is a test document about legal risks"
        embedding = self.embedding_manager.get_embedding(text)
        
        # Verify embedding properties
        self.assertIsInstance(embedding, np.ndarray)
        self.assertEqual(len(embedding.shape), 1)
        self.assertGreater(len(embedding), 0)
        self.assertTrue(np.isfinite(embedding).all())
    
    def test_claude_analysis(self):
        """Test Claude relevance analysis."""
        text = "The contract has legal liability issues"
        query = "legal risks"
        
        score = self.claude_client.analyze_relevance(text, query)
        
        # Verify score properties
        self.assertIsInstance(score, (int, float))
        self.assertGreaterEqual(float(score), 0.0)
        self.assertLessEqual(float(score), 1.0)
    
    def test_hybrid_search(self):
        """Test the hybrid search functionality."""
        query = "security and authentication"
        results = self.reranker.search(query, k=2)
        
        # Verify results
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        self.assertLessEqual(len(results), 2)
        
        # Check result structure
        for result in results:
            self.assertIn('title', result)
            self.assertIn('content', result)
            self.assertIn('tfidf_score', result)
            self.assertIn('claude_score', result)
            self.assertIn('combined_score', result)
        
        # Check that results are sorted by combined score
        if len(results) > 1:
            for i in range(len(results) - 1):
                self.assertGreaterEqual(results[i]['combined_score'], 
                                      results[i + 1]['combined_score'])


class TestAppPerformance(unittest.TestCase):
    """Performance tests for the restructured application."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.flask_app = create_app()
        self.app = self.flask_app.test_client()
        self.flask_app.testing = True
    
    def test_multiple_requests(self):
        """Test handling multiple requests."""
        queries = [
            {"query": "legal risks"},
            {"query": "security measures"},
            {"query": "financial performance"}
        ]
        
        for query_data in queries:
            response = self.app.post('/rag-query',
                                   data=json.dumps(query_data),
                                   content_type='application/json')
            self.assertEqual(response.status_code, 200)
            
            results = json.loads(response.data)
            self.assertIsInstance(results, list)
            self.assertGreater(len(results), 0)


if __name__ == '__main__':
    unittest.main()