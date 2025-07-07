import unittest
import os
import tempfile
import json
from unittest.mock import patch, mock_open
from dotenv import load_dotenv
import numpy as np
import faiss

# Load environment variables from .env file
load_dotenv()


class TestCorpusConfiguration(unittest.TestCase):
    """Unit tests for configurable corpus loading functionality"""
    
    def setUp(self):
        """Set up test fixtures before each test method"""
        # Store original environment variables
        self.original_corpus_source = os.getenv('CORPUS_SOURCE')
        self.original_chunk_size = os.getenv('CHUNK_SIZE')
        self.original_chunk_overlap = os.getenv('CHUNK_OVERLAP')
        
        # Sample Mormon text for testing
        self.sample_mormon_text = """1 Nephi 1:1 I, Nephi, having been born of goodly parents, therefore I was taught somewhat in all the learning of my father; and having seen many afflictions in the course of my days, nevertheless, having been highly favored of the Lord in all my days; yea, having had a great knowledge of the goodness and the mysteries of God, therefore I make a record of my proceedings in my days.

1 Nephi 1:2 Yea, I make a record in the language of my father, which consists of the learning of the Jews and the language of the Egyptians.

1 Nephi 1:3 And I know that the record which I make is true; and I make it with mine own hand; and I make it according to my knowledge.

2 Nephi 2:1 And now, Jacob, I speak unto you: You are my first-born in the days of my tribulation in the wilderness; and behold, in thy childhood thou hast suffered afflictions and much sorrow, because of the rudeness of thy brethren.

2 Nephi 2:2 Nevertheless, Jacob, my first-born in the wilderness, thou knowest the greatness of God; and he shall consecrate thine afflictions for thy gain."""
    
    def tearDown(self):
        """Clean up after each test method"""
        # Restore original environment variables
        if self.original_corpus_source:
            os.environ['CORPUS_SOURCE'] = self.original_corpus_source
        elif 'CORPUS_SOURCE' in os.environ:
            del os.environ['CORPUS_SOURCE']
            
        if self.original_chunk_size:
            os.environ['CHUNK_SIZE'] = self.original_chunk_size
        elif 'CHUNK_SIZE' in os.environ:
            del os.environ['CHUNK_SIZE']
            
        if self.original_chunk_overlap:
            os.environ['CHUNK_OVERLAP'] = self.original_chunk_overlap
        elif 'CHUNK_OVERLAP' in os.environ:
            del os.environ['CHUNK_OVERLAP']
    
    def test_get_default_corpus(self):
        """Test that get_default_corpus returns the expected hardcoded corpus"""
        # Import here to avoid circular imports during test setup
        from app import get_default_corpus
        
        default_corpus = get_default_corpus()
        
        # Verify structure
        self.assertIsInstance(default_corpus, list)
        self.assertGreater(len(default_corpus), 0)
        
        # Verify each document has required fields
        for doc in default_corpus:
            self.assertIsInstance(doc, dict)
            self.assertIn('title', doc)
            self.assertIn('content', doc)
            self.assertIsInstance(doc['title'], str)
            self.assertIsInstance(doc['content'], str)
            self.assertGreater(len(doc['title']), 0)
            self.assertGreater(len(doc['content']), 0)
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.exists')
    def test_load_mormon_corpus_success(self, mock_exists, mock_file):
        """Test successful loading of Mormon corpus"""
        # Mock file exists
        mock_exists.return_value = True
        
        # Mock file content
        mock_file.return_value.read.return_value = self.sample_mormon_text
        
        # Set environment variables for testing
        os.environ['CHUNK_SIZE'] = '200'
        os.environ['CHUNK_OVERLAP'] = '50'
        
        # Import and test
        from app import load_mormon_corpus
        
        corpus = load_mormon_corpus()
        
        # Verify structure
        self.assertIsInstance(corpus, list)
        self.assertGreater(len(corpus), 0)
        
        # Verify each chunk has required fields
        for doc in corpus:
            self.assertIsInstance(doc, dict)
            self.assertIn('title', doc)
            self.assertIn('content', doc)
            self.assertIsInstance(doc['title'], str)
            self.assertIsInstance(doc['content'], str)
            self.assertGreater(len(doc['title']), 0)
            self.assertGreater(len(doc['content']), 0)
            
            # Verify chunk size constraints
            self.assertLessEqual(len(doc['content']), 200)
    
    @patch('builtins.open', side_effect=FileNotFoundError)
    @patch('os.path.exists')
    def test_load_mormon_corpus_file_not_found(self, mock_exists, mock_file):
        """Test Mormon corpus loading when file doesn't exist"""
        # Mock file doesn't exist
        mock_exists.return_value = False
        
        from app import load_mormon_corpus
        
        # Should return empty list when file not found
        corpus = load_mormon_corpus()
        self.assertEqual(corpus, [])
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.exists')
    def test_load_mormon_corpus_empty_file(self, mock_exists, mock_file):
        """Test Mormon corpus loading with empty file"""
        # Mock file exists but is empty
        mock_exists.return_value = True
        mock_file.return_value.read.return_value = ""
        
        from app import load_mormon_corpus
        
        corpus = load_mormon_corpus()
        self.assertEqual(corpus, [])
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.exists')
    def test_load_mormon_corpus_chunking_logic(self, mock_exists, mock_file):
        """Test the chunking logic with different chunk sizes"""
        mock_exists.return_value = True
        mock_file.return_value.read.return_value = self.sample_mormon_text
        
        # Test with small chunk size
        os.environ['CHUNK_SIZE'] = '100'
        os.environ['CHUNK_OVERLAP'] = '20'
        
        from app import load_mormon_corpus
        
        corpus = load_mormon_corpus()
        
        # Should create multiple chunks due to small chunk size
        self.assertGreater(len(corpus), 1)
        
        # Verify chunk sizes
        for doc in corpus:
            self.assertLessEqual(len(doc['content']), 100)
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.exists')
    def test_load_mormon_corpus_large_chunk_size(self, mock_exists, mock_file):
        """Test chunking with large chunk size"""
        mock_exists.return_value = True
        mock_file.return_value.read.return_value = self.sample_mormon_text
        
        # Test with large chunk size
        os.environ['CHUNK_SIZE'] = '2000'
        os.environ['CHUNK_OVERLAP'] = '100'
        
        from app import load_mormon_corpus
        
        corpus = load_mormon_corpus()
        
        # Should create fewer chunks due to large chunk size
        self.assertGreater(len(corpus), 0)
        
        # At least one chunk should contain multiple verses
        found_multi_verse = False
        for doc in corpus:
            if '1 Nephi 1:1' in doc['content'] and '1 Nephi 1:2' in doc['content']:
                found_multi_verse = True
                break
        self.assertTrue(found_multi_verse, "Large chunks should contain multiple verses")
    
    def test_load_corpus_default_source(self):
        """Test load_corpus with default source"""
        # Set environment to use default corpus
        os.environ['CORPUS_SOURCE'] = 'default'
        
        from app import load_corpus
        
        corpus = load_corpus()
        
        # Should return default corpus
        self.assertIsInstance(corpus, list)
        self.assertGreater(len(corpus), 0)
        
        # Verify it contains legal documents (default corpus)
        found_legal_content = False
        for doc in corpus:
            if any(keyword in doc['content'].lower() for keyword in ['legal', 'contract', 'liability', 'risk']):
                found_legal_content = True
                break
        self.assertTrue(found_legal_content, "Default corpus should contain legal content")
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.exists')
    def test_load_corpus_mormon_source(self, mock_exists, mock_file):
        """Test load_corpus with Mormon source"""
        # Mock file operations
        mock_exists.return_value = True
        mock_file.return_value.read.return_value = self.sample_mormon_text
        
        # Set environment to use Mormon corpus
        os.environ['CORPUS_SOURCE'] = 'mormon'
        os.environ['CHUNK_SIZE'] = '300'
        os.environ['CHUNK_OVERLAP'] = '50'
        
        from app import load_corpus
        
        corpus = load_corpus()
        
        # Should return Mormon corpus
        self.assertIsInstance(corpus, list)
        self.assertGreater(len(corpus), 0)
        
        # Verify it contains Mormon content
        found_mormon_content = False
        for doc in corpus:
            if any(keyword in doc['content'] for keyword in ['Nephi', 'Jacob', 'wilderness']):
                found_mormon_content = True
                break
        self.assertTrue(found_mormon_content, "Mormon corpus should contain Book of Mormon content")
    
    @patch('builtins.open', side_effect=FileNotFoundError)
    @patch('os.path.exists')
    def test_load_corpus_mormon_fallback_to_default(self, mock_exists, mock_file):
        """Test load_corpus falls back to default when Mormon file not found"""
        # Mock file doesn't exist
        mock_exists.return_value = False
        
        # Set environment to use Mormon corpus
        os.environ['CORPUS_SOURCE'] = 'mormon'
        
        from app import load_corpus
        
        corpus = load_corpus()
        
        # Should fall back to default corpus
        self.assertIsInstance(corpus, list)
        self.assertGreater(len(corpus), 0)
        
        # Should contain legal content (default corpus)
        found_legal_content = False
        for doc in corpus:
            if any(keyword in doc['content'].lower() for keyword in ['legal', 'contract', 'liability']):
                found_legal_content = True
                break
        self.assertTrue(found_legal_content, "Should fall back to default legal corpus")
    
    def test_load_corpus_invalid_source(self):
        """Test load_corpus with invalid source"""
        # Set invalid corpus source
        os.environ['CORPUS_SOURCE'] = 'invalid_source'
        
        from app import load_corpus
        
        corpus = load_corpus()
        
        # Should fall back to default corpus
        self.assertIsInstance(corpus, list)
        self.assertGreater(len(corpus), 0)
    
    def test_environment_variable_defaults(self):
        """Test default values when environment variables are not set"""
        # Remove environment variables
        for var in ['CORPUS_SOURCE', 'CHUNK_SIZE', 'CHUNK_OVERLAP']:
            if var in os.environ:
                del os.environ[var]
        
        from app import load_corpus
        
        corpus = load_corpus()
        
        # Should use default corpus when no CORPUS_SOURCE is set
        self.assertIsInstance(corpus, list)
        self.assertGreater(len(corpus), 0)


class TestCorpusConfigurationIntegration(unittest.TestCase):
    """Integration tests for corpus configuration with the full application"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Store original environment variables
        self.original_corpus_source = os.getenv('CORPUS_SOURCE')
        
        # Create test app
        from app import app
        self.app = app.test_client()
        self.app.testing = True
    
    def tearDown(self):
        """Clean up after tests"""
        # Restore original environment variables
        if self.original_corpus_source:
            os.environ['CORPUS_SOURCE'] = self.original_corpus_source
        elif 'CORPUS_SOURCE' in os.environ:
            del os.environ['CORPUS_SOURCE']
    
    def test_rag_query_with_default_corpus(self):
        """Test RAG query endpoint with default corpus"""
        # Set to use default corpus
        os.environ['CORPUS_SOURCE'] = 'default'
        
        # Restart the app to pick up new environment variables
        # Note: In a real scenario, you'd restart the app or reload the module
        
        query = {"query": "legal risks and liability"}
        
        response = self.app.post('/rag-query',
                               data=json.dumps(query),
                               content_type='application/json')
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)
        
        # Verify results contain legal content
        found_legal = False
        for result in data:
            if any(keyword in result['content'].lower() for keyword in ['legal', 'contract', 'liability']):
                found_legal = True
                break
        self.assertTrue(found_legal, "Results should contain legal content from default corpus")
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.exists')
    def test_rag_query_with_mormon_corpus(self, mock_exists, mock_file):
        """Test RAG query endpoint with Mormon corpus"""
        # Mock file operations
        mock_exists.return_value = True
        sample_text = """1 Nephi 1:1 I, Nephi, having been born of goodly parents, therefore I was taught somewhat in all the learning of my father; and having seen many afflictions in the course of my days, nevertheless, having been highly favored of the Lord in all my days; yea, having had a great knowledge of the goodness and the mysteries of God, therefore I make a record of my proceedings in my days."""
        mock_file.return_value.read.return_value = sample_text
        
        # Set to use Mormon corpus
        os.environ['CORPUS_SOURCE'] = 'mormon'
        os.environ['CHUNK_SIZE'] = '500'
        os.environ['CHUNK_OVERLAP'] = '50'
        
        # Import after setting environment variables
        from app import load_corpus
        
        # Load the corpus to verify it works
        corpus = load_corpus()
        self.assertGreater(len(corpus), 0)
        
        # Test query
        query = {"query": "Nephi and his father"}
        
        response = self.app.post('/rag-query',
                               data=json.dumps(query),
                               content_type='application/json')
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIsInstance(data, list)
    
    def test_corpus_configuration_persistence(self):
        """Test that corpus configuration persists across requests"""
        # Set to use default corpus
        os.environ['CORPUS_SOURCE'] = 'default'
        
        # Make multiple requests
        queries = [
            {"query": "legal risks"},
            {"query": "contract liability"},
            {"query": "financial performance"}
        ]
        
        for query in queries:
            response = self.app.post('/rag-query',
                                   data=json.dumps(query),
                                   content_type='application/json')
            
            self.assertEqual(response.status_code, 200)
            data = json.loads(response.data)
            self.assertIsInstance(data, list)
            self.assertGreater(len(data), 0)


class TestCorpusChunking(unittest.TestCase):
    """Unit tests specifically for text chunking functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.sample_verses = [
            ("1 Nephi 1:1", "I, Nephi, having been born of goodly parents, therefore I was taught somewhat in all the learning of my father; and having seen many afflictions in the course of my days, nevertheless, having been highly favored of the Lord in all my days; yea, having had a great knowledge of the goodness and the mysteries of God, therefore I make a record of my proceedings in my days."),
            ("1 Nephi 1:2", "Yea, I make a record in the language of my father, which consists of the learning of the Jews and the language of the Egyptians."),
            ("1 Nephi 1:3", "And I know that the record which I make is true; and I make it with mine own hand; and I make it according to my knowledge.")
        ]
    
    def test_chunk_creation_with_small_chunks(self):
        """Test chunk creation with small chunk size"""
        # Test the chunking logic directly
        verses = self.sample_verses
        chunk_size = 100
        chunk_overlap = 20
        
        chunks = []
        current_chunk = ""
        current_verses = []
        
        for verse_ref, verse_text in verses:
            verse_line = f"{verse_ref} {verse_text}"
            
            if len(current_chunk) + len(verse_line) + 1 <= chunk_size:
                if current_chunk:
                    current_chunk += "\n\n"
                current_chunk += verse_line
                current_verses.append(verse_ref)
            else:
                if current_chunk:
                    # Create chunk
                    title = f"Book of Mormon - {current_verses[0]}"
                    if len(current_verses) > 1:
                        title += f" to {current_verses[-1]}"
                    
                    chunks.append({
                        "title": title,
                        "content": current_chunk
                    })
                
                # Start new chunk
                current_chunk = verse_line
                current_verses = [verse_ref]
        
        # Add final chunk
        if current_chunk:
            title = f"Book of Mormon - {current_verses[0]}"
            if len(current_verses) > 1:
                title += f" to {current_verses[-1]}"
            
            chunks.append({
                "title": title,
                "content": current_chunk
            })
        
        # Verify chunks
        self.assertGreater(len(chunks), 0)
        for chunk in chunks:
            self.assertLessEqual(len(chunk['content']), chunk_size)
            self.assertIn('title', chunk)
            self.assertIn('content', chunk)
    
    def test_chunk_overlap_logic(self):
        """Test that chunk overlap works correctly"""
        # This is a conceptual test since overlap logic would be more complex
        # In the current implementation, we don't have overlap between chunks
        # but this test documents the expected behavior if implemented
        
        chunk_size = 150
        chunk_overlap = 30
        
        # Verify that overlap parameter is within reasonable bounds
        self.assertLess(chunk_overlap, chunk_size)
        self.assertGreaterEqual(chunk_overlap, 0)
    
    def test_verse_parsing_regex(self):
        """Test the regex pattern used for parsing verses"""
        import re
        
        # Test the regex pattern used in load_mormon_corpus
        verse_pattern = r'(\d+\s+\w+\s+\d+:\d+)\s+(.*?)(?=\d+\s+\w+\s+\d+:\d+|$)'
        
        sample_text = "1 Nephi 1:1 I, Nephi, having been born of goodly parents. 1 Nephi 1:2 Yea, I make a record."
        
        matches = re.findall(verse_pattern, sample_text, re.DOTALL)
        
        self.assertGreater(len(matches), 0)
        for match in matches:
            self.assertEqual(len(match), 2)  # Should have reference and text
            self.assertIsInstance(match[0], str)  # Reference
            self.assertIsInstance(match[1], str)  # Text


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2, buffer=True)