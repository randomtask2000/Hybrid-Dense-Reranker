# Integration Testing Documentation

This document describes the comprehensive integration tests for the Hybrid Dense Reranker application.

## Overview

The integration tests are designed to test the entire application without mocking, ensuring that all components work together correctly in a real environment. The tests cover:

- All helper functions
- All API endpoints
- Error handling
- Performance characteristics
- End-to-end workflows

## Test Files

### `test_integration.py`
Main integration test file containing two test classes:

#### `TestAppIntegration`
Comprehensive tests for all application functionality:

- **`test_get_embedding_function()`** - Tests the TF-IDF embedding generation
- **`test_analyze_with_claude_function()`** - Tests Claude API integration for relevance scoring
- **`test_analyze_with_claude_error_handling()`** - Tests error handling when Claude API fails
- **`test_rag_query_endpoint_*`** - Multiple tests for the `/rag-query` endpoint with various inputs
- **`test_faiss_index_integration()`** - Tests FAISS vector search functionality
- **`test_vectorizer_integration()`** - Tests TF-IDF vectorizer integration
- **`test_corpus_data_integrity()`** - Validates corpus data structure
- **`test_end_to_end_workflow()`** - Tests complete workflow from query to response

#### `TestAppPerformance`
Performance and stress tests:

- **`test_multiple_concurrent_requests()`** - Tests handling multiple simultaneous requests
- **`test_large_query_text()`** - Tests with very large query inputs

### `test_corpus_config.py`
Unit tests for configurable corpus loading functionality:

#### `TestCorpusConfiguration`
Core corpus configuration tests:

- **`test_load_default_corpus()`** - Tests loading the default legal corpus
- **`test_load_mormon_corpus()`** - Tests loading Mormon text corpus from data files
- **`test_chunk_text_basic()`** - Tests basic text chunking functionality
- **`test_chunk_text_with_overlap()`** - Tests chunking with configurable overlap
- **`test_chunk_text_edge_cases()`** - Tests chunking with edge cases (empty text, very small chunks)
- **`test_environment_variable_loading()`** - Tests corpus source selection via environment variables

#### `TestCorpusConfigurationIntegration`
Integration tests for corpus configuration:

- **`test_corpus_integration_with_vectorizer()`** - Tests corpus integration with TF-IDF vectorizer
- **`test_corpus_integration_with_faiss()`** - Tests corpus integration with FAISS indexing
- **`test_environment_variable_changes()`** - Tests dynamic corpus switching via environment changes

#### `TestCorpusChunking`
Specialized tests for text chunking functionality:

- **`test_chunk_size_configuration()`** - Tests configurable chunk sizes via CHUNK_SIZE environment variable
- **`test_chunk_overlap_configuration()`** - Tests configurable overlap via CHUNK_OVERLAP environment variable
- **`test_chunking_preserves_meaning()`** - Tests that chunking preserves semantic meaning at boundaries

### `test_corpus_integration.py`
End-to-end integration tests for corpus configuration:

#### `TestCorpusIntegrationWorkflow`
Complete workflow tests with different corpus sources:

- **`test_end_to_end_with_default_corpus()`** - Tests complete RAG workflow with default legal corpus
- **`test_end_to_end_with_mormon_corpus()`** - Tests complete RAG workflow with Mormon text corpus
- **`test_corpus_switching_workflow()`** - Tests switching between corpus sources during runtime
- **`test_vector_index_consistency()`** - Tests that vector indices remain consistent across corpus changes

#### `TestCorpusConfigurationEdgeCases`
Edge case and error handling tests:

- **`test_invalid_corpus_source()`** - Tests handling of invalid corpus source configurations
- **`test_missing_corpus_files()`** - Tests handling when corpus data files are missing
- **`test_corrupted_corpus_data()`** - Tests handling of corrupted or malformed corpus data
- **`test_empty_corpus_handling()`** - Tests behavior with empty corpus configurations

### `test_corpus_quick.py`
Quick validation tests for corpus functionality:

- **`test_default_corpus()`** - Quick smoke test for default corpus loading
- **`test_mormon_corpus()`** - Quick smoke test for Mormon corpus loading
- **`test_corpus_structure_validation()`** - Quick validation of corpus document structure
- **`test_environment_setup()`** - Quick validation of environment variable configuration

## Test Coverage

### Functions Tested
- [`get_embedding(text)`](app.py:30) - TF-IDF embedding generation
- [`analyze_with_claude(text, query)`](app.py:35) - Claude relevance scoring
- [`load_corpus()`](app.py) - Configurable corpus loading (default legal vs Mormon text)
- [`chunk_text(text, chunk_size, overlap)`](app.py) - Text chunking with configurable parameters
- [`setup_corpus_from_environment()`](app.py) - Environment-based corpus configuration

### Endpoints Tested
- [`POST /rag-query`](app.py:63) - Main RAG query endpoint
  - Valid requests with different query types
  - Invalid requests (missing fields, malformed JSON)
  - Error conditions
  - HTTP method validation

### Integration Points Tested
- FAISS index initialization and search
- TF-IDF vectorizer training and transformation
- Anthropic Claude API integration
- Flask request/response handling
- JSON serialization/deserialization
- Configurable corpus loading and switching
- Text chunking and overlap processing
- Environment variable-based configuration
- Multiple corpus source support (default legal, Mormon text)
- Vector index consistency across corpus changes

## Running the Tests

### Step 1: Validate Setup (Recommended)
```bash
# First, ensure you're in your virtual environment and validate setup
python validate_test_setup.py
```

### Step 2: Run Tests

#### Option 1: Using the Test Runner (Recommended)
```bash
python run_integration_tests.py
```

This interactive script will:
- Check your virtual environment and .env configuration
- Install test dependencies if needed
- Offer multiple test execution options
- Provide detailed results and troubleshooting tips

#### Option 2: Direct Execution
```bash
# Using unittest (no additional dependencies)
python test_integration.py

# Using pytest (requires test dependencies)
pip install -r test_requirements.txt
pytest test_integration.py -v

# With coverage report
pytest test_integration.py -v --cov=app --cov-report=html
```

#### Option 3: Corpus Tests
```bash
# Run comprehensive corpus tests (recommended for corpus functionality)
python run_corpus_tests.py

# Run specific corpus test files
python test_corpus_config.py          # Unit tests for corpus configuration
python test_corpus_integration.py     # Integration tests for corpus workflows
python test_corpus_quick.py          # Quick validation tests

# Using pytest for corpus tests
pytest test_corpus_config.py -v
pytest test_corpus_integration.py -v
pytest test_corpus_quick.py -v

# Run all corpus tests with coverage
pytest test_corpus*.py -v --cov=app --cov-report=html
```

### Option 4: Specific Test Classes
```bash
# Run only integration tests
pytest test_integration.py::TestAppIntegration -v

# Run only performance tests
pytest test_integration.py::TestAppPerformance -v

# Run specific corpus test classes
pytest test_corpus_config.py::TestCorpusConfiguration -v
pytest test_corpus_config.py::TestCorpusChunking -v
pytest test_corpus_integration.py::TestCorpusIntegrationWorkflow -v
pytest test_corpus_integration.py::TestCorpusConfigurationEdgeCases -v
```

### Option 5: Running Specific Test Methods
You can run individual test methods for focused testing or debugging:

```bash
# Run a specific test method using pytest
pytest test_corpus_integration.py::TestCorpusIntegrationWorkflow::test_tree_of_life_citations_and_meanings_real_data -v

# Run multiple specific test methods
pytest test_integration.py::TestAppIntegration::test_rag_query_endpoint_valid -v
pytest test_corpus_config.py::TestCorpusConfiguration::test_load_mormon_corpus -v

# Run specific test method with extra verbosity and output
pytest test_corpus_integration.py::TestCorpusIntegrationWorkflow::test_tree_of_life_citations_and_meanings_real_data -v -s

# Using unittest module directly (alternative approach)
python -m unittest test_corpus_integration.TestCorpusIntegrationWorkflow.test_tree_of_life_citations_and_meanings_real_data -v

# Run test method with coverage report
pytest test_corpus_integration.py::TestCorpusIntegrationWorkflow::test_tree_of_life_citations_and_meanings_real_data -v --cov=app
```

**Common Test Method Examples:**
```bash
# Test the tree of life analysis with real Mormon data
pytest test_corpus_integration.py::TestCorpusIntegrationWorkflow::test_tree_of_life_citations_and_meanings_real_data -v

# Test basic RAG query functionality
pytest test_integration.py::TestAppIntegration::test_rag_query_endpoint_valid -v

# Test Mormon corpus loading
pytest test_corpus_config.py::TestCorpusConfiguration::test_load_mormon_corpus -v

# Test Claude API integration
pytest test_integration.py::TestAppIntegration::test_analyze_with_claude_function -v

# Test corpus switching workflow
pytest test_corpus_integration.py::TestCorpusIntegrationWorkflow::test_corpus_switching -v
```

**Pro Tips for Running Specific Tests:**
- Use `-v` for verbose output to see detailed test information
- Use `-s` to see print statements and debug output during test execution
- Use `--tb=short` for shorter traceback on failures
- Use `--tb=long` for detailed traceback when debugging
- Combine with `--cov=app` to see code coverage for just that test

## Prerequisites

### Required Environment
1. **Virtual Environment**: Activate your existing virtual environment first
2. **Dependencies**: Ensure all dependencies from `requirements.txt` are installed in the venv
3. **Environment Variables**: Ensure `.env` file exists and is configured:
   ```
   ANTHROPIC_API_KEY=your-actual-api-key
   CORPUS_SOURCE=default  # or 'mormon' for Mormon text corpus
   CHUNK_SIZE=500         # optional: chunk size for text processing
   CHUNK_OVERLAP=50       # optional: overlap between chunks
   ```
4. **Test Dependencies** (optional): Install with `pip install -r test_requirements.txt`

### Setup Steps
1. Activate your virtual environment:
   ```bash
   # On Windows
   .\venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

2. Verify your `.env` file is configured (copy from `.env.example` if needed)

3. Validate setup:
   ```bash
   python validate_test_setup.py
   ```

### API Key Configuration
- The tests automatically load the Anthropic API key from your `.env` file
- Without a valid key, Claude-related tests will use fallback behavior
- Ensure your `.env` file contains: `ANTHROPIC_API_KEY=sk-ant-api03-...`

## Test Scenarios

### Valid Query Tests
Tests with realistic queries that should return meaningful results:
- Legal risk queries: "legal risks and liability"
- Security queries: "security and authentication"  
- Financial queries: "revenue and financial performance"

### Corpus-Specific Query Tests
Tests tailored to different corpus sources:

#### Default Legal Corpus
- Contract queries: "contract terms and conditions"
- Compliance queries: "regulatory compliance requirements"
- Risk assessment queries: "liability and legal risks"

#### Mormon Text Corpus
- Scripture queries: "Nephi teachings and revelations"
- Doctrinal queries: "faith and righteousness principles"
- Historical queries: "wilderness journey and trials"

### Corpus Configuration Tests
Tests for corpus switching and configuration:
- Environment variable-based corpus selection
- Dynamic corpus switching during runtime
- Chunk size and overlap configuration
- Corpus data integrity validation

### Edge Case Tests
Tests with unusual or problematic inputs:
- Empty queries
- Very long queries
- Missing request fields
- Invalid JSON payloads
- Wrong HTTP methods

### Error Handling Tests
Tests that verify graceful error handling:
- Claude API failures
- Invalid API keys
- Network timeouts
- Malformed requests
- Invalid corpus source configuration
- Missing corpus data files
- Corrupted corpus data
- Empty or malformed chunk configurations

## Expected Results

### Successful Test Run
When all tests pass, you should see:
- All function tests passing
- All endpoint tests returning 200 status codes
- Proper JSON response structures
- Valid score ranges (0.0 to 1.0 for Claude scores)
- Sorted results by combined score

### Test Metrics
- **Total Tests**: ~20 individual test methods (integration) + ~25 corpus test methods
- **Coverage**: Tests cover all major code paths in `app.py` including corpus functionality
- **Execution Time**: Typically 30-60 seconds (depends on Claude API response times)
- **Corpus Tests**: Additional 30-45 seconds for comprehensive corpus functionality testing

## Troubleshooting

### Common Issues

#### "ANTHROPIC_API_KEY not configured"
- Copy `.env.example` to `.env`
- Add your actual Anthropic API key
- Restart the test runner

#### "Missing required package" errors
- Run: `pip install -r requirements.txt`
- Ensure you're in the correct virtual environment

#### Claude API timeout/errors
- Check your API key validity
- Verify internet connection
- Check Anthropic service status

#### FAISS-related errors
- Ensure `faiss-cpu` is properly installed
- On some systems, you may need `faiss-gpu` instead

#### Corpus configuration errors
- Check `CORPUS_SOURCE` environment variable (should be 'default' or 'mormon')
- Verify corpus data files exist in `data/` directory:
  - `mormon13.txt` and `mormon13short.txt` for Mormon corpus
- Validate `CHUNK_SIZE` and `CHUNK_OVERLAP` are positive integers
- Ensure corpus files are readable and contain valid text content

#### "Missing corpus data" errors
- Check that `data/mormon13.txt` exists for Mormon corpus
- Verify file permissions allow reading
- Ensure corpus files are not empty or corrupted

### Test Failure Analysis

If tests fail, check:
1. **Environment Setup**: Verify all dependencies and environment variables
2. **API Connectivity**: Ensure Claude API is accessible
3. **Data Integrity**: Verify corpus data is properly loaded
4. **Index State**: Check if FAISS index is properly initialized
5. **Corpus Configuration**: Verify CORPUS_SOURCE, CHUNK_SIZE, and CHUNK_OVERLAP settings
6. **Data Files**: Ensure corpus data files exist and are accessible

## Extending the Tests

### Adding New Test Cases
To add new integration tests:

1. Add test methods to `TestAppIntegration` class
2. Follow naming convention: `test_*`
3. Use `self.app.post()` for endpoint testing
4. Include assertions for response status and structure

### Adding Performance Tests
To add performance tests:

1. Add methods to `TestAppPerformance` class
2. Focus on timing, throughput, and resource usage
3. Use realistic load scenarios

### Custom Test Configurations
Modify `pytest.ini` to:
- Add new test markers
- Change coverage settings
- Adjust output formats

## Integration with CI/CD

These tests are designed to run in automated environments:

```yaml
# Example GitHub Actions step
- name: Run Integration Tests
  run: |
    pip install -r requirements.txt
    pip install -r test_requirements.txt
    pytest test_integration.py -v --cov=app
  env:
    ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
```

## Security Considerations

- API keys are loaded from environment variables
- No sensitive data is hardcoded in tests
- Test data uses safe, non-sensitive content
- Tests verify input validation and error handling