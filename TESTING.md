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

## Test Coverage

### Functions Tested
- [`get_embedding(text)`](app.py:30) - TF-IDF embedding generation
- [`analyze_with_claude(text, query)`](app.py:35) - Claude relevance scoring

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

### Option 3: Specific Test Classes
```bash
# Run only integration tests
pytest test_integration.py::TestAppIntegration -v

# Run only performance tests
pytest test_integration.py::TestAppPerformance -v
```

## Prerequisites

### Required Environment
1. **Virtual Environment**: Activate your existing virtual environment first
2. **Dependencies**: Ensure all dependencies from `requirements.txt` are installed in the venv
3. **Environment Variables**: Ensure `.env` file exists and is configured:
   ```
   ANTHROPIC_API_KEY=your-actual-api-key
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

## Expected Results

### Successful Test Run
When all tests pass, you should see:
- All function tests passing
- All endpoint tests returning 200 status codes
- Proper JSON response structures
- Valid score ranges (0.0 to 1.0 for Claude scores)
- Sorted results by combined score

### Test Metrics
- **Total Tests**: ~20 individual test methods
- **Coverage**: Tests cover all major code paths in `app.py`
- **Execution Time**: Typically 30-60 seconds (depends on Claude API response times)

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

### Test Failure Analysis

If tests fail, check:
1. **Environment Setup**: Verify all dependencies and environment variables
2. **API Connectivity**: Ensure Claude API is accessible
3. **Data Integrity**: Verify corpus data is properly loaded
4. **Index State**: Check if FAISS index is properly initialized

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