# Hybrid Dense Reranker - Pure Anthropic

## Configuration

This application uses Anthropic's Claude for intelligent reranking and TF-IDF for embeddings, providing a pure Anthropic-based solution.

## Solution

Since Anthropic doesn't provide an embeddings API (they focus on text generation with Claude), this application combines:
- **TF-IDF embeddings** for initial document retrieval
- **Anthropic Claude** for intelligent relevance scoring and reranking

### Features:

1. **TF-IDF Embeddings**:
   - Fast, local embedding generation using scikit-learn
   - No external API calls for embeddings
   - Efficient for document retrieval

2. **Anthropic Claude Integration**:
   - Uses Claude-3-Sonnet for intelligent relevance scoring
   - Analyzes query-document relevance with natural language understanding
   - Combines TF-IDF and Claude scores for optimal results

3. **Hybrid Scoring**:
   - 30% TF-IDF similarity score
   - 70% Claude relevance score
   - Results sorted by combined score

## Setup Instructions

### Quick Setup (Recommended)

For a quick automated setup, run the provided setup script:

**On macOS/Linux:**
```bash
# Make the script executable (if not already)
chmod +x setup_venv.sh

# Run the setup script
./setup_venv.sh
```

**On Windows:**
```cmd
# Run the batch script
setup_venv.bat
```

This script will:
- Create a virtual environment
- Install all dependencies
- Create a `.env` file from the template
- Provide next steps

### Manual Setup

### 1. Clone and Navigate to Project

```bash
git clone git@github.com:randomtask2000/Hybrid-Dense-Reranker.git
cd Hybrid-Dense-Reranker
```

### 2. Create Virtual Environment

Create a Python virtual environment to isolate project dependencies:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate
```

You should see `(venv)` in your terminal prompt when the virtual environment is active.

### 3. Install Dependencies

With the virtual environment activated, install the required packages:

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Copy the example environment file and configure your API key:

```bash
cp .env.example .env
```

Edit the `.env` file and replace `your-anthropic-api-key-here` with your actual Anthropic API key:

```bash
# Get your API key from: https://console.anthropic.com/
ANTHROPIC_API_KEY=your-actual-api-key-here

# Corpus Configuration (optional)
CORPUS_SOURCE=default  # Options: 'default' or 'mormon'
CHUNK_SIZE=1000        # Maximum characters per chunk (for Mormon corpus)
CHUNK_OVERLAP=100      # Character overlap between chunks
```

**Alternative method** - Set environment variable directly:

```bash
export ANTHROPIC_API_KEY='your-anthropic-api-key-here'
```

To make it permanent, add it to your shell profile:

```bash
echo 'export ANTHROPIC_API_KEY="your-anthropic-api-key-here"' >> ~/.zshrc
source ~/.zshrc
```

### 5. Test the Setup

Run the test script to verify everything works:

```bash
python test_embedding.py
```

### 6. Run the Application

Make sure your virtual environment is activated, then run:

```bash
# Ensure virtual environment is activated
source venv/bin/activate  # On macOS/Linux
# venv\Scripts\activate   # On Windows

# Run the application
python app.py
```

### 7. Virtual Environment Management

**Deactivating the virtual environment:**
```bash
deactivate
```

**Reactivating the virtual environment:**
```bash
# Navigate to project directory
cd /path/to/Hybrid-Dense-Reranker

# Activate virtual environment
source venv/bin/activate  # On macOS/Linux
# venv\Scripts\activate   # On Windows
```

**Installing additional packages:**
```bash
# With virtual environment activated
pip install package-name

# Update requirements.txt if needed
pip freeze > requirements.txt
```

## Corpus Configuration

The application supports configurable corpus sources, allowing you to switch between different document collections:

### Available Corpus Sources

1. **Default Corpus** (`CORPUS_SOURCE=default`):
   - Contains sample legal documents
   - Includes contracts, compliance memos, and risk assessments
   - Ready to use out of the box

2. **Mormon Corpus** (`CORPUS_SOURCE=mormon`):
   - Loads text from `data/mormon13short.txt`
   - Automatically chunks the Book of Mormon text into manageable pieces
   - Configurable chunk size and overlap

### Configuration Options

Set these environment variables in your `.env` file:

```bash
# Corpus source selection
CORPUS_SOURCE=default  # Options: 'default' or 'mormon'

# Text chunking configuration (applies to Mormon corpus)
CHUNK_SIZE=1000        # Maximum characters per chunk
CHUNK_OVERLAP=100      # Characters to overlap between chunks
```

### Using the Mormon Corpus

To use the Mormon corpus:

1. Ensure `data/mormon13short.txt` exists in your project
2. Set `CORPUS_SOURCE=mormon` in your `.env` file
3. Configure chunk size and overlap as needed
4. Restart the application

The application will automatically:
- Parse verse references (e.g., "1 Nephi 1:1")
- Create chunks based on your size settings
- Maintain context with configurable overlap
- Fall back to default corpus if the file is not found

### Example Queries by Corpus

**Default Corpus (Legal Documents):**
```bash
curl -X POST http://localhost:5000/rag-query \
  -H "Content-Type: application/json" \
  -d '{"query": "contract liability and legal risks"}'
```

**Mormon Corpus:**
```bash
curl -X POST http://localhost:5000/rag-query \
  -H "Content-Type: application/json" \
  -d '{"query": "Nephi and his teachings about faith"}'
```

## Usage

The application provides a RAG (Retrieval-Augmented Generation) endpoint:

```bash
curl -X POST http://localhost:5000/rag-query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the security risks?"}'
```

## API Response Format

The application returns enhanced results with multiple scoring methods:
```json
[
  {
    "title": "Security Memo",
    "content": "Ensure all employees use 2FA to reduce unauthorized access risks.",
    "tfidf_score": 0.85,
    "claude_score": 0.92,
    "combined_score": 0.899
  }
]
```

## How It Works

1. **Initial Retrieval**: TF-IDF embeddings find potentially relevant documents
2. **Claude Analysis**: Each retrieved document is analyzed by Claude for relevance
3. **Hybrid Scoring**: Combines TF-IDF similarity with Claude's understanding
4. **Intelligent Ranking**: Results sorted by combined score for optimal relevance

## Benefits

- **Pure Anthropic**: Uses only Anthropic's Claude for AI processing
- **Cost Effective**: TF-IDF embeddings are free and fast
- **Intelligent**: Claude provides nuanced relevance understanding
- **Scalable**: Can handle large document collections efficiently

## Testing

For comprehensive testing instructions, including integration tests and performance tests, see [TESTING.md](TESTING.md).

### Quick Test Run
```bash
# Validate setup
python validate_test_setup.py

# Run all tests
python run_integration_tests.py
```

### Corpus Configuration Testing

Test the new corpus configuration functionality:

```bash
# Quick validation of corpus configuration
python test_corpus_quick.py

# Comprehensive corpus configuration tests
python run_corpus_tests.py

# Unit tests for corpus functionality
python test_corpus_config.py

# Integration tests for corpus workflow
python test_corpus_integration.py
```

### Test Different Corpus Sources

```bash
# Test with default corpus
CORPUS_SOURCE=default python test_corpus_quick.py

# Test with Mormon corpus (if file exists)
CORPUS_SOURCE=mormon CHUNK_SIZE=500 python test_corpus_quick.py
```