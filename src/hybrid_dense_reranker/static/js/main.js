// Hybrid Dense Reranker UI JavaScript

document.addEventListener('DOMContentLoaded', function() {
    const searchForm = document.getElementById('searchForm');
    const queryInput = document.getElementById('queryInput');
    const searchBtn = document.getElementById('searchBtn');
    const loadingSpinner = document.getElementById('loadingSpinner');
    const resultsContainer = document.getElementById('resultsContainer');
    const exampleQueries = document.querySelectorAll('.example-query');

    // Check if Mormon corpus examples should be shown
    fetch('/health')
        .then(response => response.json())
        .then(data => {
            // Show Mormon examples if corpus size > 10 (indicating Mormon corpus is loaded)
            if (data.corpus_size > 10) {
                const mormonExamples = document.getElementById('mormonExamples');
                if (mormonExamples) {
                    mormonExamples.style.display = 'block';
                }
            }
        })
        .catch(error => console.log('Could not determine corpus type'));

    // Handle example query clicks
    exampleQueries.forEach(button => {
        button.addEventListener('click', function() {
            const query = this.getAttribute('data-query');
            queryInput.value = query;
            queryInput.focus();
        });
    });

    // Handle search form submission
    if (searchForm) {
        searchForm.addEventListener('submit', function(e) {
            e.preventDefault();
            performSearch();
        });
    }

    function performSearch() {
        const query = queryInput.value.trim();
        
        if (!query) {
            showError('Please enter a search query');
            return;
        }

        // Show loading state
        setLoadingState(true);
        resultsContainer.innerHTML = '';

        // Perform search
        fetch('/rag-query', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ query: query })
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            return response.json();
        })
        .then(data => {
            setLoadingState(false);
            // Handle response format
            const results = data.results || data;
            const hasSequentialContent = data.has_sequential_content || false;
            displayResults(results, query, hasSequentialContent);
        })
        .catch(error => {
            setLoadingState(false);
            showError(`Search failed: ${error.message}`);
        });
    }

    function setLoadingState(loading) {
        if (loading) {
            searchBtn.disabled = true;
            searchBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Searching...';
            loadingSpinner.classList.remove('d-none');
        } else {
            searchBtn.disabled = false;
            searchBtn.innerHTML = '<i class="fas fa-search"></i> Search';
            loadingSpinner.classList.add('d-none');
        }
    }

    function displayResults(results, query, hasSequentialContent = false) {
        if (!results || results.length === 0) {
            resultsContainer.innerHTML = `
                <div class="alert alert-info" role="alert">
                    <i class="fas fa-info-circle me-2"></i>
                    No results found for "<strong>${escapeHtml(query)}</strong>"
                </div>
            `;
            return;
        }

        // Add search stats with sequential content indicator
        let statsHtml = `
            <div class="search-stats">
                <i class="fas fa-chart-bar me-1"></i>
                Found <strong>${results.length}</strong> relevant documents for "<strong>${escapeHtml(query)}</strong>"
        `;
        
        if (hasSequentialContent) {
            statsHtml += `
                <br><small class="text-info">
                    <i class="fas fa-book me-1"></i>
                    Sequential content shown in <strong>narrative order</strong>
                    <i class="fas fa-sort-numeric-up ms-1"></i>
                </small>
            `;
        }
        
        statsHtml += `</div>`;

        // Generate results HTML
        const resultsHtml = results.map((result, index) => {
            const combinedScore = result.combined_score || 0;
            const tfidfScore = result.tfidf_score || 0;
            const claudeScore = result.claude_score || 0;
            
            const scoreClass = getScoreClass(combinedScore);
            const rankIcon = getRankIcon(index);
            
            return `
                <div class="result-card">
                    <div class="card">
                        <div class="card-body">
                            <div class="d-flex justify-content-between align-items-start mb-2">
                                <div class="result-title">
                                    ${rankIcon} ${escapeHtml(result.title)}
                                    ${result.chunk_id ? `
                                        <small class="badge bg-secondary ms-2">
                                            Chunk ${result.chunk_id}
                                        </small>
                                    ` : ''}
                                </div>
                                <span class="badge ${scoreClass} score-badge">
                                    ${(combinedScore * 100).toFixed(1)}%
                                </span>
                            </div>
                            
                            <div class="result-content mb-3">
                                ${escapeHtml(result.content)}
                            </div>
                            
                            <div class="score-breakdown">
                                <div class="row mb-2">
                                    <div class="col-md-3">
                                        <small>
                                            <i class="fas fa-calculator me-1"></i>
                                            TF-IDF: <strong>${(tfidfScore * 100).toFixed(1)}%</strong>
                                        </small>
                                    </div>
                                    <div class="col-md-3">
                                        <small>
                                            <i class="fas fa-robot me-1"></i>
                                            Claude: <strong>${(claudeScore * 100).toFixed(1)}%</strong>
                                        </small>
                                    </div>
                                    <div class="col-md-3">
                                        <small>
                                            <i class="fas fa-link me-1"></i>
                                            Semantic: <strong>${((result.semantic_score || 0) * 100).toFixed(1)}%</strong>
                                        </small>
                                    </div>
                                    <div class="col-md-3">
                                        <small>
                                            <i class="fas fa-trophy me-1"></i>
                                            Combined: <strong>${(combinedScore * 100).toFixed(1)}%</strong>
                                        </small>
                                    </div>
                                </div>
                                ${result.explanation ? `
                                <div class="row">
                                    <div class="col-12">
                                        <small class="text-info">
                                            <i class="fas fa-info-circle me-1"></i>
                                            ${escapeHtml(result.explanation)}
                                        </small>
                                    </div>
                                </div>
                                ` : ''}
                            </div>
                        </div>
                    </div>
                </div>
            `;
        }).join('');

        resultsContainer.innerHTML = statsHtml + resultsHtml;
    }

    function showError(message) {
        resultsContainer.innerHTML = `
            <div class="alert alert-danger" role="alert">
                <i class="fas fa-exclamation-triangle me-2"></i>
                ${escapeHtml(message)}
            </div>
        `;
    }

    function getScoreClass(score) {
        if (score >= 0.8) return 'score-excellent';
        if (score >= 0.6) return 'score-good';
        if (score >= 0.4) return 'score-fair';
        return 'score-poor';
    }

    function getRankIcon(index) {
        const icons = ['🥇', '🥈', '🥉'];
        return icons[index] || `${index + 1}.`;
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    // Add keyboard shortcut for search (Ctrl+Enter or Cmd+Enter)
    queryInput.addEventListener('keydown', function(e) {
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            performSearch();
        }
    });

    // Auto-focus on search input
    if (queryInput) {
        queryInput.focus();
    }
});