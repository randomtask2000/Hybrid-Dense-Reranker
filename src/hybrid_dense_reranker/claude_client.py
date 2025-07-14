"""Enhanced Claude API client for advanced relevance scoring."""
import anthropic
import re
from .config import ANTHROPIC_API_KEY


class ClaudeClient:
    """Enhanced client for sophisticated relevance analysis with Claude."""
    
    def __init__(self):
        """Initialize the Claude client."""
        self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    
    def analyze_relevance(self, text, query):
        """Advanced relevance analysis with detailed reasoning."""
        try:
            # Enhanced prompt with better instructions and reasoning
            prompt = f"""You are an expert information retrieval system analyzing document relevance.

TASK: Determine how relevant this document is to the user's query.

QUERY: "{query}"

DOCUMENT: "{text}"

ANALYSIS FRAMEWORK:
1. Semantic Match: How well does the document content match the query's meaning?
2. Topic Relevance: Does the document discuss the same topic/domain as the query?
3. Information Value: How useful would this document be for someone with this query?
4. Specificity: How directly does the document address the query's specific aspects?

SCORING CRITERIA:
- 0.9-1.0: Highly relevant, directly answers or addresses the query
- 0.7-0.8: Very relevant, contains substantial related information
- 0.5-0.6: Moderately relevant, some useful information but not directly on topic
- 0.3-0.4: Minimally relevant, tangentially related information
- 0.0-0.2: Not relevant, unrelated to the query

IMPORTANT: Consider synonyms, related concepts, and contextual meaning. A document about "legal risks" is highly relevant to "liability issues".

Provide your analysis in this exact format:
RELEVANCE_SCORE: [number between 0.0 and 1.0]
REASONING: [brief explanation of the score]"""

            message = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",  # Use latest model
                max_tokens=250,
                temperature=0.1,  # Low temperature for consistent scoring
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            
            response = message.content[0].text.strip()
            
            # Extract score using regex
            score_match = re.search(r'RELEVANCE_SCORE:\s*([0-9]*\.?[0-9]+)', response)
            if score_match:
                score = float(score_match.group(1))
                return max(0, min(1, score))  # Ensure score is between 0 and 1
            else:
                # Fallback: try to find any number in the response
                numbers = re.findall(r'([0-9]*\.?[0-9]+)', response)
                if numbers:
                    score = float(numbers[0])
                    if score <= 1.0:
                        return max(0, score)
                
                print(f"Could not parse Claude response: {response}")
                return 0.5
                
        except Exception as e:
            print(f"Claude analysis error: {e}")
            return 0.5  # Default score if Claude fails
    
    def analyze_relevance_with_context(self, text, query, context_docs=None):
        """Analyze relevance with additional context from other documents."""
        if not context_docs:
            return self.analyze_relevance(text, query)
        
        try:
            # Build context from other documents
            context_summary = "\n".join([f"- {doc['title']}: {doc['content'][:100]}..." 
                                       for doc in context_docs[:2]])
            
            prompt = f"""You are analyzing document relevance in context of a document collection.

QUERY: "{query}"

TARGET DOCUMENT: "{text}"

CONTEXT (other documents in collection):
{context_summary}

Given this context, how relevant is the TARGET DOCUMENT to the query?
Consider:
1. Does it provide unique information not covered by other documents?
2. Does it complement or enhance the information from other documents?
3. Is it directly relevant to the query's intent?

Return only a relevance score between 0.0 and 1.0."""

            message = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=150,
                temperature=0.1,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            
            response = message.content[0].text.strip()
            
            # Extract score
            numbers = re.findall(r'([0-9]*\.?[0-9]+)', response)
            if numbers:
                score = float(numbers[0])
                return max(0, min(1, score))
            
            return 0.5
            
        except Exception as e:
            print(f"Context analysis error: {e}")
            return self.analyze_relevance(text, query)