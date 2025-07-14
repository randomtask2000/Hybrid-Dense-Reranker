#!/usr/bin/env python3
"""
Test script to verify the Anthropic integration works correctly.
"""

import os
import anthropic
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

load_dotenv()


def test_anthropic_connection():
    """Test the Anthropic Claude API connection."""
    
    # Check if API key is set
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ ANTHROPIC_API_KEY environment variable is not set")
        print("Please set it with: export ANTHROPIC_API_KEY='your-api-key-here'")
        return False
    
    print("✅ ANTHROPIC_API_KEY is set")
    
    try:
        print("🔄 Testing Anthropic Claude API...")
        client = anthropic.Anthropic(api_key=api_key)
        
        message = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=50,
            messages=[
                {
                    "role": "user",
                    "content": "Say 'Hello, I am Claude!' to test the connection."
                }
            ]
        )
        
        response_text = message.content[0].text
        print(f"✅ Claude responded: {response_text}")
        return True
        
    except Exception as e:
        print(f"❌ Error testing Anthropic API: {e}")
        return False


def test_tfidf_embeddings():
    """Test the TF-IDF embedding functionality."""
    
    try:
        print("🔄 Testing TF-IDF embeddings...")
        
        # Sample texts
        texts = [
            "The contract exposes the organization to liability due to lack of indemnification clauses.",
            "Ensure all employees use 2FA to reduce unauthorized access risks.",
            "Revenue grew by 15% but legal expenses increased due to ongoing litigation."
        ]
        
        # Initialize vectorizer
        vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        vectorizer.fit(texts)
        
        # Test embedding generation
        test_text = "This is a test sentence for embedding."
        embedding = vectorizer.transform([test_text]).toarray()[0]
        
        print(f"✅ TF-IDF embedding generated! Dimension: {len(embedding)}")
        print(f"   Sample values: {embedding[:5]}")
        return True
        
    except Exception as e:
        print(f"❌ Error testing TF-IDF embeddings: {e}")
        return False


def test_claude_scoring():
    """Test Claude's relevance scoring functionality."""
    
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ Cannot test Claude scoring without ANTHROPIC_API_KEY")
        return False
    
    try:
        print("🔄 Testing Claude relevance scoring...")
        client = anthropic.Anthropic(api_key=api_key)
        
        query = "What are the security risks?"
        text = "Ensure all employees use 2FA to reduce unauthorized access risks."
        
        message = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=10,
            messages=[
                {
                    "role": "user",
                    "content": f"Rate the relevance of this text to the query on a scale of 0-1. Query: '{query}' Text: '{text}' Return only a number between 0 and 1."
                }
            ]
        )
        
        score = float(message.content[0].text.strip())
        print(f"✅ Claude relevance score: {score}")
        
        if 0 <= score <= 1:
            print("✅ Score is within valid range")
            return True
        else:
            print("❌ Score is outside valid range (0-1)")
            return False
            
    except Exception as e:
        print(f"❌ Error testing Claude scoring: {e}")
        return False


if __name__ == "__main__":
    print("=== Testing Anthropic Integration ===\n")
    
    tests_passed = 0
    total_tests = 3
    
    if test_anthropic_connection():
        tests_passed += 1
    
    print()
    if test_tfidf_embeddings():
        tests_passed += 1
    
    print()
    if test_claude_scoring():
        tests_passed += 1
    
    print(f"\n=== Test Results: {tests_passed}/{total_tests} passed ===")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! Your Anthropic integration is working.")
        print("You can run: python app.py")
    else:
        print("❌ Some tests failed. Please check your configuration.")