#!/usr/bin/env python3
"""
Test script to verify KnowledgeOps AI retrieval system
"""
import os
import sys
from pathlib import Path

def test_retrieval_imports():
    """Test that retrieval components can be imported"""
    print("📦 Testing Retrieval Imports...")
    
    try:
        # Test retrieval imports
        from app.retrieval import AdvancedRetriever, RetrievalQAChain, RetrievalResult, QAResult
        print("   ✅ Retrieval classes imported")
        
        # Test LangChain imports
        from langchain.embeddings.openai import OpenAIEmbeddings
        print("   ✅ LangChain embeddings imported")
        
        # Test additional dependencies
        import rank_bm25
        import tiktoken
        import numpy as np
        print("   ✅ Additional dependencies imported")
        
        print("✅ All retrieval imports successful")
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_retrieval_classes():
    """Test retrieval class definitions"""
    print("\n🎯 Testing Retrieval Classes...")
    
    try:
        from app.retrieval import AdvancedRetriever, RetrievalQAChain, RetrievalResult, QAResult
        
        # Test RetrievalResult dataclass
        result = RetrievalResult(
            chunk_id="test-chunk",
            document_id="test-doc",
            text="Test text content",
            score=0.95,
            metadata={"test": "data"},
            source_title="Test Document",
            source_url="https://example.com"
        )
        print("   ✅ RetrievalResult dataclass works")
        
        # Test QAResult dataclass
        qa_result = QAResult(
            answer="Test answer",
            confidence=0.85,
            sources=[{"test": "source"}],
            total_tokens=100,
            processing_time=1.5
        )
        print("   ✅ QAResult dataclass works")
        
        # Test class attributes
        checks = [
            ("AdvancedRetriever class", hasattr(AdvancedRetriever, '__init__')),
            ("RetrievalQAChain class", hasattr(RetrievalQAChain, '__init__')),
            ("AdvancedRetriever methods", hasattr(AdvancedRetriever, 'retrieve')),
            ("RetrievalQAChain methods", hasattr(RetrievalQAChain, 'answer_question')),
        ]
        
        passed = 0
        for check_name, check_result in checks:
            if check_result:
                print(f"   ✅ {check_name}")
                passed += 1
            else:
                print(f"   ❌ {check_name}")
        
        if passed == len(checks):
            print("✅ All retrieval classes are properly defined")
            return True
        else:
            print(f"❌ {len(checks) - passed} class checks failed")
            return False
            
    except Exception as e:
        print(f"❌ Class test error: {e}")
        return False

def test_tokenizer():
    """Test token counting functionality"""
    print("\n🔢 Testing Tokenizer...")
    
    try:
        import tiktoken
        
        # Test tokenizer initialization
        tokenizer = tiktoken.get_encoding("cl100k_base")
        print("   ✅ Tokenizer initialized")
        
        # Test token counting
        test_text = "This is a test sentence with some words."
        token_count = len(tokenizer.encode(test_text))
        print(f"   ✅ Token counting works: {token_count} tokens")
        
        # Test with longer text
        long_text = "This is a longer test text with more words to count tokens properly."
        long_token_count = len(tokenizer.encode(long_text))
        print(f"   ✅ Long text token counting: {long_token_count} tokens")
        
        print("✅ Tokenizer functionality works correctly")
        return True
        
    except Exception as e:
        print(f"❌ Tokenizer error: {e}")
        return False

def test_bm25():
    """Test BM25 functionality"""
    print("\n📊 Testing BM25...")
    
    try:
        from rank_bm25 import BM25Okapi
        
        # Test documents
        documents = [
            "This is a test document about machine learning",
            "Another document about artificial intelligence",
            "A third document about data science and algorithms"
        ]
        
        # Tokenize documents
        tokenized_docs = [doc.lower().split() for doc in documents]
        
        # Create BM25 model
        bm25 = BM25Okapi(tokenized_docs)
        print("   ✅ BM25 model created")
        
        # Test scoring
        query = "machine learning algorithms"
        query_tokens = query.lower().split()
        scores = bm25.get_scores(query_tokens)
        
        print(f"   ✅ BM25 scoring works: {len(scores)} scores generated")
        print(f"   ✅ Score range: {min(scores):.3f} - {max(scores):.3f}")
        
        print("✅ BM25 functionality works correctly")
        return True
        
    except Exception as e:
        print(f"❌ BM25 error: {e}")
        return False

def test_api_integration():
    """Test API integration with retrieval"""
    print("\n🌐 Testing API Integration...")
    
    try:
        # Test that the API can import retrieval components
        from app.main import app
        print("   ✅ FastAPI app can import retrieval components")
        
        # Check that the new endpoints are available
        routes = [route.path for route in app.routes]
        required_routes = [
            "/query",
            "/retrieve"
        ]
        
        missing_routes = []
        for route in required_routes:
            if route not in routes:
                missing_routes.append(route)
        
        if not missing_routes:
            print("   ✅ All required retrieval API routes are available")
            return True
        else:
            print(f"   ❌ Missing API routes: {missing_routes}")
            return False
            
    except Exception as e:
        print(f"❌ API integration error: {e}")
        return False

def test_retrieval_configuration():
    """Test retrieval configuration options"""
    print("\n⚙️ Testing Retrieval Configuration...")
    
    try:
        from app.retrieval import AdvancedRetriever
        
        # Test configuration parameters
        config_checks = [
            ("top_k", 20),
            ("similarity_threshold", 0.7),
            ("dedup_threshold", 0.95),
            ("max_tokens", 4000),
            ("use_bm25_rerank", True),
            ("bm25_weight", 0.3)
        ]
        
        # Create a mock session (we won't actually use it)
        class MockSession:
            pass
        
        # Test retriever initialization
        retriever = AdvancedRetriever(
            session=MockSession(),
            org_id="test-org",
            top_k=10,
            similarity_threshold=0.8,
            dedup_threshold=0.9,
            max_tokens=2000,
            use_bm25_rerank=False,
            bm25_weight=0.5
        )
        
        print("   ✅ AdvancedRetriever initialization works")
        
        # Test tokenizer
        test_text = "Test text for token counting"
        token_count = retriever.count_tokens(test_text)
        print(f"   ✅ Token counting: {token_count} tokens")
        
        print("✅ Retrieval configuration works correctly")
        return True
        
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

def main():
    """Run all retrieval tests"""
    print("🚀 Testing KnowledgeOps AI Retrieval System")
    print("=" * 50)
    
    tests = [
        ("Retrieval Imports", test_retrieval_imports),
        ("Retrieval Classes", test_retrieval_classes),
        ("Tokenizer", test_tokenizer),
        ("BM25", test_bm25),
        ("API Integration", test_api_integration),
        ("Retrieval Configuration", test_retrieval_configuration)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        if test_func():
            passed += 1
        else:
            print(f"   ❌ {test_name} failed")
    
    print("\n" + "=" * 50)
    print(f"📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All retrieval tests passed! Advanced retrieval is ready.")
        print("\n🚀 Next steps:")
        print("1. Ensure you have documents processed and embedded")
        print("2. Test retrieval via API endpoints:")
        print("   - POST /retrieve - Document retrieval only")
        print("   - POST /query - Full RAG with answer generation")
        print("\n📚 Example usage:")
        print("   curl -X POST http://localhost:8000/retrieve \\")
        print("     -H 'Content-Type: application/json' \\")
        print("     -d '{\"query\": \"What is machine learning?\"}'")
        print("\n   curl -X POST http://localhost:8000/query \\")
        print("     -H 'Content-Type: application/json' \\")
        print("     -d '{\"query\": \"What is machine learning?\"}'")
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
