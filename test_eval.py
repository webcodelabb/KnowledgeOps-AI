#!/usr/bin/env python3
"""
Test script to verify KnowledgeOps AI evaluation system
"""
import os
import sys
from pathlib import Path

def test_eval_imports():
    """Test that evaluation components can be imported"""
    print("📦 Testing Evaluation Imports...")
    
    try:
        # Test evaluation imports
        from scripts.eval import RAGEvaluator, EvaluationResult
        print("   ✅ Evaluation classes imported")
        
        # Test additional dependencies
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        from jinja2 import Template
        print("   ✅ Additional dependencies imported")
        
        print("✅ All evaluation imports successful")
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_eval_classes():
    """Test evaluation class definitions"""
    print("\n🎯 Testing Evaluation Classes...")
    
    try:
        from scripts.eval import RAGEvaluator, EvaluationResult
        
        # Test EvaluationResult dataclass
        result = EvaluationResult(
            question="What is machine learning?",
            gold_doc_id="doc_001",
            gold_text="Machine learning is a subset of AI",
            answer="Machine learning is a subset of artificial intelligence",
            faithfulness_score=0.85,
            processing_time=2.5
        )
        print("   ✅ EvaluationResult dataclass works")
        
        # Test RAGEvaluator initialization
        evaluator = RAGEvaluator(
            api_base_url="http://localhost:8000",
            openai_api_key="fake-key",
            org_id="test-org"
        )
        print("   ✅ RAGEvaluator initialization works")
        
        # Test class attributes
        checks = [
            ("RAGEvaluator methods", hasattr(RAGEvaluator, 'load_evaluation_data')),
            ("RAGEvaluator compute_relevance", hasattr(RAGEvaluator, 'compute_relevance_scores')),
            ("RAGEvaluator compute_faithfulness", hasattr(RAGEvaluator, 'compute_faithfulness_score')),
            ("RAGEvaluator generate_report", hasattr(RAGEvaluator, 'generate_html_report')),
        ]
        
        passed = 0
        for check_name, check_result in checks:
            if check_result:
                print(f"   ✅ {check_name}")
                passed += 1
            else:
                print(f"   ❌ {check_name}")
        
        if passed == len(checks):
            print("✅ All evaluation classes are properly defined")
            return True
        else:
            print(f"❌ {len(checks) - passed} class checks failed")
            return False
            
    except Exception as e:
        print(f"❌ Class test error: {e}")
        return False

def test_csv_loading():
    """Test CSV data loading functionality"""
    print("\n📁 Testing CSV Loading...")
    
    try:
        from scripts.eval import RAGEvaluator
        
        # Create evaluator
        evaluator = RAGEvaluator()
        
        # Check if sample CSV exists
        sample_csv = Path("scripts/sample_eval_data.csv")
        if not sample_csv.exists():
            print("   ⚠️  Sample CSV not found, creating test data")
            # Create minimal test CSV
            test_csv_content = """question,gold_doc_id,gold_text
"What is AI?",doc_001,"Artificial intelligence is the simulation of human intelligence"
"How does ML work?",doc_002,"Machine learning uses algorithms to learn from data"
"""
            with open("test_eval_data.csv", "w") as f:
                f.write(test_csv_content)
            sample_csv = Path("test_eval_data.csv")
        
        # Test loading
        data = evaluator.load_evaluation_data(str(sample_csv))
        print(f"   ✅ Loaded {len(data)} examples from CSV")
        
        # Validate structure
        if data:
            first_example = data[0]
            required_fields = ['question']
            optional_fields = ['gold_doc_id', 'gold_text']
            
            for field in required_fields:
                if field not in first_example:
                    print(f"   ❌ Missing required field: {field}")
                    return False
            
            has_optional = any(field in first_example for field in optional_fields)
            if not has_optional:
                print("   ❌ Missing optional fields (gold_doc_id or gold_text)")
                return False
            
            print("   ✅ CSV structure validation passed")
        
        # Clean up test file
        if sample_csv.name == "test_eval_data.csv" and sample_csv.exists():
            sample_csv.unlink()
        
        print("✅ CSV loading functionality works")
        return True
        
    except Exception as e:
        print(f"❌ CSV loading error: {e}")
        return False

def test_relevance_computation():
    """Test relevance score computation"""
    print("\n📊 Testing Relevance Computation...")
    
    try:
        from scripts.eval import RAGEvaluator
        
        evaluator = RAGEvaluator()
        
        # Test data
        question = "What is machine learning?"
        retrieved_chunks = [
            {"document_id": "doc_001", "text": "Machine learning is a subset of AI"},
            {"document_id": "doc_002", "text": "Neural networks are computing systems"},
            {"document_id": "doc_003", "text": "Deep learning uses multiple layers"}
        ]
        
        # Test document ID matching
        relevance_scores = evaluator.compute_relevance_scores(
            question, retrieved_chunks, gold_doc_id="doc_001"
        )
        
        if len(relevance_scores) == 3:
            print("   ✅ Relevance scores computed for document ID matching")
        else:
            print("   ❌ Incorrect number of relevance scores")
            return False
        
        # Test text similarity
        relevance_scores = evaluator.compute_relevance_scores(
            question, retrieved_chunks, gold_text="Machine learning is a subset of artificial intelligence"
        )
        
        if len(relevance_scores) == 3:
            print("   ✅ Relevance scores computed for text similarity")
        else:
            print("   ❌ Incorrect number of relevance scores")
            return False
        
        # Test relevance@k computation
        k_values = [1, 3, 5]
        relevance_at_k = evaluator.compute_relevance_at_k(relevance_scores, k_values)
        
        if all(k in relevance_at_k for k in k_values):
            print("   ✅ Relevance@k computation works")
        else:
            print("   ❌ Relevance@k computation failed")
            return False
        
        print("✅ Relevance computation functionality works")
        return True
        
    except Exception as e:
        print(f"❌ Relevance computation error: {e}")
        return False

def test_faithfulness_computation():
    """Test faithfulness score computation"""
    print("\n🎯 Testing Faithfulness Computation...")
    
    try:
        from scripts.eval import RAGEvaluator
        
        evaluator = RAGEvaluator()
        
        # Test data
        question = "What is machine learning?"
        answer = "Machine learning is a subset of artificial intelligence that enables computers to learn from data."
        retrieved_chunks = [
            {"text": "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions without being explicitly programmed."}
        ]
        
        # Test faithfulness computation (without OpenAI key, should return default)
        faithfulness_score = evaluator.compute_faithfulness_score(
            question, answer, retrieved_chunks
        )
        
        if 0.0 <= faithfulness_score <= 1.0:
            print(f"   ✅ Faithfulness score computed: {faithfulness_score}")
        else:
            print(f"   ❌ Invalid faithfulness score: {faithfulness_score}")
            return False
        
        print("✅ Faithfulness computation functionality works")
        return True
        
    except Exception as e:
        print(f"❌ Faithfulness computation error: {e}")
        return False

def test_html_report_generation():
    """Test HTML report generation"""
    print("\n📄 Testing HTML Report Generation...")
    
    try:
        from scripts.eval import RAGEvaluator, EvaluationResult
        
        evaluator = RAGEvaluator()
        
        # Create test results
        results = [
            EvaluationResult(
                question="What is machine learning?",
                gold_doc_id="doc_001",
                answer="Machine learning is a subset of AI",
                relevance_scores=[1.0, 0.3, 0.1],
                relevance_at_k={1: 1.0, 3: 0.47, 5: 0.28},
                faithfulness_score=0.85,
                processing_time=2.5
            ),
            EvaluationResult(
                question="How does neural networks work?",
                gold_doc_id="doc_002",
                answer="Neural networks are computing systems",
                relevance_scores=[0.8, 0.6, 0.2],
                relevance_at_k={1: 0.8, 3: 0.53, 5: 0.32},
                faithfulness_score=0.92,
                processing_time=3.1
            )
        ]
        
        # Test HTML generation
        output_path = "test_evaluation_report.html"
        evaluator.generate_html_report(results, output_path)
        
        if os.path.exists(output_path):
            print("   ✅ HTML report generated successfully")
            
            # Check file size
            file_size = os.path.getsize(output_path)
            if file_size > 1000:  # Should be substantial
                print(f"   ✅ HTML report size: {file_size} bytes")
            else:
                print(f"   ⚠️  HTML report seems small: {file_size} bytes")
            
            # Clean up
            os.remove(output_path)
            print("   ✅ Test file cleaned up")
        else:
            print("   ❌ HTML report not generated")
            return False
        
        print("✅ HTML report generation functionality works")
        return True
        
    except Exception as e:
        print(f"❌ HTML report generation error: {e}")
        return False

def test_api_integration():
    """Test API integration"""
    print("\n🌐 Testing API Integration...")
    
    try:
        from scripts.eval import RAGEvaluator
        
        # Test evaluator creation with API URL
        evaluator = RAGEvaluator(api_base_url="http://localhost:8000")
        
        # Test that the evaluator can be created
        if evaluator.api_base_url == "http://localhost:8000":
            print("   ✅ API URL configuration works")
        else:
            print("   ❌ API URL configuration failed")
            return False
        
        print("✅ API integration configuration works")
        return True
        
    except Exception as e:
        print(f"❌ API integration error: {e}")
        return False

def main():
    """Run all evaluation tests"""
    print("🚀 Testing KnowledgeOps AI Evaluation System")
    print("=" * 50)
    
    tests = [
        ("Evaluation Imports", test_eval_imports),
        ("Evaluation Classes", test_eval_classes),
        ("CSV Loading", test_csv_loading),
        ("Relevance Computation", test_relevance_computation),
        ("Faithfulness Computation", test_faithfulness_computation),
        ("HTML Report Generation", test_html_report_generation),
        ("API Integration", test_api_integration)
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
        print("🎉 All evaluation tests passed! RAG evaluation system is ready.")
        print("\n🚀 Next steps:")
        print("1. Ensure the KnowledgeOps AI API is running:")
        print("   python run.py")
        print("\n2. Run evaluation on sample data:")
        print("   python scripts/eval.py scripts/sample_eval_data.csv --openai-key YOUR_KEY")
        print("\n3. View the generated HTML report:")
        print("   open evaluation_report.html")
        print("\n📊 Features:")
        print("   - Relevance@k scoring (k=1,3,5,10)")
        print("   - Faithfulness scoring with LLM-as-judge")
        print("   - Comprehensive HTML reports with charts")
        print("   - Processing time tracking")
        print("   - Error handling and reporting")
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
