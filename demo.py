#!/usr/bin/env python3
"""
Demo script for KnowledgeOps AI API
"""
import requests
import json
import time
from typing import Dict, Any

BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("🔍 Testing Health Check...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health: {data['status']}")
            print(f"   Version: {data['version']}")
            print(f"   Timestamp: {data['timestamp']}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_metrics():
    """Test metrics endpoint"""
    print("\n📊 Testing Metrics...")
    try:
        response = requests.get(f"{BASE_URL}/metrics")
        if response.status_code == 200:
            print("✅ Metrics endpoint accessible")
            # Print first few lines of metrics
            lines = response.text.split('\n')[:10]
            for line in lines:
                if line.strip():
                    print(f"   {line}")
            return True
        else:
            print(f"❌ Metrics failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Metrics error: {e}")
        return False

def test_ingest():
    """Test document ingestion"""
    print("\n📄 Testing Document Ingestion...")
    try:
        payload = {
            "file_url": "https://example.com/sample.pdf",
            "metadata": {
                "title": "Sample Document",
                "author": "Demo User",
                "category": "test"
            },
            "chunk_size": 800,
            "overlap": 100
        }
        
        response = requests.post(f"{BASE_URL}/ingest", json=payload)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Ingestion job created")
            print(f"   Job ID: {data['job_id']}")
            print(f"   Status: {data['status']}")
            print(f"   Message: {data['message']}")
            return data['job_id']
        else:
            print(f"❌ Ingestion failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return None
    except Exception as e:
        print(f"❌ Ingestion error: {e}")
        return None

def test_query():
    """Test document querying"""
    print("\n❓ Testing Document Query...")
    try:
        payload = {
            "query": "What is the main topic of the document?",
            "top_k": 5,
            "include_sources": True
        }
        
        response = requests.post(f"{BASE_URL}/query", json=payload)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Query processed successfully")
            print(f"   Query ID: {data['query_id']}")
            print(f"   Answer: {data['answer']}")
            print(f"   Confidence: {data['confidence']}")
            print(f"   Processing Time: {data['processing_time']:.3f}s")
            print(f"   Sources: {len(data['sources'])} found")
            return True
        else:
            print(f"❌ Query failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Query error: {e}")
        return False


def test_retrieve():
    """Test document retrieval"""
    print("\n🔍 Testing Document Retrieval...")
    try:
        payload = {
            "query": "What is the main topic of the document?",
            "top_k": 5,
            "org_id": "default",
            "use_bm25": True,
            "max_tokens": 4000
        }
        
        response = requests.post(f"{BASE_URL}/retrieve", json=payload)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Retrieval processed successfully")
            print(f"   Query: {data['query']}")
            print(f"   Total Results: {data['total_results']}")
            print(f"   Top Score: {data['top_score']:.3f}")
            print(f"   Results: {len(data['results'])} chunks found")
            return True
        else:
            print(f"❌ Retrieval failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Retrieval error: {e}")
        return False


def test_intelligent_query():
    """Test intelligent query with reformulation"""
    print("\n🧠 Testing Intelligent Query...")
    try:
        payload = {
            "query": "What is the main topic of the document?",
            "top_k": 5,
            "confidence_threshold": 0.7,
            "max_attempts": 2,
            "org_id": "default"
        }
        
        response = requests.post(f"{BASE_URL}/query/intelligent", json=payload)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Intelligent query processed successfully")
            print(f"   Query: {data['query']}")
            print(f"   Final Answer: {data['final_answer'][:100]}...")
            print(f"   Final Confidence: {data['final_confidence']:.3f}")
            print(f"   Total Attempts: {data['total_attempts']}")
            print(f"   Reformulation Used: {data['reformulation_used']}")
            print(f"   Processing Time: {data['processing_time']:.3f}s")
            
            if data.get('attempts'):
                print(f"   Attempt Details:")
                for attempt in data['attempts']:
                    print(f"     Attempt {attempt['attempt']}: {attempt['confidence']:.3f} confidence")
            
            return True
        else:
            print(f"❌ Intelligent query failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Intelligent query error: {e}")
        return False


def test_metrics():
    """Test metrics endpoint"""
    print("\n📊 Testing Metrics Endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/metrics")
        if response.status_code == 200:
            metrics_data = response.text
            print(f"✅ Metrics endpoint working")
            print(f"   Content-Type: {response.headers.get('content-type', 'unknown')}")
            print(f"   Data Size: {len(metrics_data)} characters")
            
            # Check for some expected metrics
            expected_metrics = [
                "http_requests_total",
                "query_requests_total",
                "query_tokens_in_total",
                "query_tokens_out_total"
            ]
            
            found_metrics = 0
            for metric in expected_metrics:
                if metric in metrics_data:
                    print(f"   ✅ Found metric: {metric}")
                    found_metrics += 1
            
            print(f"   📈 Found {found_metrics}/{len(expected_metrics)} expected metrics")
            return True
        else:
            print(f"❌ Metrics endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Metrics error: {e}")
        return False

def test_root():
    """Test root endpoint"""
    print("\n🏠 Testing Root Endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Root endpoint accessible")
            print(f"   Message: {data['message']}")
            print(f"   Version: {data['version']}")
            print(f"   Docs: {data['docs']}")
            return True
        else:
            print(f"❌ Root failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Root error: {e}")
        return False

def main():
    """Run all demo tests"""
    print("🚀 KnowledgeOps AI Demo")
    print("=" * 50)
    print(f"📍 Testing API at: {BASE_URL}")
    print("=" * 50)
    
    # Check if server is running
    try:
        requests.get(f"{BASE_URL}/health", timeout=5)
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to KnowledgeOps AI server")
        print("   Make sure the server is running:")
        print("   python run.py")
        return
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return
    
    # Run tests
    tests = [
        ("Health Check", test_health),
        ("Metrics", test_metrics),
        ("Root Endpoint", test_root),
        ("Document Ingestion", test_ingest),
        ("Document Query", test_query),
        ("Document Retrieval", test_retrieve),
        ("Intelligent Query", test_intelligent_query),
        ("Prometheus Metrics", test_metrics)
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Demo Results Summary")
    print("=" * 50)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! KnowledgeOps AI is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the server logs for details.")

if __name__ == "__main__":
    main()
