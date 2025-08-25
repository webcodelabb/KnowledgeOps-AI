#!/usr/bin/env python3
"""
RAG Pipeline Evaluation Script
Evaluates relevance@k and faithfulness scores for KnowledgeOps AI
"""
import os
import sys
import csv
import json
import time
import asyncio
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np
import requests
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from jinja2 import Template
import openai

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from app.retrieval import RetrievalQAChain, AdvancedRetriever
from app.logging import get_logger

logger = get_logger(__name__)


@dataclass
class EvaluationResult:
    """Result from a single evaluation"""
    question: str
    gold_doc_id: Optional[str] = None
    gold_text: Optional[str] = None
    retrieved_chunks: List[Dict[str, Any]] = None
    answer: str = ""
    relevance_scores: List[float] = None
    relevance_at_k: Dict[int, float] = None
    faithfulness_score: float = 0.0
    processing_time: float = 0.0
    error: Optional[str] = None
    
    def __post_init__(self):
        if self.retrieved_chunks is None:
            self.retrieved_chunks = []
        if self.relevance_scores is None:
            self.relevance_scores = []
        if self.relevance_at_k is None:
            self.relevance_at_k = {}


class RAGEvaluator:
    """Evaluates RAG pipeline performance"""
    
    def __init__(
        self,
        api_base_url: str = "http://localhost:8000",
        openai_api_key: str = None,
        org_id: str = "default"
    ):
        self.api_base_url = api_base_url
        self.openai_api_key = openai_api_key
        self.org_id = org_id
        
        if openai_api_key:
            openai.api_key = openai_api_key
    
    def load_evaluation_data(self, csv_path: str) -> List[Dict[str, Any]]:
        """Load evaluation data from CSV"""
        logger.info(f"Loading evaluation data from {csv_path}")
        
        data = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Validate required fields
                if 'question' not in row:
                    raise ValueError("CSV must contain 'question' column")
                
                if 'gold_doc_id' not in row and 'gold_text' not in row:
                    raise ValueError("CSV must contain either 'gold_doc_id' or 'gold_text' column")
                
                data.append(row)
        
        logger.info(f"Loaded {len(data)} evaluation examples")
        return data
    
    def compute_relevance_scores(
        self, 
        question: str, 
        retrieved_chunks: List[Dict[str, Any]], 
        gold_doc_id: str = None,
        gold_text: str = None
    ) -> List[float]:
        """Compute relevance scores for retrieved chunks"""
        if not retrieved_chunks:
            return []
        
        relevance_scores = []
        
        for chunk in retrieved_chunks:
            score = 0.0
            
            # Method 1: Document ID matching
            if gold_doc_id and chunk.get('document_id') == gold_doc_id:
                score = 1.0
            # Method 2: Text similarity (if gold_text provided)
            elif gold_text:
                # Simple text overlap scoring
                chunk_text = chunk.get('text', '').lower()
                gold_text_lower = gold_text.lower()
                
                # Calculate overlap ratio
                chunk_words = set(chunk_text.split())
                gold_words = set(gold_text_lower.split())
                
                if gold_words:
                    overlap = len(chunk_words.intersection(gold_words))
                    score = overlap / len(gold_words)
            
            relevance_scores.append(score)
        
        return relevance_scores
    
    def compute_relevance_at_k(
        self, 
        relevance_scores: List[float], 
        k_values: List[int] = None
    ) -> Dict[int, float]:
        """Compute relevance@k for different k values"""
        if k_values is None:
            k_values = [1, 3, 5, 10]
        
        relevance_at_k = {}
        
        for k in k_values:
            if len(relevance_scores) >= k:
                # Check if any of the top-k chunks are relevant (score > 0.5)
                top_k_scores = relevance_scores[:k]
                relevant_count = sum(1 for score in top_k_scores if score > 0.5)
                relevance_at_k[k] = relevant_count / k
            else:
                # If we have fewer than k results, compute based on available
                relevant_count = sum(1 for score in relevance_scores if score > 0.5)
                relevance_at_k[k] = relevant_count / len(relevance_scores) if relevance_scores else 0.0
        
        return relevance_at_k
    
    def compute_faithfulness_score(
        self, 
        question: str, 
        answer: str, 
        retrieved_chunks: List[Dict[str, Any]]
    ) -> float:
        """Compute faithfulness score using LLM-as-judge"""
        if not answer or not retrieved_chunks:
            return 0.0
        
        try:
            # Prepare context from retrieved chunks
            context = "\n\n".join([
                f"Source {i+1}: {chunk.get('text', '')}"
                for i, chunk in enumerate(retrieved_chunks[:3])  # Use top 3 chunks
            ])
            
            # Create faithfulness evaluation prompt
            prompt = f"""You are an expert evaluator assessing the faithfulness of an AI-generated answer to a question based on provided source material.

Question: {question}

Source Material:
{context}

Generated Answer:
{answer}

Please evaluate how faithful the generated answer is to the source material on a scale of 0.0 to 1.0, where:
- 0.0: The answer contains information not supported by the source material or contradicts it
- 0.5: The answer is partially faithful but contains some unsupported claims
- 1.0: The answer is completely faithful and only contains information directly supported by the source material

Consider:
1. Does the answer only contain information present in the source material?
2. Are there any claims or statements not supported by the sources?
3. Does the answer accurately represent the information from the sources?

Return only a single number between 0.0 and 1.0:"""
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a precise evaluator. Return only the numerical score."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=10,
                temperature=0.0  # Deterministic scoring
            )
            
            # Extract score from response
            score_text = response.choices[0].message.content.strip()
            try:
                score = float(score_text)
                # Ensure score is between 0 and 1
                score = max(0.0, min(1.0, score))
                return score
            except ValueError:
                logger.warning(f"Could not parse faithfulness score: {score_text}")
                return 0.5  # Default score
        
        except Exception as e:
            logger.error(f"Error computing faithfulness score: {e}")
            return 0.5  # Default score
    
    async def evaluate_single_example(
        self, 
        example: Dict[str, Any]
    ) -> EvaluationResult:
        """Evaluate a single example"""
        start_time = time.time()
        
        try:
            question = example['question']
            gold_doc_id = example.get('gold_doc_id')
            gold_text = example.get('gold_text')
            
            logger.info(f"Evaluating question: {question[:50]}...")
            
            # Step 1: Retrieve documents
            retrieval_response = requests.post(
                f"{self.api_base_url}/retrieve",
                json={
                    "query": question,
                    "top_k": 10,
                    "org_id": self.org_id,
                    "use_bm25": True,
                    "max_tokens": 4000
                },
                timeout=30
            )
            
            if retrieval_response.status_code != 200:
                raise Exception(f"Retrieval failed: {retrieval_response.status_code}")
            
            retrieval_data = retrieval_response.json()
            retrieved_chunks = retrieval_data.get('results', [])
            
            # Step 2: Generate answer using intelligent query
            query_response = requests.post(
                f"{self.api_base_url}/query/intelligent",
                json={
                    "query": question,
                    "top_k": 5,
                    "confidence_threshold": 0.7,
                    "max_attempts": 2,
                    "org_id": self.org_id
                },
                timeout=60
            )
            
            if query_response.status_code != 200:
                raise Exception(f"Query failed: {query_response.status_code}")
            
            query_data = query_response.json()
            answer = query_data.get('final_answer', '')
            
            # Step 3: Compute relevance scores
            relevance_scores = self.compute_relevance_scores(
                question, retrieved_chunks, gold_doc_id, gold_text
            )
            
            # Step 4: Compute relevance@k
            relevance_at_k = self.compute_relevance_at_k(relevance_scores)
            
            # Step 5: Compute faithfulness score
            faithfulness_score = self.compute_faithfulness_score(
                question, answer, retrieved_chunks
            )
            
            processing_time = time.time() - start_time
            
            return EvaluationResult(
                question=question,
                gold_doc_id=gold_doc_id,
                gold_text=gold_text,
                retrieved_chunks=retrieved_chunks,
                answer=answer,
                relevance_scores=relevance_scores,
                relevance_at_k=relevance_at_k,
                faithfulness_score=faithfulness_score,
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Error evaluating example: {e}")
            
            return EvaluationResult(
                question=example.get('question', ''),
                gold_doc_id=example.get('gold_doc_id'),
                gold_text=example.get('gold_text'),
                error=str(e),
                processing_time=processing_time
            )
    
    async def evaluate_dataset(
        self, 
        csv_path: str, 
        max_examples: int = None
    ) -> List[EvaluationResult]:
        """Evaluate entire dataset"""
        logger.info(f"Starting evaluation of dataset: {csv_path}")
        
        # Load data
        data = self.load_evaluation_data(csv_path)
        
        if max_examples:
            data = data[:max_examples]
            logger.info(f"Limiting evaluation to {max_examples} examples")
        
        # Evaluate each example
        results = []
        for i, example in enumerate(data):
            logger.info(f"Evaluating example {i+1}/{len(data)}")
            result = await self.evaluate_single_example(example)
            results.append(result)
            
            # Small delay to avoid overwhelming the API
            await asyncio.sleep(0.5)
        
        logger.info(f"Completed evaluation of {len(results)} examples")
        return results
    
    def generate_html_report(
        self, 
        results: List[EvaluationResult], 
        output_path: str
    ) -> None:
        """Generate HTML report with charts"""
        logger.info(f"Generating HTML report: {output_path}")
        
        # Filter out failed evaluations
        successful_results = [r for r in results if r.error is None]
        failed_results = [r for r in results if r.error is not None]
        
        if not successful_results:
            logger.error("No successful evaluations to report")
            return
        
        # Compute aggregate metrics
        total_examples = len(results)
        success_rate = len(successful_results) / total_examples
        
        # Relevance@k metrics
        k_values = [1, 3, 5, 10]
        avg_relevance_at_k = {}
        for k in k_values:
            scores = [r.relevance_at_k.get(k, 0.0) for r in successful_results]
            avg_relevance_at_k[k] = np.mean(scores) if scores else 0.0
        
        # Faithfulness metrics
        faithfulness_scores = [r.faithfulness_score for r in successful_results]
        avg_faithfulness = np.mean(faithfulness_scores) if faithfulness_scores else 0.0
        
        # Processing time metrics
        processing_times = [r.processing_time for r in successful_results]
        avg_processing_time = np.mean(processing_times) if processing_times else 0.0
        
        # Create charts
        charts = self._create_charts(successful_results, k_values)
        
        # Generate HTML
        html_content = self._generate_html_template(
            total_examples=total_examples,
            success_rate=success_rate,
            avg_relevance_at_k=avg_relevance_at_k,
            avg_faithfulness=avg_faithfulness,
            avg_processing_time=avg_processing_time,
            charts=charts,
            results=successful_results,
            failed_results=failed_results
        )
        
        # Write HTML file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"HTML report generated: {output_path}")
    
    def _create_charts(self, results: List[EvaluationResult], k_values: List[int]) -> Dict[str, str]:
        """Create charts for the report"""
        charts = {}
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Chart 1: Relevance@k
        fig, ax = plt.subplots(figsize=(10, 6))
        k_scores = []
        for k in k_values:
            scores = [r.relevance_at_k.get(k, 0.0) for r in results]
            k_scores.append(np.mean(scores) if scores else 0.0)
        
        bars = ax.bar(k_values, k_scores, color='skyblue', alpha=0.7)
        ax.set_xlabel('k')
        ax.set_ylabel('Relevance@k')
        ax.set_title('Average Relevance@k Scores')
        ax.set_xticks(k_values)
        
        # Add value labels on bars
        for bar, score in zip(bars, k_scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{score:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save chart
        chart_path = 'relevance_at_k.png'
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Convert to base64 for HTML embedding
        import base64
        with open(chart_path, 'rb') as f:
            chart_data = base64.b64encode(f.read()).decode()
        charts['relevance_at_k'] = f"data:image/png;base64,{chart_data}"
        
        # Chart 2: Faithfulness Score Distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        faithfulness_scores = [r.faithfulness_score for r in results]
        
        ax.hist(faithfulness_scores, bins=20, color='lightgreen', alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(faithfulness_scores), color='red', linestyle='--', 
                  label=f'Mean: {np.mean(faithfulness_scores):.3f}')
        ax.set_xlabel('Faithfulness Score')
        ax.set_ylabel('Frequency')
        ax.set_title('Faithfulness Score Distribution')
        ax.legend()
        
        plt.tight_layout()
        
        # Save chart
        chart_path = 'faithfulness_distribution.png'
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Convert to base64
        with open(chart_path, 'rb') as f:
            chart_data = base64.b64encode(f.read()).decode()
        charts['faithfulness_distribution'] = f"data:image/png;base64,{chart_data}"
        
        # Chart 3: Processing Time Distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        processing_times = [r.processing_time for r in results]
        
        ax.hist(processing_times, bins=20, color='lightcoral', alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(processing_times), color='red', linestyle='--',
                  label=f'Mean: {np.mean(processing_times):.2f}s')
        ax.set_xlabel('Processing Time (seconds)')
        ax.set_ylabel('Frequency')
        ax.set_title('Processing Time Distribution')
        ax.legend()
        
        plt.tight_layout()
        
        # Save chart
        chart_path = 'processing_time_distribution.png'
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Convert to base64
        with open(chart_path, 'rb') as f:
            chart_data = base64.b64encode(f.read()).decode()
        charts['processing_time_distribution'] = f"data:image/png;base64,{chart_data}"
        
        # Clean up temporary files
        for chart_file in ['relevance_at_k.png', 'faithfulness_distribution.png', 'processing_time_distribution.png']:
            if os.path.exists(chart_file):
                os.remove(chart_file)
        
        return charts
    
    def _generate_html_template(
        self,
        total_examples: int,
        success_rate: float,
        avg_relevance_at_k: Dict[int, float],
        avg_faithfulness: float,
        avg_processing_time: float,
        charts: Dict[str, str],
        results: List[EvaluationResult],
        failed_results: List[EvaluationResult]
    ) -> str:
        """Generate HTML template"""
        
        html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>KnowledgeOps AI - RAG Evaluation Report</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }
        h1 {
            color: #2c3e50;
            text-align: center;
            margin-bottom: 30px;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }
        .summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        .metric-value {
            font-size: 2.5em;
            font-weight: bold;
            margin-bottom: 5px;
        }
        .metric-label {
            font-size: 1.1em;
            opacity: 0.9;
        }
        .charts-section {
            margin: 30px 0;
        }
        .chart-container {
            text-align: center;
            margin: 20px 0;
            padding: 20px;
            background-color: #f8f9fa;
            border-radius: 10px;
        }
        .chart-container img {
            max-width: 100%;
            height: auto;
            border-radius: 5px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        .results-section {
            margin: 30px 0;
        }
        .result-item {
            background-color: #f8f9fa;
            padding: 15px;
            margin: 10px 0;
            border-radius: 8px;
            border-left: 4px solid #3498db;
        }
        .question {
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 10px;
        }
        .answer {
            background-color: white;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
            border-left: 3px solid #27ae60;
        }
        .metrics-row {
            display: flex;
            gap: 20px;
            margin: 10px 0;
            flex-wrap: wrap;
        }
        .metric {
            background-color: #e8f4fd;
            padding: 5px 10px;
            border-radius: 15px;
            font-size: 0.9em;
        }
        .failed-item {
            background-color: #ffe6e6;
            padding: 15px;
            margin: 10px 0;
            border-radius: 8px;
            border-left: 4px solid #e74c3c;
        }
        .error-message {
            color: #e74c3c;
            font-family: monospace;
        }
        .timestamp {
            text-align: center;
            color: #7f8c8d;
            margin-top: 30px;
            font-style: italic;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 KnowledgeOps AI - RAG Evaluation Report</h1>
        
        <div class="summary-grid">
            <div class="metric-card">
                <div class="metric-value">{{ "%.1f"|format(success_rate * 100) }}%</div>
                <div class="metric-label">Success Rate</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{{ "%.3f"|format(avg_faithfulness) }}</div>
                <div class="metric-label">Avg Faithfulness</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{{ "%.2f"|format(avg_processing_time) }}s</div>
                <div class="metric-label">Avg Processing Time</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{{ total_examples }}</div>
                <div class="metric-label">Total Examples</div>
            </div>
        </div>
        
        <div class="charts-section">
            <h2>📊 Performance Charts</h2>
            
            <div class="chart-container">
                <h3>Relevance@k Scores</h3>
                <img src="{{ charts.relevance_at_k }}" alt="Relevance@k Chart">
            </div>
            
            <div class="chart-container">
                <h3>Faithfulness Score Distribution</h3>
                <img src="{{ charts.faithfulness_distribution }}" alt="Faithfulness Distribution">
            </div>
            
            <div class="chart-container">
                <h3>Processing Time Distribution</h3>
                <img src="{{ charts.processing_time_distribution }}" alt="Processing Time Distribution">
            </div>
        </div>
        
        <div class="results-section">
            <h2>📋 Detailed Results</h2>
            
            {% for result in results %}
            <div class="result-item">
                <div class="question">❓ {{ result.question }}</div>
                <div class="answer">💡 {{ result.answer }}</div>
                <div class="metrics-row">
                    <div class="metric">Faithfulness: {{ "%.3f"|format(result.faithfulness_score) }}</div>
                    <div class="metric">Processing Time: {{ "%.2f"|format(result.processing_time) }}s</div>
                    {% for k, score in result.relevance_at_k.items() %}
                    <div class="metric">R@{{ k }}: {{ "%.3f"|format(score) }}</div>
                    {% endfor %}
                </div>
            </div>
            {% endfor %}
        </div>
        
        {% if failed_results %}
        <div class="results-section">
            <h2>❌ Failed Evaluations</h2>
            
            {% for result in failed_results %}
            <div class="failed-item">
                <div class="question">❓ {{ result.question }}</div>
                <div class="error-message">Error: {{ result.error }}</div>
            </div>
            {% endfor %}
        </div>
        {% endif %}
        
        <div class="timestamp">
            Report generated on {{ timestamp }}
        </div>
    </div>
</body>
</html>
        """
        
        template = Template(html_template)
        return template.render(
            total_examples=total_examples,
            success_rate=success_rate,
            avg_relevance_at_k=avg_relevance_at_k,
            avg_faithfulness=avg_faithfulness,
            avg_processing_time=avg_processing_time,
            charts=charts,
            results=results,
            failed_results=failed_results,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )


async def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description="Evaluate RAG pipeline performance")
    parser.add_argument("csv_path", help="Path to CSV file with evaluation data")
    parser.add_argument("--output", "-o", default="evaluation_report.html", 
                       help="Output HTML report path")
    parser.add_argument("--api-url", default="http://localhost:8000",
                       help="KnowledgeOps AI API base URL")
    parser.add_argument("--openai-key", help="OpenAI API key for faithfulness scoring")
    parser.add_argument("--org-id", default="default", help="Organization ID")
    parser.add_argument("--max-examples", type=int, help="Maximum number of examples to evaluate")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.csv_path):
        print(f"Error: CSV file not found: {args.csv_path}")
        sys.exit(1)
    
    if not args.openai_key:
        print("Warning: No OpenAI API key provided. Faithfulness scoring will use default values.")
    
    # Create evaluator
    evaluator = RAGEvaluator(
        api_base_url=args.api_url,
        openai_api_key=args.openai_key,
        org_id=args.org_id
    )
    
    try:
        # Run evaluation
        print(f"🚀 Starting RAG evaluation...")
        print(f"📁 Input CSV: {args.csv_path}")
        print(f"🌐 API URL: {args.api_url}")
        print(f"📊 Max examples: {args.max_examples or 'all'}")
        
        results = await evaluator.evaluate_dataset(
            args.csv_path, 
            max_examples=args.max_examples
        )
        
        # Generate report
        print(f"📈 Generating HTML report: {args.output}")
        evaluator.generate_html_report(results, args.output)
        
        # Print summary
        successful_results = [r for r in results if r.error is None]
        failed_results = [r for r in results if r.error is not None]
        
        print(f"\n✅ Evaluation completed!")
        print(f"📊 Total examples: {len(results)}")
        print(f"✅ Successful: {len(successful_results)}")
        print(f"❌ Failed: {len(failed_results)}")
        print(f"📈 Success rate: {len(successful_results)/len(results)*100:.1f}%")
        
        if successful_results:
            avg_faithfulness = np.mean([r.faithfulness_score for r in successful_results])
            avg_processing_time = np.mean([r.processing_time for r in successful_results])
            
            print(f"🎯 Average faithfulness: {avg_faithfulness:.3f}")
            print(f"⏱️  Average processing time: {avg_processing_time:.2f}s")
            
            # Print relevance@k summary
            k_values = [1, 3, 5, 10]
            for k in k_values:
                scores = [r.relevance_at_k.get(k, 0.0) for r in successful_results]
                avg_score = np.mean(scores) if scores else 0.0
                print(f"📋 Relevance@{k}: {avg_score:.3f}")
        
        print(f"\n📄 HTML report saved: {args.output}")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
