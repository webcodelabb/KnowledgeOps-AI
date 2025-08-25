"""
Advanced retrieval system for KnowledgeOps AI
"""
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_, text
from sqlalchemy.dialects.postgresql import VECTOR
from rank_bm25 import BM25Okapi
import tiktoken

from app.models_db import Document, Chunk
from app.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RetrievalResult:
    """Result from retrieval with metadata"""
    chunk_id: str
    document_id: str
    text: str
    score: float
    metadata: Dict[str, Any]
    source_title: Optional[str] = None
    source_url: Optional[str] = None


@dataclass
class QAResult:
    """Result from question answering"""
    answer: str
    confidence: float
    sources: List[Dict[str, Any]]
    total_tokens: int
    processing_time: float


class AdvancedRetriever:
    """Advanced retriever with vector similarity, BM25, and deduplication"""
    
    def __init__(
        self,
        session: AsyncSession,
        org_id: str,
        top_k: int = 20,
        similarity_threshold: float = 0.7,
        dedup_threshold: float = 0.95,
        max_tokens: int = 4000,
        use_bm25_rerank: bool = True,
        bm25_weight: float = 0.3
    ):
        self.session = session
        self.org_id = org_id
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.dedup_threshold = dedup_threshold
        self.max_tokens = max_tokens
        self.use_bm25_rerank = use_bm25_rerank
        self.bm25_weight = bm25_weight
        self.tokenizer = tiktoken.get_encoding("cl100k_base")  # GPT-4 tokenizer
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken"""
        return len(self.tokenizer.encode(text))
    
    async def vector_search(
        self, 
        query_embedding: List[float], 
        top_k: int = None
    ) -> List[RetrievalResult]:
        """Perform vector similarity search using pgvector"""
        if top_k is None:
            top_k = self.top_k
            
        try:
            # Build vector similarity query
            query = select(
                Chunk.id,
                Chunk.doc_id,
                Chunk.text,
                Chunk.metadata,
                Document.source,
                Document.author,
                func.cosine_similarity(Chunk.embedding, query_embedding).label('similarity_score')
            ).join(
                Document, Chunk.doc_id == Document.id
            ).where(
                and_(
                    Document.org_id == self.org_id,
                    Chunk.embedding.isnot(None)
                )
            ).order_by(
                text('similarity_score DESC')
            ).limit(top_k * 2)  # Get more results for reranking
            
            result = await self.session.execute(query)
            rows = result.fetchall()
            
            # Convert to RetrievalResult objects
            results = []
            for row in rows:
                if row.similarity_score >= self.similarity_threshold:
                    results.append(RetrievalResult(
                        chunk_id=str(row.id),
                        document_id=str(row.doc_id),
                        text=row.text,
                        score=float(row.similarity_score),
                        metadata=row.metadata or {},
                        source_title=row.metadata.get('title') if row.metadata else None,
                        source_url=row.source
                    ))
            
            logger.info("Vector search completed", 
                       query_length=len(query_embedding),
                       results_count=len(results),
                       top_score=max([r.score for r in results]) if results else 0)
            
            return results
            
        except Exception as e:
            logger.error("Vector search failed", error=str(e))
            raise
    
    def bm25_rerank(
        self, 
        query: str, 
        results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """Rerank results using BM25"""
        if not results or not self.use_bm25_rerank:
            return results
            
        try:
            # Prepare documents for BM25
            documents = [result.text for result in results]
            tokenized_docs = [doc.lower().split() for doc in documents]
            
            # Create BM25 model
            bm25 = BM25Okapi(tokenized_docs)
            
            # Get BM25 scores
            query_tokens = query.lower().split()
            bm25_scores = bm25.get_scores(query_tokens)
            
            # Combine vector and BM25 scores
            for i, result in enumerate(results):
                vector_score = result.score
                bm25_score = bm25_scores[i]
                
                # Normalize BM25 score to 0-1 range
                normalized_bm25 = min(bm25_score / 10.0, 1.0)  # Rough normalization
                
                # Weighted combination
                combined_score = (1 - self.bm25_weight) * vector_score + self.bm25_weight * normalized_bm25
                result.score = combined_score
            
            # Sort by combined score
            results.sort(key=lambda x: x.score, reverse=True)
            
            logger.info("BM25 reranking completed", 
                       query=query,
                       results_count=len(results),
                       top_score=max([r.score for r in results]) if results else 0)
            
            return results
            
        except Exception as e:
            logger.error("BM25 reranking failed", error=str(e))
            return results  # Return original results if BM25 fails
    
    def deduplicate_chunks(
        self, 
        results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """Remove near-duplicate chunks based on text similarity"""
        if not results:
            return results
            
        try:
            # Simple deduplication using text similarity
            unique_results = []
            seen_hashes = set()
            
            for result in results:
                # Create hash of normalized text
                normalized_text = result.text.lower().strip()
                text_hash = hashlib.md5(normalized_text.encode()).hexdigest()
                
                # Check if we've seen a very similar chunk
                is_duplicate = False
                for seen_hash in seen_hashes:
                    # Simple similarity check (can be improved with more sophisticated methods)
                    if self._text_similarity(normalized_text, seen_hash) > self.dedup_threshold:
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    unique_results.append(result)
                    seen_hashes.add(text_hash)
            
            logger.info("Deduplication completed", 
                       original_count=len(results),
                       unique_count=len(unique_results))
            
            return unique_results
            
        except Exception as e:
            logger.error("Deduplication failed", error=str(e))
            return results
    
    def _text_similarity(self, text1: str, text2_hash: str) -> float:
        """Calculate text similarity (simplified version)"""
        # This is a simplified similarity check
        # In production, you might want to use more sophisticated methods
        # like Jaccard similarity, cosine similarity on TF-IDF, etc.
        return 0.0  # Placeholder
    
    def enforce_token_budget(
        self, 
        results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """Enforce token budget by selecting chunks within limit"""
        if not results:
            return results
            
        try:
            selected_results = []
            current_tokens = 0
            
            for result in results:
                chunk_tokens = self.count_tokens(result.text)
                
                if current_tokens + chunk_tokens <= self.max_tokens:
                    selected_results.append(result)
                    current_tokens += chunk_tokens
                else:
                    # Try to fit a partial chunk if it's not too long
                    if chunk_tokens <= self.max_tokens * 0.5:  # Allow chunks up to 50% of budget
                        selected_results.append(result)
                        current_tokens += chunk_tokens
                    break
            
            logger.info("Token budget enforcement completed", 
                       max_tokens=self.max_tokens,
                       used_tokens=current_tokens,
                       selected_chunks=len(selected_results))
            
            return selected_results
            
        except Exception as e:
            logger.error("Token budget enforcement failed", error=str(e))
            return results[:5]  # Return first 5 results as fallback
    
    async def retrieve(
        self, 
        query: str, 
        query_embedding: List[float],
        top_k: int = None
    ) -> List[RetrievalResult]:
        """Main retrieval method with all enhancements"""
        try:
            logger.info("Starting retrieval", 
                       query=query[:100],  # Log first 100 chars
                       org_id=self.org_id,
                       top_k=top_k or self.top_k)
            
            # Step 1: Vector similarity search
            results = await self.vector_search(query_embedding, top_k)
            
            if not results:
                logger.warning("No results from vector search")
                return []
            
            # Step 2: BM25 reranking
            if self.use_bm25_rerank:
                results = self.bm25_rerank(query, results)
            
            # Step 3: Deduplication
            results = self.deduplicate_chunks(results)
            
            # Step 4: Enforce token budget
            results = self.enforce_token_budget(results)
            
            # Step 5: Return top-k results
            final_results = results[:top_k or self.top_k]
            
            logger.info("Retrieval completed successfully", 
                       final_count=len(final_results),
                       total_tokens=sum(self.count_tokens(r.text) for r in final_results))
            
            return final_results
            
        except Exception as e:
            logger.error("Retrieval failed", error=str(e), exc_info=True)
            raise


class RetrievalQAChain:
    """Retrieval-Augmented Generation (RAG) chain"""
    
    def __init__(
        self,
        session: AsyncSession,
        org_id: str,
        openai_api_key: str,
        model_name: str = "gpt-3.5-turbo",
        max_tokens: int = 4000,
        temperature: float = 0.1
    ):
        self.session = session
        self.org_id = org_id
        self.openai_api_key = openai_api_key
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.retriever = AdvancedRetriever(
            session=session,
            org_id=org_id,
            max_tokens=max_tokens
        )
    
    def _create_prompt(self, query: str, contexts: List[RetrievalResult]) -> str:
        """Create prompt for the LLM"""
        context_text = "\n\n".join([
            f"Context {i+1} (Score: {ctx.score:.3f}):\n{ctx.text}"
            for i, ctx in enumerate(contexts)
        ])
        
        prompt = f"""You are a helpful AI assistant. Answer the question based on the provided context. If the answer cannot be found in the context, say "I cannot answer this question based on the provided context."

Context:
{context_text}

Question: {query}

Answer:"""
        
        return prompt
    
    def _calculate_confidence(
        self, 
        query: str, 
        answer: str, 
        contexts: List[RetrievalResult]
    ) -> float:
        """Calculate confidence score based on various factors"""
        try:
            # Factor 1: Average relevance score of contexts
            avg_score = np.mean([ctx.score for ctx in contexts]) if contexts else 0.0
            
            # Factor 2: Number of relevant contexts
            context_count_score = min(len(contexts) / 5.0, 1.0)  # Normalize to 0-1
            
            # Factor 3: Answer length (longer answers might be more confident)
            answer_length_score = min(len(answer) / 200.0, 1.0)  # Normalize to 0-1
            
            # Factor 4: Check if answer contains uncertainty indicators
            uncertainty_indicators = [
                "i don't know", "cannot answer", "not provided", 
                "unclear", "uncertain", "maybe", "possibly"
            ]
            uncertainty_score = 1.0
            answer_lower = answer.lower()
            for indicator in uncertainty_indicators:
                if indicator in answer_lower:
                    uncertainty_score *= 0.7
            
            # Combine factors
            confidence = (
                0.4 * avg_score +
                0.3 * context_count_score +
                0.2 * answer_length_score +
                0.1 * uncertainty_score
            )
            
            return min(max(confidence, 0.0), 1.0)
            
        except Exception as e:
            logger.error("Confidence calculation failed", error=str(e))
            return 0.5  # Default confidence
    
    async def answer_question(
        self, 
        query: str, 
        query_embedding: List[float],
        top_k: int = 5
    ) -> QAResult:
        """Answer a question using RAG"""
        import time
        start_time = time.time()
        
        try:
            logger.info("Starting QA chain", query=query[:100], org_id=self.org_id)
            
            # Step 1: Retrieve relevant contexts
            contexts = await self.retriever.retrieve(query, query_embedding, top_k)
            
            if not contexts:
                return QAResult(
                    answer="I cannot answer this question as no relevant information was found.",
                    confidence=0.0,
                    sources=[],
                    total_tokens=0,
                    processing_time=time.time() - start_time
                )
            
            # Step 2: Create prompt
            prompt = self._create_prompt(query, contexts)
            
            # Step 3: Generate answer using OpenAI
            import openai
            openai.api_key = self.openai_api_key
            
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant that answers questions based on provided context."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=self.temperature
            )
            
            answer = response.choices[0].message.content.strip()
            
            # Step 4: Calculate confidence
            confidence = self._calculate_confidence(query, answer, contexts)
            
            # Step 5: Prepare sources
            sources = []
            for ctx in contexts:
                sources.append({
                    "chunk_id": ctx.chunk_id,
                    "document_id": ctx.document_id,
                    "title": ctx.source_title or "Unknown",
                    "url": ctx.source_url or "Unknown",
                    "score": ctx.score,
                    "text_preview": ctx.text[:200] + "..." if len(ctx.text) > 200 else ctx.text
                })
            
            # Step 6: Calculate total tokens
            total_tokens = sum(self.retriever.count_tokens(ctx.text) for ctx in contexts)
            total_tokens += self.retriever.count_tokens(answer)
            
            processing_time = time.time() - start_time
            
            logger.info("QA chain completed", 
                       query=query[:100],
                       answer_length=len(answer),
                       confidence=confidence,
                       sources_count=len(sources),
                       processing_time=processing_time)
            
            return QAResult(
                answer=answer,
                confidence=confidence,
                sources=sources,
                total_tokens=total_tokens,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error("QA chain failed", error=str(e), exc_info=True)
            processing_time = time.time() - start_time
            
            return QAResult(
                answer="I encountered an error while processing your question. Please try again.",
                confidence=0.0,
                sources=[],
                total_tokens=0,
                processing_time=processing_time
            )
