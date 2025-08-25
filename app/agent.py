"""
LangGraph agent for query reformulation and re-retrieval
"""
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import time
from langchain.graphs import StateGraph, END
from langchain.schema import BaseMessage, HumanMessage, AIMessage
import openai

from app.retrieval import RetrievalQAChain, AdvancedRetriever, RetrievalResult, QAResult
from app.logging import get_logger

logger = get_logger(__name__)


@dataclass
class AgentState:
    """State for the LangGraph agent"""
    query: str
    original_query: str
    org_id: str
    top_k: int
    confidence_threshold: float
    max_attempts: int
    current_attempt: int = 1
    
    # Retrieval results
    retrieval_results: Optional[List[RetrievalResult]] = None
    qa_result: Optional[QAResult] = None
    
    # Reformulation data
    reformulation_reason: Optional[str] = None
    keywords_extracted: Optional[List[str]] = None
    metadata_filters: Optional[Dict[str, Any]] = None
    
    # Comparison data
    attempts: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.attempts is None:
            self.attempts = []


class QueryReformulator:
    """Handles query reformulation strategies"""
    
    def __init__(self, openai_api_key: str, model_name: str = "gpt-3.5-turbo"):
        self.openai_api_key = openai_api_key
        self.model_name = model_name
        openai.api_key = openai_api_key
    
    def extract_keywords_from_chunks(self, chunks: List[RetrievalResult], max_keywords: int = 5) -> List[str]:
        """Extract important keywords from top chunks"""
        try:
            # Combine text from top chunks
            combined_text = " ".join([chunk.text for chunk in chunks[:3]])
            
            # Use OpenAI to extract keywords
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "Extract the most important keywords from the given text. Return only the keywords separated by commas, no explanations."},
                    {"role": "user", "content": f"Extract up to {max_keywords} important keywords from this text:\n\n{combined_text}"}
                ],
                max_tokens=100,
                temperature=0.1
            )
            
            keywords_text = response.choices[0].message.content.strip()
            keywords = [kw.strip() for kw in keywords_text.split(",") if kw.strip()]
            
            logger.info("Keywords extracted from chunks", 
                       keywords=keywords,
                       chunk_count=len(chunks))
            
            return keywords[:max_keywords]
            
        except Exception as e:
            logger.error("Failed to extract keywords", error=str(e))
            return []
    
    def extract_metadata_filters(self, chunks: List[RetrievalResult]) -> Dict[str, Any]:
        """Extract metadata filters from top chunks"""
        try:
            filters = {}
            
            # Collect metadata from chunks
            all_metadata = {}
            for chunk in chunks[:3]:
                if chunk.metadata:
                    for key, value in chunk.metadata.items():
                        if key not in all_metadata:
                            all_metadata[key] = []
                        all_metadata[key].append(value)
            
            # Create filters based on common metadata
            for key, values in all_metadata.items():
                if len(values) >= 2:  # At least 2 chunks share this metadata
                    # Use the most common value
                    from collections import Counter
                    counter = Counter(values)
                    most_common = counter.most_common(1)[0][0]
                    filters[key] = most_common
            
            logger.info("Metadata filters extracted", filters=filters)
            return filters
            
        except Exception as e:
            logger.error("Failed to extract metadata filters", error=str(e))
            return {}
    
    def reformulate_query(
        self, 
        original_query: str, 
        keywords: List[str], 
        metadata_filters: Dict[str, Any],
        confidence: float
    ) -> str:
        """Reformulate the query using extracted information"""
        try:
            # Build context for reformulation
            context_parts = []
            
            if keywords:
                context_parts.append(f"Important keywords: {', '.join(keywords)}")
            
            if metadata_filters:
                filter_str = ", ".join([f"{k}: {v}" for k, v in metadata_filters.items()])
                context_parts.append(f"Relevant metadata: {filter_str}")
            
            context = "\n".join(context_parts) if context_parts else "No additional context available."
            
            # Create reformulation prompt
            prompt = f"""The original query had low confidence ({confidence:.2f}). 
Please reformulate the query to be more specific and include relevant keywords or context.

Original query: {original_query}
Additional context: {context}

Reformulated query:"""
            
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a query reformulation expert. Create a more specific and detailed version of the original query that incorporates relevant keywords and context."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.3
            )
            
            reformulated_query = response.choices[0].message.content.strip()
            
            logger.info("Query reformulated", 
                       original_query=original_query,
                       reformulated_query=reformulated_query,
                       confidence=confidence)
            
            return reformulated_query
            
        except Exception as e:
            logger.error("Failed to reformulate query", error=str(e))
            # Fallback: add keywords to original query
            if keywords:
                return f"{original_query} {' '.join(keywords[:2])}"
            return original_query


class IntelligentQAAgent:
    """LangGraph agent for intelligent question answering with reformulation"""
    
    def __init__(
        self,
        session,
        org_id: str,
        openai_api_key: str,
        confidence_threshold: float = 0.7,
        max_attempts: int = 2,
        model_name: str = "gpt-3.5-turbo"
    ):
        self.session = session
        self.org_id = org_id
        self.openai_api_key = openai_api_key
        self.confidence_threshold = confidence_threshold
        self.max_attempts = max_attempts
        self.model_name = model_name
        
        # Initialize components
        self.qa_chain = RetrievalQAChain(
            session=session,
            org_id=org_id,
            openai_api_key=openai_api_key,
            model_name=model_name
        )
        self.reformulator = QueryReformulator(openai_api_key, model_name)
        
        # Build the graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("initial_retrieval", self._initial_retrieval)
        workflow.add_node("check_confidence", self._check_confidence)
        workflow.add_node("extract_context", self._extract_context)
        workflow.add_node("reformulate_query", self._reformulate_query)
        workflow.add_node("re_retrieval", self._re_retrieval)
        workflow.add_node("compare_results", self._compare_results)
        
        # Add edges
        workflow.add_edge("initial_retrieval", "check_confidence")
        workflow.add_edge("check_confidence", "extract_context")
        workflow.add_edge("extract_context", "reformulate_query")
        workflow.add_edge("reformulate_query", "re_retrieval")
        workflow.add_edge("re_retrieval", "compare_results")
        workflow.add_edge("compare_results", END)
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "check_confidence",
            self._should_reformulate,
            {
                "reformulate": "extract_context",
                "accept": END
            }
        )
        
        return workflow.compile()
    
    async def _initial_retrieval(self, state: AgentState) -> AgentState:
        """Perform initial retrieval and QA"""
        logger.info("Starting initial retrieval", 
                   query=state.query,
                   attempt=state.current_attempt)
        
        try:
            # Generate query embedding
            from langchain.embeddings.openai import OpenAIEmbeddings
            embeddings = OpenAIEmbeddings(
                openai_api_key=self.openai_api_key,
                model="text-embedding-ada-002"
            )
            query_embedding = embeddings.embed_query(state.query)
            
            # Get QA result
            qa_result = await self.qa_chain.answer_question(
                query=state.query,
                query_embedding=query_embedding,
                top_k=state.top_k
            )
            
            state.qa_result = qa_result
            
            # Store attempt data
            attempt_data = {
                "attempt": state.current_attempt,
                "query": state.query,
                "answer": qa_result.answer,
                "confidence": qa_result.confidence,
                "sources_count": len(qa_result.sources),
                "processing_time": qa_result.processing_time,
                "total_tokens": qa_result.total_tokens
            }
            state.attempts.append(attempt_data)
            
            logger.info("Initial retrieval completed", 
                       confidence=qa_result.confidence,
                       sources_count=len(qa_result.sources))
            
        except Exception as e:
            logger.error("Initial retrieval failed", error=str(e))
            # Create a failed result
            state.qa_result = QAResult(
                answer="Failed to process query",
                confidence=0.0,
                sources=[],
                total_tokens=0,
                processing_time=0.0
            )
        
        return state
    
    def _should_reformulate(self, state: AgentState) -> str:
        """Decide whether to reformulate the query"""
        if not state.qa_result:
            return "accept"
        
        should_reformulate = (
            state.qa_result.confidence < state.confidence_threshold and
            state.current_attempt < state.max_attempts
        )
        
        logger.info("Confidence check", 
                   confidence=state.qa_result.confidence,
                   threshold=state.confidence_threshold,
                   should_reformulate=should_reformulate)
        
        return "reformulate" if should_reformulate else "accept"
    
    async def _extract_context(self, state: AgentState) -> AgentState:
        """Extract context from current results for reformulation"""
        logger.info("Extracting context for reformulation")
        
        try:
            if state.qa_result and state.qa_result.sources:
                # Extract keywords from sources
                keywords = self.reformulator.extract_keywords_from_chunks(
                    [RetrievalResult(
                        chunk_id=source["chunk_id"],
                        document_id=source["document_id"],
                        text=source["text_preview"],
                        score=source["score"],
                        metadata={},
                        source_title=source["title"],
                        source_url=source["url"]
                    ) for source in state.qa_result.sources]
                )
                
                # Extract metadata filters
                metadata_filters = self.reformulator.extract_metadata_filters([
                    RetrievalResult(
                        chunk_id=source["chunk_id"],
                        document_id=source["document_id"],
                        text=source["text_preview"],
                        score=source["score"],
                        metadata={},
                        source_title=source["title"],
                        source_url=source["url"]
                    ) for source in state.qa_result.sources
                ])
                
                state.keywords_extracted = keywords
                state.metadata_filters = metadata_filters
                state.reformulation_reason = f"Low confidence: {state.qa_result.confidence:.2f}"
                
                logger.info("Context extracted", 
                           keywords=keywords,
                           metadata_filters=metadata_filters)
            
        except Exception as e:
            logger.error("Failed to extract context", error=str(e))
            state.keywords_extracted = []
            state.metadata_filters = {}
        
        return state
    
    async def _reformulate_query(self, state: AgentState) -> AgentState:
        """Reformulate the query using extracted context"""
        logger.info("Reformulating query")
        
        try:
            reformulated_query = self.reformulator.reformulate_query(
                original_query=state.original_query,
                keywords=state.keywords_extracted or [],
                metadata_filters=state.metadata_filters or {},
                confidence=state.qa_result.confidence if state.qa_result else 0.0
            )
            
            state.query = reformulated_query
            state.current_attempt += 1
            
            logger.info("Query reformulated", 
                       original=state.original_query,
                       reformulated=reformulated_query)
            
        except Exception as e:
            logger.error("Failed to reformulate query", error=str(e))
            # Keep original query as fallback
            state.query = state.original_query
        
        return state
    
    async def _re_retrieval(self, state: AgentState) -> AgentState:
        """Perform re-retrieval with reformulated query"""
        logger.info("Performing re-retrieval", 
                   query=state.query,
                   attempt=state.current_attempt)
        
        try:
            # Generate query embedding for reformulated query
            from langchain.embeddings.openai import OpenAIEmbeddings
            embeddings = OpenAIEmbeddings(
                openai_api_key=self.openai_api_key,
                model="text-embedding-ada-002"
            )
            query_embedding = embeddings.embed_query(state.query)
            
            # Get QA result for reformulated query
            qa_result = await self.qa_chain.answer_question(
                query=state.query,
                query_embedding=query_embedding,
                top_k=state.top_k
            )
            
            state.qa_result = qa_result
            
            # Store attempt data
            attempt_data = {
                "attempt": state.current_attempt,
                "query": state.query,
                "answer": qa_result.answer,
                "confidence": qa_result.confidence,
                "sources_count": len(qa_result.sources),
                "processing_time": qa_result.processing_time,
                "total_tokens": qa_result.total_tokens,
                "reformulation_reason": state.reformulation_reason,
                "keywords_used": state.keywords_extracted,
                "metadata_filters": state.metadata_filters
            }
            state.attempts.append(attempt_data)
            
            logger.info("Re-retrieval completed", 
                       confidence=qa_result.confidence,
                       sources_count=len(qa_result.sources))
            
        except Exception as e:
            logger.error("Re-retrieval failed", error=str(e))
            # Keep the previous result
            pass
        
        return state
    
    async def _compare_results(self, state: AgentState) -> AgentState:
        """Compare results from all attempts and select the best one"""
        logger.info("Comparing results from all attempts")
        
        if len(state.attempts) <= 1:
            return state
        
        # Find the attempt with highest confidence
        best_attempt = max(state.attempts, key=lambda x: x["confidence"])
        
        # Update the final result
        if state.qa_result and best_attempt["confidence"] > state.qa_result.confidence:
            # We need to reconstruct the QAResult from the best attempt
            # This is a simplified version - in practice you might want to store more data
            state.qa_result = QAResult(
                answer=best_attempt["answer"],
                confidence=best_attempt["confidence"],
                sources=[],  # We don't have the full sources here
                total_tokens=best_attempt["total_tokens"],
                processing_time=best_attempt["processing_time"]
            )
        
        logger.info("Results compared", 
                   attempts_count=len(state.attempts),
                   best_confidence=best_attempt["confidence"],
                   best_attempt=best_attempt["attempt"])
        
        return state
    
    async def answer_question(
        self,
        query: str,
        top_k: int = 5,
        confidence_threshold: float = None,
        max_attempts: int = None
    ) -> Dict[str, Any]:
        """Answer a question using the intelligent agent"""
        start_time = time.time()
        
        # Use instance defaults if not provided
        confidence_threshold = confidence_threshold or self.confidence_threshold
        max_attempts = max_attempts or self.max_attempts
        
        logger.info("Starting intelligent QA agent", 
                   query=query[:100],
                   confidence_threshold=confidence_threshold,
                   max_attempts=max_attempts)
        
        try:
            # Initialize state
            state = AgentState(
                query=query,
                original_query=query,
                org_id=self.org_id,
                top_k=top_k,
                confidence_threshold=confidence_threshold,
                max_attempts=max_attempts
            )
            
            # Run the graph
            final_state = await self.graph.ainvoke(state)
            
            # Prepare response
            response = {
                "query": query,
                "final_answer": final_state.qa_result.answer if final_state.qa_result else "No answer generated",
                "final_confidence": final_state.qa_result.confidence if final_state.qa_result else 0.0,
                "total_attempts": len(final_state.attempts),
                "attempts": final_state.attempts,
                "reformulation_used": len(final_state.attempts) > 1,
                "processing_time": time.time() - start_time
            }
            
            if final_state.reformulation_reason:
                response["reformulation_reason"] = final_state.reformulation_reason
            
            if final_state.keywords_extracted:
                response["keywords_extracted"] = final_state.keywords_extracted
            
            if final_state.metadata_filters:
                response["metadata_filters"] = final_state.metadata_filters
            
            logger.info("Intelligent QA agent completed", 
                       final_confidence=response["final_confidence"],
                       total_attempts=response["total_attempts"],
                       reformulation_used=response["reformulation_used"])
            
            return response
            
        except Exception as e:
            logger.error("Intelligent QA agent failed", error=str(e), exc_info=True)
            
            return {
                "query": query,
                "final_answer": "Failed to process query",
                "final_confidence": 0.0,
                "total_attempts": 0,
                "attempts": [],
                "reformulation_used": False,
                "processing_time": time.time() - start_time,
                "error": str(e)
            }
