from __future__ import annotations

import json
import logging
import os
import re
import uuid
from dataclasses import dataclass
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple
from typing import TypedDict

import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import TextLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.documents import Document
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.graph import END, StateGraph
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from src.core.config import Settings
from src.services.vector_store import VectorStoreService

logger = logging.getLogger(__name__)


class Step(BaseModel):
    sub_question: str = Field(description="Specific question to answer during this step.")
    justification: str = Field(description="Why this step is needed.")
    tool: Literal["search_10k", "search_web"] = Field(description="Tool to use for this step.")
    keywords: List[str] = Field(description="Important keywords for search.")
    document_section: Optional[str] = Field(
        default=None,
        description="Likely document section to target when using the 10-K search tool.",
    )


class Plan(BaseModel):
    steps: List[Step] = Field(description="Ordered list of reasoning steps.")


class RetrievalDecision(BaseModel):
    strategy: Literal["vector_search", "keyword_search", "hybrid_search"]
    justification: str


class Decision(BaseModel):
    next_action: Literal["CONTINUE_PLAN", "FINISH"]
    justification: str


class PastStep(TypedDict):
    step_index: int
    sub_question: str
    retrieved_docs: List[Document]
    summary: str


class RAGState(TypedDict, total=False):
    original_question: str
    plan: Plan
    past_steps: List[PastStep]
    current_step_index: int
    retrieved_docs: List[Document]
    reranked_docs: List[Document]
    synthesized_context: str
    final_answer: str


@dataclass
class IngestionResult:
    chunks_ingested: int
    sections_found: int


class DeepThinkingRAGService:
    """Encapsulates the Deep Thinking RAG pipeline and ingestion workflows."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.vector_store_service = VectorStoreService(settings)
        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
        )
        self._llm = ChatOllama(
            model=settings.ollama_model_id,
            base_url=settings.ollama_base_url,
            temperature=0,
        )
        self._structured_llm = ChatOllama(
            model=settings.ollama_model_id,
            base_url=settings.ollama_base_url,
            temperature=0,
        )
        self._reranker = CrossEncoder(settings.reranker_model)
        self._web_search_tool = self._initialise_web_search()
        self._documents_cache: List[Document] = []
        self._doc_ids: List[str] = []
        self._doc_map: Dict[str, Document] = {}
        self._bm25: Optional[BM25Okapi] = None

        self._planner_chain = self._build_planner_chain()
        self._query_rewriter_chain = self._build_query_rewriter_chain()
        self._retrieval_supervisor_chain = self._build_retrieval_supervisor_chain()
        self._distiller_chain = self._build_distiller_chain()
        self._reflection_chain = self._build_reflection_chain()
        self._policy_chain = self._build_policy_chain()
        self._final_answer_chain = self._build_final_answer_chain()

        self._load_existing_documents()
        self._graph = self._build_state_graph()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def ingest_document(
        self,
        *,
        file_path: Optional[str] = None,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> IngestionResult:
        """Ingest a document into the PGVector store and rebuild keyword indices."""

        text = content or self._load_text_from_file(file_path)
        if not text:
            raise ValueError("Either 'content' or a valid 'file_path' must be provided for ingestion.")

        source = metadata.get("source") if metadata else None
        if not source:
            source = file_path or "user_provided_content"

        documents, sections_found = self._prepare_documents(text, source, metadata)
        self.vector_store_service.add_documents(documents)

        self._documents_cache.extend(documents)
        self._rebuild_keyword_index()

        return IngestionResult(chunks_ingested=len(documents), sections_found=sections_found)

    def query(self, question: str) -> str:
        if not question:
            raise ValueError("Query text cannot be empty.")

        initial_state: RAGState = {"original_question": question}
        result_state = self._graph.invoke(initial_state)
        answer = result_state.get("final_answer")
        if not answer:
            logger.warning("RAG graph did not produce a final answer. Returning empty string.")
            return ""
        return answer

    # ------------------------------------------------------------------
    # Ingestion helpers
    # ------------------------------------------------------------------
    def _load_text_from_file(self, file_path: Optional[str]) -> str:
        if not file_path:
            return ""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        loader = TextLoader(file_path, encoding="utf-8")
        documents = loader.load()
        if not documents:
            raise ValueError(f"No content could be loaded from {file_path}")
        return "\n\n".join(doc.page_content for doc in documents)

    def _prepare_documents(
        self, text: str, source: str, metadata: Optional[Dict[str, str]]
    ) -> Tuple[List[Document], int]:
        sections = self._extract_sections(text)
        documents: List[Document] = []
        base_metadata = metadata.copy() if metadata else {}
        for section_title, section_text in sections:
            chunks = self._text_splitter.split_text(section_text)
            for chunk in chunks:
                doc_id = str(uuid.uuid4())
                enriched_metadata = {
                    **base_metadata,
                    "section": section_title,
                    "source": source,
                    "id": doc_id,
                }
                documents.append(Document(page_content=chunk, metadata=enriched_metadata))
        logger.info("Prepared %d chunks across %d sections for ingestion.", len(documents), len(sections))
        return documents, len(sections)

    def _extract_sections(self, text: str) -> List[Tuple[str, str]]:
        section_pattern = re.compile(r"(ITEM\s+\d[A-Z]?\.\s*.*?)(?=\nITEM\s+\d[A-Z]?\.|$)", re.IGNORECASE | re.DOTALL)
        matches = list(section_pattern.finditer(text))
        sections: List[Tuple[str, str]] = []

        if not matches:
            sections.append(("Document", text))
            return sections

        for index, match in enumerate(matches):
            title = match.group(1).strip().replace("\n", " ")
            start = match.end()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            section_text = text[start:end].strip()
            if section_text:
                sections.append((title, section_text))

        if not sections:
            sections.append(("Document", text))
        return sections

    def _rebuild_keyword_index(self) -> None:
        if not self._documents_cache:
            self._bm25 = None
            self._doc_map = {}
            self._doc_ids = []
            logger.info("Keyword index cleared; no documents available.")
            return

        corpus_tokens = []
        doc_ids = []
        doc_map: Dict[str, Document] = {}
        for doc in self._documents_cache:
            doc_id = doc.metadata.get("id")
            if not doc_id:
                doc_id = str(uuid.uuid4())
                doc.metadata["id"] = doc_id
            doc_ids.append(doc_id)
            doc_map[doc_id] = doc
            corpus_tokens.append(self._tokenize_for_bm25(doc.page_content))

        self._bm25 = BM25Okapi(corpus_tokens)
        self._doc_ids = doc_ids
        self._doc_map = doc_map
        logger.info("Keyword index built with %d documents.", len(doc_ids))

    def _tokenize_for_bm25(self, text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

    def _load_existing_documents(self) -> None:
        try:
            documents = self.vector_store_service.load_all_documents()
            if documents:
                self._documents_cache = documents
                self._rebuild_keyword_index()
        except Exception as exc:  # pragma: no cover - database connectivity
            logger.warning("Unable to preload documents from PGVector: %s", exc)

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------
    def _build_planner_chain(self):
        parser = PydanticOutputParser(pydantic_object=Plan)
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an expert research planner. Decompose the user's query into a sequenced plan.\n"""
                    "Use the tools `search_10k` (for internal filings) and `search_web` (for external news).\n"
                    "Return the plan as JSON that conforms to the following schema:\n{format_instructions}""",
                ),
                ("human", "User Query: {question}"),
            ]
        ).partial(format_instructions=parser.get_format_instructions())
        chain = prompt | self._structured_llm | parser
        return chain

    def _build_query_rewriter_chain(self):
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """Rewrite the sub-question into a precise search query. Focus on actionable keywords.\n"""
                    "Leverage the provided keywords and past context when available.""",
                ),
                (
                    "human",
                    "Current sub-question: {sub_question}\n"
                    "Relevant keywords: {keywords}\n"
                    "Context from previous steps:\n{past_context}",
                ),
            ]
        )
        return prompt | self._llm | StrOutputParser()

    def _build_retrieval_supervisor_chain(self):
        parser = PydanticOutputParser(pydantic_object=RetrievalDecision)
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You choose retrieval strategies for a research assistant.\n"""
                    "Options: vector_search, keyword_search, hybrid_search.\n"
                    "Return JSON following this schema:\n{format_instructions}""",
                ),
                ("human", "Search request: {sub_question}"),
            ]
        ).partial(format_instructions=parser.get_format_instructions())
        return prompt | self._structured_llm | parser

    def _build_distiller_chain(self):
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """Combine the retrieved snippets into a concise paragraph answering the question.\n"""
                    "Return only the synthesized context.""",
                ),
                (
                    "human",
                    "Question: {question}\nRetrieved Context:\n{context}",
                ),
            ]
        )
        return prompt | self._llm | StrOutputParser()

    def _build_reflection_chain(self):
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """Summarize the distilled context into one factual sentence describing the finding.""",
                ),
                (
                    "human",
                    "Sub-question: {sub_question}\nDistilled context:\n{context}",
                ),
            ]
        )
        return prompt | self._llm | StrOutputParser()

    def _build_policy_chain(self):
        parser = PydanticOutputParser(pydantic_object=Decision)
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """Decide whether the research has enough information to answer the question.\n"""
                    "Respond in JSON following this schema:\n{format_instructions}""",
                ),
                (
                    "human",
                    "Original question: {question}\nPlan: {plan}\nResearch history:\n{history}",
                ),
            ]
        ).partial(format_instructions=parser.get_format_instructions())
        return prompt | self._structured_llm | parser

    def _build_final_answer_chain(self):
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an expert analyst. Synthesize the research findings into a detailed answer.\n"""
                    "Add citations in square brackets using the provided metadata (section titles or URLs).""",
                ),
                (
                    "human",
                    "Original Question: {question}\n\nResearch Context:\n{context}",
                ),
            ]
        )
        return prompt | self._llm | StrOutputParser()

    def _initialise_web_search(self) -> Optional[TavilySearchResults]:
        if not self.settings.enable_web_search:
            logger.info("Web search integration disabled in configuration.")
            return None
        try:
            return TavilySearchResults(k=self.settings.tavily_max_results)
        except Exception as exc:  # pragma: no cover - external dependency
            logger.warning("Failed to initialize Tavily web search: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Retrieval helpers
    # ------------------------------------------------------------------
    def _vector_search_only(self, query: str, section_filter: Optional[str], k: int) -> List[Document]:
        metadata_filter = None
        if section_filter and "unknown" not in section_filter.lower():
            metadata_filter = {"section": section_filter}
        return self.vector_store_service.similarity_search(query, k=k, metadata_filter=metadata_filter)

    def _bm25_search_only(self, query: str, k: int) -> List[Document]:
        if not self._bm25:
            logger.debug("BM25 index unavailable. Returning empty result set.")
            return []
        tokens = self._tokenize_for_bm25(query)
        scores = self._bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:k]
        return [self._doc_map[self._doc_ids[i]] for i in top_indices if scores[i] > 0]

    def _hybrid_search(self, query: str, section_filter: Optional[str], k: int) -> List[Document]:
        keyword_docs = self._bm25_search_only(query, k)
        semantic_docs = self._vector_search_only(query, section_filter, k)
        all_docs = {doc.metadata.get("id"): doc for doc in keyword_docs + semantic_docs if doc.metadata.get("id")}
        if not all_docs:
            return semantic_docs or keyword_docs

        ranked_lists = [
            [doc.metadata.get("id") for doc in keyword_docs if doc.metadata.get("id")],
            [doc.metadata.get("id") for doc in semantic_docs if doc.metadata.get("id")],
        ]
        rrf_scores: Dict[str, float] = {}
        for doc_list in ranked_lists:
            for rank, doc_id in enumerate(doc_list):
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (rank + 61)
        sorted_ids = sorted(rrf_scores, key=lambda doc_id: rrf_scores[doc_id], reverse=True)
        return [all_docs[doc_id] for doc_id in sorted_ids[:k]]

    def _rerank_documents(self, query: str, documents: Sequence[Document]) -> List[Document]:
        if not documents:
            return []
        pairs = [(query, doc.page_content) for doc in documents]
        scores = self._reranker.predict(pairs)
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda item: item[1], reverse=True)
        return [doc for doc, _ in scored_docs[: self.settings.top_n_rerank]]

    def _web_search(self, query: str) -> List[Document]:
        if not self._web_search_tool:
            logger.info("Web search requested but tool unavailable. Returning empty context.")
            return []
        try:
            results = self._web_search_tool.invoke({"query": query})
        except Exception as exc:  # pragma: no cover - network request
            logger.warning("Web search failed: %s", exc)
            return []
        return [Document(page_content=item["content"], metadata={"source": item["url"]}) for item in results]

    # ------------------------------------------------------------------
    # LangGraph construction
    # ------------------------------------------------------------------
    def _build_state_graph(self):
        def format_docs(docs: Iterable[Document]) -> str:
            return "\n\n---\n\n".join(doc.page_content for doc in docs)

        def get_past_context_str(past_steps: List[PastStep]) -> str:
            if not past_steps:
                return ""
            return "\n\n".join(
                f"Step {step['step_index']}: {step['sub_question']}\nSummary: {step['summary']}" for step in past_steps
            )

        def plan_node(state: RAGState) -> Dict:
            question = state["original_question"]
            plan: Plan = self._planner_chain.invoke({"question": question})
            logger.info("Generated plan with %d steps.", len(plan.steps))
            return {"plan": plan, "current_step_index": 0, "past_steps": [], "retrieved_docs": []}

        def retrieval_node(state: RAGState) -> Dict:
            plan: Plan = state["plan"]
            current_index = state.get("current_step_index", 0)
            current_step = plan.steps[current_index]
            past_context = get_past_context_str(state.get("past_steps", []))
            rewritten_query = self._query_rewriter_chain.invoke(
                {
                    "sub_question": current_step.sub_question,
                    "keywords": ", ".join(current_step.keywords),
                    "past_context": past_context,
                }
            )
            logger.info("Step %d query rewritten: %s", current_index + 1, rewritten_query)
            decision: RetrievalDecision = self._retrieval_supervisor_chain.invoke({"sub_question": rewritten_query})
            logger.info("Supervisor strategy: %s", decision.strategy)

            if decision.strategy == "vector_search":
                docs = self._vector_search_only(rewritten_query, current_step.document_section, self.settings.top_k_retrieval)
            elif decision.strategy == "keyword_search":
                docs = self._bm25_search_only(rewritten_query, self.settings.top_k_retrieval)
            else:
                docs = self._hybrid_search(rewritten_query, current_step.document_section, self.settings.top_k_retrieval)

            return {"retrieved_docs": docs}

        def web_search_node(state: RAGState) -> Dict:
            plan: Plan = state["plan"]
            current_index = state.get("current_step_index", 0)
            current_step = plan.steps[current_index]
            past_context = get_past_context_str(state.get("past_steps", []))
            rewritten_query = self._query_rewriter_chain.invoke(
                {
                    "sub_question": current_step.sub_question,
                    "keywords": ", ".join(current_step.keywords),
                    "past_context": past_context,
                }
            )
            logger.info("Step %d web search query: %s", current_index + 1, rewritten_query)
            docs = self._web_search(rewritten_query)
            return {"retrieved_docs": docs}

        def rerank_node(state: RAGState) -> Dict:
            plan: Plan = state["plan"]
            current_index = state.get("current_step_index", 0)
            current_step = plan.steps[current_index]
            reranked = self._rerank_documents(current_step.sub_question, state.get("retrieved_docs", []))
            return {"reranked_docs": reranked}

        def compression_node(state: RAGState) -> Dict:
            plan: Plan = state["plan"]
            current_index = state.get("current_step_index", 0)
            current_step = plan.steps[current_index]
            context = format_docs(state.get("reranked_docs", []))
            synthesized_context = self._distiller_chain.invoke(
                {"question": current_step.sub_question, "context": context}
            )
            return {"synthesized_context": synthesized_context}

        def reflection_node(state: RAGState) -> Dict:
            plan: Plan = state["plan"]
            current_index = state.get("current_step_index", 0)
            current_step = plan.steps[current_index]
            summary = self._reflection_chain.invoke(
                {
                    "sub_question": current_step.sub_question,
                    "context": state.get("synthesized_context", ""),
                }
            )
            past_steps = state.get("past_steps", [])
            new_step: PastStep = {
                "step_index": current_index + 1,
                "sub_question": current_step.sub_question,
                "retrieved_docs": state.get("reranked_docs", []),
                "summary": summary,
            }
            return {
                "past_steps": past_steps + [new_step],
                "current_step_index": current_index + 1,
                "retrieved_docs": [],
                "reranked_docs": [],
            }

        def final_answer_node(state: RAGState) -> Dict:
            past_steps = state.get("past_steps", [])
            context_parts = []
            for idx, step in enumerate(past_steps):
                context_parts.append(f"--- Findings from Step {idx + 1} ---")
                for doc in step["retrieved_docs"]:
                    source = doc.metadata.get("section") or doc.metadata.get("source")
                    context_parts.append(f"Source: {source}\nContent: {doc.page_content}")
            combined_context = "\n\n".join(context_parts)
            answer = self._final_answer_chain.invoke(
                {"question": state.get("original_question", ""), "context": combined_context}
            )
            return {"final_answer": answer}

        def route_by_tool(state: RAGState) -> str:
            plan: Plan = state["plan"]
            current_index = state.get("current_step_index", 0)
            if not plan.steps or current_index >= len(plan.steps):
                return "finish"
            return plan.steps[current_index].tool

        def should_continue_node(state: RAGState) -> str:
            current_index = state.get("current_step_index", 0)
            plan: Plan = state["plan"]
            if current_index >= len(plan.steps):
                return "finish"
            if current_index >= self.settings.max_reasoning_iterations:
                logger.info("Maximum reasoning iterations reached. Finishing.")
                return "finish"
            if not state.get("reranked_docs"):
                logger.info("No reranked documents found. Continuing to next step.")
                return "continue"
            history = "" if not state.get("past_steps") else json.dumps(
                [
                    {
                        "step_index": step["step_index"],
                        "sub_question": step["sub_question"],
                        "summary": step["summary"],
                    }
                    for step in state.get("past_steps", [])
                ]
            )
            plan_payload = json.dumps([step.dict() for step in plan.steps])
            decision: Decision = self._policy_chain.invoke(
                {
                    "question": state.get("original_question", ""),
                    "plan": plan_payload,
                    "history": history,
                }
            )
            logger.info("Policy decision: %s", decision.next_action)
            return "finish" if decision.next_action == "FINISH" else "continue"

        graph = StateGraph(RAGState)
        graph.add_node("plan", plan_node)
        graph.add_node("retrieve_10k", retrieval_node)
        graph.add_node("retrieve_web", web_search_node)
        graph.add_node("rerank", rerank_node)
        graph.add_node("compress", compression_node)
        graph.add_node("reflect", reflection_node)
        graph.add_node("final", final_answer_node)

        graph.set_entry_point("plan")
        graph.add_conditional_edges(
            "plan",
            route_by_tool,
            {
                "search_10k": "retrieve_10k",
                "search_web": "retrieve_web",
                "finish": "final",
            },
        )
        graph.add_edge("retrieve_10k", "rerank")
        graph.add_edge("retrieve_web", "rerank")
        graph.add_edge("rerank", "compress")
        graph.add_edge("compress", "reflect")
        graph.add_conditional_edges(
            "reflect",
            should_continue_node,
            {
                "continue": "plan",
                "finish": "final",
            },
        )
        graph.add_edge("final", END)

        return graph.compile()


__all__ = ["DeepThinkingRAGService", "IngestionResult"]
