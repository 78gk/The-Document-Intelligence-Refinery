import sqlite3
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from ..models.refinery_models import LDU, PageIndex, ProvenanceChain, Provenance, ChunkType
from .indexer import PageIndexBuilder


class QueryInterfaceAgent:
    """Agentic query interface with three tools:
      1. pageindex_navigate – section-level traversal via PageIndex
      2. semantic_search    – vector retrieval over LDUs
      3. structured_query   – SQL over the FactTable

    Also provides an Audit Mode (verify_claim) for claim verification.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.db_path = Path(".refinery/fact_table.db")
        self._init_db()
        # doc_id -> list of {ldu: LDU, vector: np.ndarray}
        self.vector_store: Dict[str, List[Dict[str, Any]]] = {}
        self.page_indices: Dict[str, PageIndex] = {}

    # ------------------------------------------------------------------ #
    #  Initialisation                                                      #
    # ------------------------------------------------------------------ #

    def _init_db(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id TEXT,
                key TEXT,
                value TEXT,
                page_ref INTEGER,
                content_hash TEXT
            )
        """)
        conn.commit()
        conn.close()

    # ------------------------------------------------------------------ #
    #  Ingestion                                                           #
    # ------------------------------------------------------------------ #

    def ingest_ldus(self, doc_id: str, ldus: List[LDU]):
        """Ingest LDUs into the vector store and fact table."""
        self.vector_store[doc_id] = []
        for ldu in ldus:
            vector = self._get_embedding(ldu.content)
            self.vector_store[doc_id].append({"ldu": ldu, "vector": vector})

        # Ingest numerical facts from TABLE chunks into SQLite
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        for ldu in ldus:
            if ldu.chunk_type == ChunkType.TABLE:
                facts = self._extract_facts_from_table(ldu)
                for key, val in facts:
                    cursor.execute(
                        "INSERT INTO facts (doc_id, key, value, page_ref, content_hash) VALUES (?, ?, ?, ?, ?)",
                        (doc_id, key, val, ldu.page_refs[0], ldu.content_hash),
                    )
        conn.commit()
        conn.close()

    def _get_embedding(self, text: str) -> np.ndarray:
        """Hash-based mock embedding for demonstration."""
        vec = np.zeros(128)
        for word in text.lower().split():
            h = hash(word) % 128
            vec[h] += 1
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def _extract_facts_from_table(self, ldu: LDU) -> List[tuple]:
        """Heuristic for extracting key-value numerical facts."""
        facts = []
        lines = ldu.content.split("\n")
        for line in lines:
            if "|" in line:
                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 2 and any(
                    c in parts[1] for c in ["$", "%", "€", "Birr"]
                ):
                    facts.append((parts[0], parts[1]))
        return facts

    # ------------------------------------------------------------------ #
    #  Tool 1: PageIndex Navigation                                        #
    # ------------------------------------------------------------------ #

    def load_page_index(self, doc_id: str):
        """Load a previously serialized PageIndex from disk."""
        path = Path(f".refinery/pageindex/{doc_id}.json")
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                self.page_indices[doc_id] = PageIndex.model_validate(
                    json.load(f)
                )

    def pageindex_navigate(self, doc_id: str, query: str) -> str:
        """Tool: section-level traversal using PageIndexBuilder.traverse()."""
        index = self.page_indices.get(doc_id)
        if not index:
            return "PageIndex not loaded."

        builder = PageIndexBuilder(self.config)
        relevant = builder.traverse(index, query)
        if not relevant:
            return "No relevant sections found in PageIndex."

        lines = []
        for node in relevant[:5]:
            entities = (
                ", ".join(node.key_entities[:5]) if node.key_entities else "N/A"
            )
            lines.append(
                f"Section: {node.title} (Pages {node.page_start}-{node.page_end})\n"
                f"Summary: {node.summary}\n"
                f"Key Entities: {entities}"
            )
        return "\n\n".join(lines)

    # ------------------------------------------------------------------ #
    #  Tool 2: Semantic Search                                             #
    # ------------------------------------------------------------------ #

    def semantic_search(self, doc_id: str, query: str) -> List[LDU]:
        """Tool: vector similarity search over ingested LDUs."""
        stored = self.vector_store.get(doc_id, [])
        if not stored:
            return []

        query_vec = self._get_embedding(query)
        scored = []
        for item in stored:
            score = float(np.dot(query_vec, item["vector"]))
            scored.append((score, item["ldu"]))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [ldu for score, ldu in scored if score > 0.1][:3]

    # ------------------------------------------------------------------ #
    #  Tool 3: Structured Query (SQL)                                      #
    # ------------------------------------------------------------------ #

    def structured_query(self, doc_id: str, sql: str) -> List[Dict[str, Any]]:
        """Tool: execute SQL against the FactTable."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]

    # ------------------------------------------------------------------ #
    #  Audit Mode                                                          #
    # ------------------------------------------------------------------ #

    def verify_claim(self, doc_id: str, claim: str) -> Dict[str, Any]:
        """Audit Mode: verify a claim against document provenance."""
        results = self.semantic_search(doc_id, claim)
        if not results:
            return {
                "status": "unverifiable",
                "reason": "No supporting evidence found in document.",
            }

        top_hit = results[0]
        overlap = [
            w
            for w in claim.split()
            if len(w) > 3 and w.lower() in top_hit.content.lower()
        ]
        if overlap:
            provenance = ProvenanceChain(
                document_name=doc_id,
                citations=[
                    Provenance(
                        document_name=doc_id,
                        page_number=top_hit.page_refs[0],
                        bbox=top_hit.bounding_box,
                        content_hash=top_hit.content_hash,
                    )
                ],
            )
            return {
                "status": "verified",
                "source": top_hit.content[:200],
                "provenance": provenance.model_dump(),
            }

        return {"status": "uncertain", "evidence": top_hit.content[:200]}

    # ------------------------------------------------------------------ #
    #  Answer orchestration                                                #
    # ------------------------------------------------------------------ #

    def answer_query(self, doc_id: str, query: str) -> Dict[str, Any]:
        """Top-level agent orchestrator with tool-selection logic."""
        context: List[str] = []
        used_tools: List[str] = []
        citations: List[Provenance] = []

        is_numerical = any(
            kw in query.lower()
            for kw in [
                "total", "revenue", "how many", "count",
                "amount", "percent", "average",
            ]
        )

        # Tool 1: PageIndex navigation
        nav = self.pageindex_navigate(doc_id, query)
        if nav not in ("PageIndex not loaded.", "No relevant sections found in PageIndex."):
            context.append(f"Navigation: {nav}")
            used_tools.append("pageindex_navigate")

        # Tool 2: Structured query for numerical facts
        if is_numerical:
            try:
                sql = f"SELECT key, value FROM facts WHERE doc_id = '{doc_id}'"
                rows = self.structured_query(doc_id, sql)
                if rows:
                    context.append(f"FactTable: {rows[:5]}")
                    used_tools.append("structured_query")
            except sqlite3.Error:
                pass

        # Tool 3: Semantic search
        hits = self.semantic_search(doc_id, query)
        for ldu in hits:
            context.append(ldu.content)
            citations.append(
                Provenance(
                    document_name=doc_id,
                    page_number=ldu.page_refs[0],
                    bbox=ldu.bounding_box,
                    content_hash=ldu.content_hash,
                )
            )
        used_tools.append("semantic_search")

        # Synthesize answer
        if context:
            snippet = "\n".join(context[:5])[:300]
            answer = f"According to the document ({', '.join(used_tools)}), {snippet}..."
        else:
            answer = "I'm sorry, I couldn't find information matching your query."

        return {
            "query": query,
            "answer": answer,
            "provenance": (
                ProvenanceChain(
                    document_name=doc_id, citations=citations
                ).model_dump()
                if citations
                else []
            ),
            "metadata": {"used_tools": used_tools},
        }
