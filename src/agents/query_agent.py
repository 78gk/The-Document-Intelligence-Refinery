import sqlite3
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from ..models.refinery_models import LDU, PageIndex, ProvenanceChain, Provenance, ChunkType

class QueryInterfaceAgent:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.db_path = Path(".refinery/fact_table.db")
        self._init_db()
        self.vector_store = {} # Mock vector store
        self.page_indices = {}

    def _init_db(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id TEXT,
                key TEXT,
                value TEXT,
                page_ref INTEGER,
                content_hash TEXT
            )
        ''')
        conn.commit()
        conn.close()

    def ingest_ldus(self, doc_id: str, ldus: List[LDU]):
        # Mock vector ingestion
        self.vector_store[doc_id] = ldus
        
        # Ingest into FactTable (Simulated KV extraction from tables)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        for ldu in ldus:
            if ldu.chunk_type == ChunkType.TABLE:
                # Simplistic KV extraction for demo
                cursor.execute(
                    "INSERT INTO facts (doc_id, key, value, page_ref, content_hash) VALUES (?, ?, ?, ?, ?)",
                    (doc_id, "table_summary", ldu.content[:100], ldu.page_refs[0], ldu.content_hash)
                )
        conn.commit()
        conn.close()

    def load_page_index(self, doc_id: str):
        path = Path(f".refinery/pageindex/{doc_id}.json")
        if path.exists():
            with open(path, 'r') as f:
                self.page_indices[doc_id] = PageIndex.model_validate(json.load(f))

    def pageindex_navigate(self, doc_id: str, query: str) -> str:
        index = self.page_indices.get(doc_id)
        if not index:
            return "PageIndex not loaded."
        
        # Simplistic navigation: find sections with query words in title
        results = []
        def traverse(node):
            if query.lower() in node.title.lower():
                results.append(f"Section: {node.title} (Pages {node.page_start}-{node.page_end})")
            for child in node.child_sections:
                traverse(child)
        
        traverse(index.root)
        return "\n".join(results) or "No relevant sections found in PageIndex."

    def semantic_search(self, doc_id: str, query: str) -> List[LDU]:
        ldus = self.vector_store.get(doc_id, [])
        # Simulated semantic search (keyword based for mock)
        results = [ldu for ldu in ldus if query.lower() in ldu.content.lower()]
        return results[:3]

    def structured_query(self, doc_id: str, sql: str) -> List[Dict[str, Any]]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def answer_query(self, doc_id: str, query: str) -> Dict[str, Any]:
        # Orchestrate tools to answer
        # 1. Check PageIndex
        # 2. Search LDUs
        # 3. Formulate Answer with Provenance
        
        search_results = self.semantic_search(doc_id, query)
        if not search_results:
            return {"answer": "I'm sorry, I couldn't find information matching your query.", "provenance": []}
        
        top_hit = search_results[0]
        answer = f"Based on the document, {top_hit.content[:200]}..."
        
        provenance = ProvenanceChain(
            document_name=doc_id,
            citations=[
                Provenance(
                    document_name=doc_id,
                    page_number=top_hit.page_refs[0],
                    bbox=top_hit.bounding_box,
                    content_hash=top_hit.content_hash
                )
            ],
        )
        
        return {
            "answer": answer,
            "provenance": provenance.model_dump()
        }
