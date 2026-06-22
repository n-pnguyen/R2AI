# retriever.py
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict

class LegalRetriever:
    def __init__(self, index_path: str, meta_path: str, model_name: str):
        self.model = SentenceTransformer(model_name, device="cpu")
        self.index = faiss.read_index(index_path)
        with open(meta_path, "rb") as f:
            self.corpus = pickle.load(f)
    
    def retrieve(self, query: str, top_k: int = 10) -> List[Dict]:
        """Truy hồi top_k điều luật liên quan nhất"""
        # Embed query
        q_emb = self.model.encode(
            [query], 
            normalize_embeddings=True
        ).astype(np.float32)
        
        # Search
        scores, indices = self.index.search(q_emb, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            item = self.corpus[idx].copy()
            item["score"] = float(score)
            results.append(item)
        
        return results
    
    def hybrid_retrieve(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        Hybrid: kết hợp dense retrieval + BM25 keyword search
        Tăng recall đáng kể
        """
        dense_results = self.retrieve(query, top_k * 2)
        bm25_results  = self.bm25_search(query, top_k * 2)
        
        # Reciprocal Rank Fusion
        return self._rrf_merge(dense_results, bm25_results, top_k)
    
    def bm25_search(self, query: str, top_k: int) -> List[Dict]:
        """BM25 keyword search"""
        from rank_bm25 import BM25Okapi
        # Khuyên dùng underthesea để tokenize tiếng Việt thay vì split() chay
        from underthesea import word_tokenize
        
        # SỬA Ở ĐÂY: Trích xuất từ item["text"]
        tokenized_corpus = [word_tokenize(item["text"].lower()) for item in self.corpus]
        bm25 = BM25Okapi(tokenized_corpus)
        
        tokenized_query = word_tokenize(query.lower())
        scores = bm25.get_scores(tokenized_query)
        
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            item = self.corpus[idx].copy()
            item["score"] = float(scores[idx])
            results.append(item)
        return results
    
    def _rrf_merge(self, list1, list2, top_k, k=60):
        """Reciprocal Rank Fusion"""
        scores = {}
        for rank, item in enumerate(list1):
            cid = item["chunk_id"]
            scores[cid] = scores.get(cid, 0) + 1/(k + rank + 1)
        for rank, item in enumerate(list2):
            cid = item["chunk_id"]
            scores[cid] = scores.get(cid, 0) + 1/(k + rank + 1)
        
        # Map chunk_id → item
        all_items = {item["chunk_id"]: item for item in list1 + list2}
        sorted_ids = sorted(scores, key=scores.get, reverse=True)[:top_k]
        return [all_items[cid] for cid in sorted_ids]