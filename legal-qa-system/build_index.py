# build_index.py
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pickle

EMBEDDING_MODEL = "bkai-foundation-models/vietnamese-bi-encoder"
DATA_PATH = "data/legal_chunks.jsonl"  # Đổi đuôi thành .jsonl
INDEX_PATH = "index/faiss.index"
META_PATH  = "index/metadata.pkl"

def build_faiss_index():
    corpus = []
    # Đọc file JSONL từng dòng
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                corpus.append(json.loads(line))
    
    print(f"Loaded {len(corpus)} chunks.")
    
    model = SentenceTransformer(EMBEDDING_MODEL)
    
    # Tạo text để embed: Lấy từ metadata và text thực tế
    texts = []
    for item in corpus:
        meta = item["metadata"]
        # Ví dụ: "Bộ Luật dân sự (91/2015/QH13) - Điều 1.: Điều 1. Phạm vi điều chỉnh..."
        text = f"{meta['law_name']} ({meta['law_id']}) - {meta['article']}: {item['text']}"
        texts.append(text)
        
    print("Encoding articles (this might take a while)...")
    embeddings = model.encode(
        texts, 
        batch_size=32, 
        show_progress_bar=True,
        normalize_embeddings=True 
    )
    
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))
    
    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "wb") as f:
        pickle.dump(corpus, f)
        
    print(f"Index built: {index.ntotal} vectors, dim={dim}")

if __name__ == "__main__":
    build_faiss_index()