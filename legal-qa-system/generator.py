# generator.py
import requests
import json

OLLAMA_URL = "http://localhost:11434/api/generate"

SYSTEM_PROMPT = """Bạn là trợ lý pháp lý chuyên về luật doanh nghiệp Việt Nam.
Nhiệm vụ: Trả lời câu hỏi pháp lý DỰA TRÊN các điều luật được cung cấp.
Yêu cầu:
- Trả lời bằng tiếng Việt, rõ ràng, dễ hiểu
- CHỈ dùng thông tin từ điều luật được cung cấp
- Trích dẫn cụ thể "theo Điều X, Luật Y"
- Nếu điều luật không đủ thông tin, nói rõ giới hạn
- KHÔNG bịa đặt điều luật không có trong context"""

def generate_answer(question: str, retrieved_articles: list) -> str:
    context_parts = []
    for art in retrieved_articles:
        meta = art['metadata']
        # SỬA Ở ĐÂY: Lấy law_name, article từ metadata và nội dung từ text
        context_parts.append(
            f"[{meta['law_name']} - {meta['article']}]\n{art['text']}"
        )
    
    context = "\n\n---\n\n".join(context_parts)
    
    prompt = f"""CÁC ĐIỀU LUẬT LIÊN QUAN:\n{context}\n\nCÂU HỎI: {question}\nHãy trả lời câu hỏi trên dựa vào các điều luật được cung cấp."""
    
    payload = {
        "model": "qwen2.5:7b",
        "prompt": prompt,
        "system": SYSTEM_PROMPT,
        "stream": False,
        "options": {
            "temperature": 0.1,   # Thấp để câu trả lời ổn định, ít hallucination
            "num_predict": 1024,
        }
    }
    
    resp = requests.post(OLLAMA_URL, json=payload)
    return resp.json()["response"]