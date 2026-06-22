# main_pipeline.py
import json
import re
from tqdm import tqdm
from retriever import LegalRetriever
from generator import generate_answer

RETRIEVER = LegalRetriever(
    index_path="index/faiss.index",
    meta_path="index/metadata.pkl",
    model_name="bkai-foundation-models/vietnamese-bi-encoder"
)

TOP_K = 5  # Số điều luật đưa vào context cho LLM

def format_relevant_docs(articles: list) -> list:
    seen = set()
    result = []
    for art in articles:
        meta = art["metadata"]
        law_id = meta["law_id"]  # VD: "04/2017/QH14"
        
        # BẮT BUỘC: Bạn phải ghép chuỗi sao cho ra đúng định dạng BTC yêu cầu:
        # "Loại văn bản + Mã văn bản + Trích yếu"
        # Ví dụ: Nếu meta["law_name"] đang là "Luật Hỗ trợ doanh nghiệp nhỏ và vừa"
        # và meta["type"] là "Luật"
        # Thì chuỗi chuẩn sẽ là: f"{meta['type']} {law_id} {meta['law_name']}"
        # (Lưu ý: Tùy file data của bạn mà biến tên có thể khác, hãy in thử ra để kiểm tra)
        
        # Giả định chuẩn hóa tên:
        full_law_name = f"Luật {law_id} {meta['law_name']}" 
        
        formatted_str = f"{law_id}|{full_law_name}"
        if formatted_str not in seen:
            seen.add(formatted_str)
            result.append(formatted_str)
    return result

def format_relevant_articles(articles: list) -> list:
    seen = set()
    result = []
    for art in articles:
        meta = art["metadata"]
        law_id = meta["law_id"]
        full_law_name = f"Luật {law_id} {meta['law_name']}" 
        
        # Đảm bảo chữ "Điều" gọn gàng, không có dấu chấm ở cuối
        clean_article = meta["article"].strip().rstrip('.') 
        
        formatted_str = f"{law_id}|{full_law_name}|{clean_article}"
        if formatted_str not in seen:
            seen.add(formatted_str)
            result.append(formatted_str)
    return result

def process_question(q: dict) -> dict:
    question = q["question"]
    
    # Bước 1: Retrieve
    retrieved = RETRIEVER.hybrid_retrieve(question, top_k=TOP_K)
    
    # Bước 2: Generate answer
    answer = generate_answer(question, retrieved)
    
    # Bước 3: Format output
    return {
        "id": q["id"],
        "question": question,
        "answer": answer,
        "relevant_docs": format_relevant_docs(retrieved),
        "relevant_articles": format_relevant_articles(retrieved)
    }

def run_pipeline(test_file: str, output_file: str):
    with open(test_file, "r", encoding="utf-8") as f:
        questions = json.load(f)
    
    results = []
    for q in tqdm(questions, desc="Processing"):
        try:
            result = process_question(q)
            results.append(result)
        except Exception as e:
            print(f"Error on id={q['id']}: {e}")
            # Fallback: nộp bài rỗng để không bị thiếu câu
            results.append({
                "id": q["id"],
                "question": q["question"],
                "answer": "",
                "relevant_docs": [],
                "relevant_articles": []
            })
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"Done! {len(results)} results saved to {output_file}")

if __name__ == "__main__":
    run_pipeline("data/test.json", "results.json")