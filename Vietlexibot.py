
import os
import json
import fitz  # PyMuPDF
import re
import logging
from datetime import datetime
import traceback
import cv2
import numpy as np
from PIL import Image
import pytesseract
import ollama
import time
from concurrent.futures import ThreadPoolExecutor
import unicodedata

# === MODELS ===
EMBEDDING_MODEL = 'bge-m3:latest'
VECTOR_DB = []

def setup_logging():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    conversation_logger = logging.getLogger('conversation')
    conversation_logger.setLevel(logging.INFO)
    ch = logging.FileHandler(f"{log_dir}/conversation_{timestamp}.log", encoding="utf-8")
    conversation_logger.addHandler(ch)
    return conversation_logger

def enhance_for_ocr(pix):
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    sharpened = cv2.addWeighted(gray, 1.5, cv2.GaussianBlur(gray, (0, 0), 3), -0.5, 0)
    _, thresh = cv2.threshold(sharpened, 180, 255, cv2.THRESH_BINARY)
    return Image.fromarray(thresh)

def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            pix = page.get_pixmap(dpi=300)
            image = enhance_for_ocr(pix)
            text += pytesseract.image_to_string(image, lang='vie') + "\n"
    return text.strip()

def extract_structured_fields(raw_text, filename=None):
    fields = {}

    # Extract số hiệu and ngày ban hành (same as before)
    match_so_hieu = re.search(r'Số:\s*([0-9][^\s\n]*)', raw_text)
    if match_so_hieu and match_so_hieu.group(1):
        fields["số hiệu"] = f"{match_so_hieu.group(1)}"
    elif filename:
        match_from_filename = re.search(r'(\d{3,5})', filename)
        if match_from_filename:
            fields["số hiệu"] = f"{match_from_filename.group(1)}"
        else:
            fields["số hiệu"] = "Không rõ"
    else:
        fields["số hiệu"] = "Không rõ"

    # Extract date (same logic as before)
    match_ngay = re.search(
        r'\s*ngày\s*(\d{1,2}|[^\d\s]*)?\s*tháng\s*(\d{1,2}|[^\d\s]*)?\s*năm\s*(\d{4})',
        raw_text
    )
    if match_ngay:
        day, month, year = match_ngay.groups()
        replacements = {
            "70": "10", "l0": "10", "to": "10", "T0": "10", "O": "0", "l": "1"
        }
        for wrong, correct in replacements.items():
            if day:
                day = day.replace(wrong, correct)
            if month:
                month = month.replace(wrong, correct)
        day = day.zfill(2) if day and day.isdigit() else "??"
        month = month.zfill(2) if month and month.isdigit() else "??"
        fields["ngày ban hành"] = f"{day}/{month}/{year}"
    else:
        fields["ngày ban hành"] = "Không rõ"

    # Key-value format prompt
    prompt = f"""
Hãy phân tích văn bản hành chính và trả lời theo ĐÚNG định dạng sau:

TRICH_YEU: [Một câu tóm tắt ngắn gọn nội dung chính]
NOI_DUNG: [2-3 câu mô tả chi tiết hành động, quyết định cụ thể]

QUAN TRỌNG:
- Chỉ trả lời đúng 2 dòng theo định dạng trên
- Bắt đầu mỗi dòng bằng "TRICH_YEU:" hoặc "NOI_DUNG:"
- Không thêm bất kỳ văn bản nào khác

Văn bản:
\"\"\"{raw_text[:8000]}\"\"\"
""".strip()

    try:
        response = ollama.chat(
            model='hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF',
            messages=[{"role": "user", "content": prompt}],
            stream=False,
            options={"temperature": 0.1}
        )
        llm_reply = response['message']['content'].strip()

        # Extract using key-value format
        extracted_fields = extract_keyvalue_format(llm_reply)
        
        fields["trích yếu"] = extracted_fields.get("trích yếu", "Không rõ")
        fields["nội dung"] = extracted_fields.get("nội dung", "Không rõ")

    except Exception as e:
        fields["trích yếu"] = "Không rõ"
        fields["nội dung"] = "Không rõ"
        print(f"[ERROR] Lỗi LLM: {e}")

    return validate_extracted_fields(fields)

def extract_keyvalue_format(llm_reply):
    fields = {}
    
    # Split by lines and process each
    lines = llm_reply.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if line.startswith('TRICH_YEU:'):
            fields["trích yếu"] = line.replace('TRICH_YEU:', '').strip()
        elif line.startswith('NOI_DUNG:'):
            fields["nội dung"] = line.replace('NOI_DUNG:', '').strip()
        # Also handle variations
        elif line.lower().startswith('trích yếu:'):
            fields["trích yếu"] = line.split(':', 1)[1].strip()
        elif line.lower().startswith('nội dung:'):
            fields["nội dung"] = line.split(':', 1)[1].strip()
    
    return fields

def extract_text_from_pdf_parallel(pdf_path, max_workers=os.cpu_count()):
    def ocr_page(page_number):
        try:
            page = doc.load_page(page_number)
            pix = page.get_pixmap(dpi=300)
            image = enhance_for_ocr(pix)
            return pytesseract.image_to_string(image, lang='vie')
        except Exception as e:
            print(f"[ERROR] OCR failed on page {page_number}: {e}")
            return ""

    with fitz.open(pdf_path) as doc:
        page_numbers = list(range(len(doc)))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            texts = list(executor.map(ocr_page, page_numbers))
    return "\n".join(texts).strip()

def validate_extracted_fields(fields):
    required_fields = ["số hiệu", "ngày ban hành", "trích yếu", "nội dung"]
    
    for field in required_fields:
        if field not in fields or not fields[field] or fields[field].strip() == "":
            fields[field] = "Không rõ"
        else:
            # Clean up the field value
            value = str(fields[field]).strip()
            # Remove excessive whitespace
            value = re.sub(r'\s+', ' ', value)
            # Remove quotes if they wrap the entire value
            if value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            elif value.startswith("'") and value.endswith("'"):
                value = value[1:-1]
            fields[field] = value
    
    return fields

# === EMBEDDING ===
def split_chunks_with_context(text, max_chars=1000):
    paras = re.split(r'\n\s*\n', text)
    chunks = []
    current = ""
    for para in paras:
        if len(current) + len(para) > max_chars:
            chunks.append(current.strip())
            current = para
        else:
            current += "\n\n" + para
    if current:
        chunks.append(current.strip())
    return chunks

def add_chunk_to_database(chunk):
    try:
        result = ollama.embed(model=EMBEDDING_MODEL, input=chunk)
        embeddings = result['embeddings']
        for chunk, emb in zip(chunk, embeddings):
            VECTOR_DB.append((chunk, emb))
    except Exception as e:
        print("[ERROR] Batch embedding failed:", e)
        for chunk in chunk:
            try:
                emb = ollama.embed(model=EMBEDDING_MODEL, input=chunk)['embeddings'][0]
                VECTOR_DB.append((chunk, emb))
            except Exception as e2:
                print(f"[ERROR] Failed on chunk: {chunk[:100]}... - {e2}")

def cosine_similarity(a, b):
    dot = sum(x*y for x, y in zip(a, b))
    norm_a = sum(x**2 for x in a)**0.5
    norm_b = sum(x**2 for x in b)**0.5
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0

def retrieve(query, top_n=8):
    query_emb = ollama.embed(model=EMBEDDING_MODEL, input=query)['embeddings'][0]
    sims = [(chunk, cosine_similarity(query_emb, emb)) for chunk, emb in VECTOR_DB]
    return sorted(sims, key=lambda x: x[1], reverse=True)[:top_n]

# === LLM Answering ===
def generate_answer(context_chunks, query, metadata):
    prompt = f"""
Bạn là một trợ lý AI chuyên trả lời câu hỏi chỉ dựa vào nội dung được cung cấp và trả lời bằng tiếng việt.

**QUY TẮC BẮT BUỘC:**
1.  **CHỈ SỬ DỤNG DỮ LIỆU DƯỚI ĐÂY.** Tuyệt đối nghiêm cấm trả lời và sử dụng bất kỳ kiến thức nào bên ngoài.
2.  **Khi trả lời viết ngắn gọn, tuyệt đối không trích dẫn/nhắc lại nội dung được cung cấp.**
3.  **Nếu câu trả lời không có trong văn bản được cung cấp**, hãy trả lời rõ ràng là: "Thông tin này không có trong tài liệu."


**Dữ liệu được cung cấp:**
---
{chr(10).join(context_chunks)}
---

**Câu hỏi của người dùng:** {query}

**Câu trả lời của bạn:**
"""
    
    response = ollama.chat(
        model='llama3:8b-instruct-q5_K_S',
        messages=[{'role': 'user', 'content': prompt}],
        options={"temperature": 0.6},
        stream=False
    )
    return response['message']['content'].strip()

def suggest_tasks_from_document(content):
    prompt = f"""
Dựa vào nội dung tài liệu sau, hãy liệt kê những công việc hoặc hành động cụ thể và rõ ràng/chi tiết cho các đơn vị liên quan có thể thực hiện. Không suy đoán các đơn vị và không nêu nhiệm vụ không được đề cập.
Sau đó, hãy phân loại nhiệm vụ thành 3 nhóm theo mức độ ưu tiên , luôn luôn nêu rõ từng đơn vị cụ thể theo định dạng như sau:
Ưu tiên cao:
- [tên đơn vị] Các nhiệm vụ có tính chất khẩn cấp, có hạn gần hoặc yêu cầu thực hiện ngay.
lọc Từ khóa gợi ý trong văn bản: "khẩn", "hỏa tốc", "ngay lập tức", "báo cáo gấp", "trước ngày", "phải xong trong hôm nay", "trình ký gấp", "hoàn thành trong X ngày".

Ưu tiên trung bình:
- [tên đơn vị] Nhiệm vụ mang tính hỗ trợ, phối hợp, thường có hạn trong vài ngày.
lọc Từ khóa gợi ý trong văn bản: "phối hợp", "chuẩn bị", "tổng hợp", "làm rõ", "soạn thảo", "xử lý trong tuần", "trao đổi thêm".

Ưu tiên thấp:
- [tên đơn vị] Nhiệm vụ dài hạn, chưa có deadline cụ thể, mang tính định hướng.
lọc Từ khóa gợi ý trong văn bản: "nghiên cứu", "xem xét", "đề xuất", "dự thảo", "tìm hiểu", "sẽ thực hiện", "trong tuần tới".

Văn bản đầu vào:
\"\"\"{content[:100000]}\"\"\"
""".strip()

    try:
        response = ollama.chat(
            model='hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF',
            messages=[{'role': 'user', 'content': prompt}],
            stream=False,
            options={"temperature": 0.7}
        )
        return response['message']['content'].strip()
    except Exception as e:
        return f"Lỗi khi gợi ý công việc: {e}"

# === MAIN ===
def main():
    logger = setup_logging()
    total_start_time = time.time()

    # Force PDF mode
    pdf_files = [f for f in os.listdir('.') if f.lower().endswith('.pdf')]
    if not pdf_files:
        print("Không tìm thấy tệp PDF nào trong thư mục hiện tại.")
        return

    print("Các tệp PDF có sẵn:")
    for i, f in enumerate(pdf_files):
        print(f"{i + 1}. {f}")

    while True:
        try:
            choice = int(input("Chọn số thứ tự của tệp PDF: "))
            if 1 <= choice <= len(pdf_files):
                pdf_path = pdf_files[choice - 1]
                break
            else:
                print("Lựa chọn không hợp lệ. Vui lòng thử lại.")
        except ValueError:
            print("Vui lòng nhập một số.")

    # OCR & extract
    text = extract_text_from_pdf_parallel(pdf_path)
    text = unicodedata.normalize("NFC", text)

    start_structured = time.time()
    structured = extract_structured_fields(text, filename=pdf_path)
    end_structured = time.time()
    print(f"[INFO] Thời gian trích xuất thông tin: {end_structured - start_structured:.2f} giây")

    metadata = structured

    start_vector = time.time()
    chunks = split_chunks_with_context(text)
    add_chunk_to_database(chunks)
    end_vector = time.time()
    print(f"[INFO] Đã thêm {len(chunks)} văn bản. Thời gian xử lý: {end_vector - start_vector:.2f} giây")
    total_end_time = time.time()
    print(f"[INFO] Tổng thời gian tiền xử lý tài liệu: {total_end_time - total_start_time:.2f} giây")

    print("\nCác trường được trích xuất:")
    for k, v in structured.items():
        print(f"- {k.capitalize()}: {v[:1200]}{'...' if len(v) > 1200 else ''}")

    print("\nGợi ý công việc từ tài liệu:")
    start_task = time.time()
    suggestions = suggest_tasks_from_document(text)
    end_task = time.time()
    print(suggestions)
    print(f"\n\n[INFO] Thời gian xử lý gợi ý công việc: {end_task - start_task:.2f} giây\n\n")

    while True:
        q = input("\nHỏi gì đó (/bye để thoát): ")
        if q.strip() == '/bye':
            break
        #logger.info(f"User: {q.strip()}")
        retrieved = retrieve(q, top_n=8)
        best_chunks = [chunk for chunk, _ in retrieved]

        start_answer = time.time()
        answer = generate_answer(best_chunks, q, metadata)
        end_answer = time.time()

        print(f"\nAI: {answer}")
        #logger.info(f"AI: {answer}")
        print(f"[INFO] Thời gian tạo câu trả lời: {end_answer - start_answer:.2f} giây")
if __name__ == "__main__":
    main()