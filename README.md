# Vietlexicon

## Table of Contents

- [About](#about)
- [Getting Started](#getting-started)
- [Output Example](#output-example)

## About <a name = "about"></a>

**Vietlexicon** is a Vietnamese-centric document intelligence system that extracts, structures, and understands information specialized for PDFs. Main goal is to support government's workers handle forms, and legal documents.  

Built with support for OCR (Optical Character Recognition), NLP (Natural Language Processing), and Vietnamese linguistic nuances, the system leverages LLMs and embedding models to assist in intelligent search, classification, and content parsing.

It supports hybrid pipelines combining traditional text extraction, and vector search with reranking. Vietlexicon is ideal for building smart assistants, search tools, or analysis engines over Vietnamese document repositories.


## Getting Started <a name = "getting_started"></a>

This was built inside WSL on windows machine, make sure to install WSL before continue on windows. Mac and Linux can skip this part
### Prerequisites

Make sure you have the following installed:

```
# Python & Virtual Environment
sudo apt install python3 python3-venv python3-pip -y

# Tesseract OCR (Vietnamese language support)
sudo apt install tesseract-ocr tesseract-ocr-vie

# Ollama (for running LLMs locally)
curl -fsSL https://ollama.com/install.sh | sh

```

### Installing

A step by step series of examples that tell you how to get a development env running.

Say what the step will be

```
# Clone the repository
git clone https://github.com/Tungbeoi/Vietlexibot.git
cd vietlexicon

# Set up Python virtual environment
python3 -m venv venv
source venv/bin/activate

# Install all dependencies
pip install -r requirements.txt

# Install models
ollama pull bge-m3
ollama pull llama3:8b-instruct-q5_K_S
ollama pull hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF

# Run the program
python Vietlexibot.py
```


## Output Example <a name = "output-example"></a>
```
Chọn chế độ đầu vào:
> 1
Các tệp PDF có sẵn:
1. HỘI ĐỒNG NHÂN DÂN.pdf
2. Quyết định-1360-QĐ-TTg.pdf
3. Quyết định-1380-QĐ-TTg.pdf
4. Quyết định-1629-QĐ-UBND.pdf
5. Quyết định-2093-QĐ-BYT.pdf
6. Quyết định-2858-QĐ-UBND.pdf
Chọn số thứ tự của tệp PDF: 1
[INFO] Thời gian trích xuất thông tin: 29.46 giây
[INFO] Đã thêm 6 văn bản. Thời gian xử lý: 4.19 giây
[INFO] Tổng thời gian tiền xử lý tài liệu: 35.58 giây

Các trường được trích xuất:
- Số hiệu: 33/2022/NQ-HĐND
- Ngày ban hành: 15/07/2022
- Trích yếu: Nghị quyết quy định mức thu, chế độ thu, nộp, quản lý và sử dụng lệ phí cấp giấy phép lao động cho người nước ngoài làm việc trên địa bàn tỉnh Cao Bằng.
- Nội dung: Nghị quyết này quy định mức thu, chế độ thu, nộp, quản lý và sử dụng lệ phí cấp giấy phép lao động cho người nước ngoài làm việc trên địa bàn tỉnh Cao Bằng với mức thu 600.000 đồng/01 giấy phép mới, 450.000 đồng/01 giấy phép lại và 450.000 đồng/01 giấy phép gia hạn.

Gợi ý công việc từ tài liệu:
Dựa vào nội dung tài liệu, các công việc hoặc hành động cụ thể và rõ ràng cho các đơn vị liên quan có thể thực hiện như sau:

**Ưu tiên cao:**

- Sở Lao động - Thương binh và Xã hội tỉnh Cao Bằng:
  + Thực hiện thu lệ phí cấp mới giấy phép lao động theo quy định.
  + Thực hiện thu lệ phí cấp lại giấy phép lao động theo quy định.
  + Thực hiện thu lệ phí gia hạn giấy phép lao động theo quy định.
- Ban Quản lý Khu kinh tế tỉnh Cao Bằng:
  + Thực hiện thu lệ phí cấp mới giấy phép lao động cho người nước ngoài làm việc tại Khu kinh tế.
  + Thực hiện thu lệ phí cấp lại giấy phép lao động cho người nước ngoài làm việc tại Khu kinh tế.
  + Thực hiện thu lệ phí gia hạn giấy phép lao động cho người nước ngoài làm việc tại Khu kinh tế.

**Ưu tiên trung bình:**

- Sở Lao động - Thương binh và Xã hội tỉnh Cao Bằng:
  + Chuẩn bị và soạn thảo văn bản quy định mức thu, chế độ thu, nộp lệ phí cấp giấy phép lao động.
  + Tìm kiếm thông tin về các quy định pháp luật liên quan đến phí và lệ phí.
- Ban Quản lý Khu kinh tế tỉnh Cao Bằng:
  + Tìm hiểu về yêu cầu kỹ thuật để áp dụng công nghệ số hóa thu lệ phí.
  + Chuẩn bị và lập kế hoạch thực hiện việc số hóa thu lệ phí.

**Ưu tiên thấp:**

- Sở Lao động - Thương binh và Xã hội tỉnh Cao Bằng:
  + Nghiên cứu và xem xét các quy định pháp luật liên quan đến phí và lệ phí.
  + Đề xuất các giải pháp để nâng cao hiệu quả quản lý và sử dụng lệ phí.
- Ban Quản lý Khu kinh tế tỉnh Cao Bằng:
  + Tìm hiểu về các đề xuất của Sở Lao động - Thương binh và Xã hội.
  + Xem xét và đề xuất các giải pháp để nâng cao hiệu quả quản lý và sử dụng lệ phí.

[INFO] Thời gian xử lý gợi ý công việc: 51.01 giây

Hỏi gì đó (/bye để thoát): các ban bộ sở liên quan gì 

AI: Theo nội dung được cung cấp, các sở, ban, ngành, đoàn thể tỉnh bao gồm:

* Sở Lao động — Thương binh và Xã hội tỉnh Cao Bằng
* Ban Quản lý Khu kinh tế tỉnh Cao Bằng

Cả hai đơn vị này đều liên quan đến lĩnh vực lao động và có trách nhiệm thực hiện nghị quyết về lệ phí cấp giấy phép lao động cho người nước ngoài làm việc trên địa bàn tỉnh Cao Bằng.

Hỏi gì đó (/bye để thoát): Nghị quyết này có hiệu lực từ ngày nào?

AI: Theo nội dung được cung cấp, Nghị quyết số 33/2022/NQ-HĐND Cao Bằng có hiệu lực thi hành kể từ ngày 25 tháng 7 năm 2022.

Hỏi gì đó (/bye để thoát):Phạm vi áp dụng (địa bàn, lĩnh vực, thời gian) là gì?

AI: Theo Nghị quyết số 33/2022/NQ-HĐND ngày 15 tháng 7 năm 2022 của Hội đồng nhân dân tỉnh Cao Bằng, Phạm vi điều chỉnh của Nghị quyết này quy định mức thu, chế độ thu, nộp, quản lý và sử dụng lệ phí cấp giấy phép lao động cho người nước ngoài làm việc trên địa bàn tỉnh Cao Bằng.

Hỏi gì đó (/bye để thoát): Có quy định cụ thể về cách quản lý, sử dụng, hoặc báo cáo kết quả không?

AI: Theo Nghị quyết số 33/2022/NQ-HĐND ngày 15 tháng 7 năm 2022 của Hội đồng nhân dân tỉnh Cao Bằng, Điều 9 quy định:

"Chế độ thu, nộp, quản lý và sử dụng lệ phí:

10. a) Người nộp lệ phí thực hiện kê khai, nộp theo từng lần phát sinh.

11.b) Tổ chức thu lệ phí thực hiện nộp 100% tiền lệ phí thu được vào ngân sách nhà nước và thực hiện kê khai, nộp lệ phí theo tháng theo quy định của pháp luật quản lý thuế.

12. c) Tổ chức thu lệ phí thực hiện lập và cấp chứng từ thu lệ phí theo hướng dẫn của Bộ Tài chính."

Theo đó, người sử dụng lao động phải kê khai, nộp lệ phí theo từng lần phát sinh và tổ chức thu lệ phí phải nộp 100% tiền lệ phí thu được vào ngân sách nhà nước.


Hỏi gì đó (/bye để thoát):
```
