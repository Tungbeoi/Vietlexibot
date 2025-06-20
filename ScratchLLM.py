import os
import fitz  # PyMuPDF
import ollama
import re
import logging
from datetime import datetime
import traceback

# === CONFIGURATION ===
EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-large-en-v1.5-gguf'
LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF'
VECTOR_DB = []

# === LOGGING SETUP ===
def setup_logging():
    """Setup logging configuration for conversation and error tracking"""
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Setup conversation logger
    conversation_logger = logging.getLogger('conversation')
    conversation_logger.setLevel(logging.INFO)
    conversation_logger.handlers.clear()  # Clear any existing handlers
    
    # Create conversation log handler
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    conv_handler = logging.FileHandler(f'logs/conversation_{timestamp}.txt', encoding='utf-8', errors='replace')
    conv_formatter = logging.Formatter('%(asctime)s - %(message)s')
    conv_handler.setFormatter(conv_formatter)
    conversation_logger.addHandler(conv_handler)
    
    # Setup error logger
    error_logger = logging.getLogger('error')
    error_logger.setLevel(logging.ERROR)
    error_logger.handlers.clear()  # Clear any existing handlers
    
    # Create error log handler
    error_handler = logging.FileHandler(f'logs/errors_{timestamp}.txt', encoding='utf-8', errors='replace')
    error_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s\n%(exc_info)s\n')
    error_handler.setFormatter(error_formatter)
    error_logger.addHandler(error_handler)
    
    return conversation_logger, error_logger

def clean_text_for_logging(text):
    """Clean text to handle Unicode issues before logging"""
    if not text:
        return ""
    
    try:
        # Handle potential Unicode issues
        if isinstance(text, bytes):
            text = text.decode('utf-8', errors='replace')
        
        # Remove or replace problematic characters
        cleaned_text = text.encode('utf-8', errors='replace').decode('utf-8')
        
        # Remove any remaining surrogate characters
        cleaned_text = ''.join(char for char in cleaned_text if not (0xD800 <= ord(char) <= 0xDFFF))
        
        return cleaned_text
    except Exception as e:
        return f"[Text encoding error: {str(e)}]"

def log_conversation(logger, user_query, bot_response):
    """Log conversation exchange with proper Unicode handling"""
    try:
        clean_user_query = clean_text_for_logging(user_query)
        clean_bot_response = clean_text_for_logging(bot_response)
        
        logger.info(f"USER: {clean_user_query}")
        logger.info(f"BOT: {clean_bot_response}")
        logger.info("-" * 80)
    except Exception as e:
        # Fallback logging if there are still issues
        try:
            logger.info(f"USER: [Unicode error in user query: {str(e)}]")
            logger.info(f"BOT: [Unicode error in bot response: {str(e)}]")
            logger.info("-" * 80)
        except:
            pass  # If even this fails, just skip logging this exchange

def log_error(logger, error_msg, exception=None):
    """Log error with optional exception details and proper Unicode handling"""
    try:
        clean_error_msg = clean_text_for_logging(error_msg)
        if exception:
            clean_exception = clean_text_for_logging(str(exception))
            logger.error(f"{clean_error_msg}: {clean_exception}\nTraceback: {traceback.format_exc()}")
        else:
            logger.error(clean_error_msg)
    except Exception as e:
        # Fallback error logging
        try:
            logger.error(f"Error logging failed: {str(e)}")
        except:
            pass  # If even this fails, just skip logging this error

# === PDF UTILITIES ===
def list_pdf_files():
    return [f for f in os.listdir('.') if f.lower().endswith('.pdf')]

def select_pdf_file(pdfs):
    print("Các tệp PDF có sẵn:")
    for i, f in enumerate(pdfs):
        print(f"{i + 1}. {f}")
    while True:
        try:
            choice = int(input("Chọn số thứ tự của tệp PDF: "))
            if 1 <= choice <= len(pdfs):
                return pdfs[choice - 1]
            else:
                print("Lựa chọn không hợp lệ. Vui lòng thử lại.")
        except ValueError:
            print("Vui lòng nhập một số.")

def validate_pdf_file(pdf_path, error_logger):
    """Validate if PDF file is readable and not corrupted"""
    try:
        # Try to open the PDF file
        doc = fitz.open(pdf_path)
        
        # Check if document has pages
        if doc.page_count == 0:
            error_msg = f"PDF file '{pdf_path}' appears to be empty (0 pages)"
            log_error(error_logger, error_msg)
            doc.close()
            return False, error_msg
        
        # Try to access first page to check for corruption
        try:
            first_page = doc[0]
            # Try to get text from first page as a basic corruption test
            _ = first_page.get_text()
        except Exception as page_error:
            error_msg = f"PDF file '{pdf_path}' appears to be corrupted - cannot read pages"
            log_error(error_logger, error_msg, page_error)
            doc.close()
            return False, error_msg
        
        doc.close()
        return True, "PDF file is valid"
        
    except fitz.FileDataError as e:
        error_msg = f"PDF file '{pdf_path}' is corrupted or has invalid format"
        log_error(error_logger, error_msg, e)
        return False, error_msg
    
    except fitz.FileNotFoundError as e:
        error_msg = f"PDF file '{pdf_path}' not found"
        log_error(error_logger, error_msg, e)
        return False, error_msg
    
    except PermissionError as e:
        error_msg = f"Permission denied to access PDF file '{pdf_path}'"
        log_error(error_logger, error_msg, e)
        return False, error_msg
    
    except Exception as e:
        error_msg = f"Unexpected error while validating PDF file '{pdf_path}'"
        log_error(error_logger, error_msg, e)
        return False, error_msg

def extract_text_from_pdf(pdf_path, mode='text', error_logger=None):
    """Extract text from PDF with comprehensive error handling"""
    try:
        if mode == 'table':
            return extract_table_text_from_pdf(pdf_path, error_logger)
        else:
            text = ""
            with fitz.open(pdf_path) as doc:
                for page_num in range(doc.page_count):
                    try:
                        page = doc[page_num]
                        page_text = page.get_text()
                        text += page_text
                    except Exception as e:
                        error_msg = f"Error extracting text from page {page_num + 1} of '{pdf_path}'"
                        if error_logger:
                            log_error(error_logger, error_msg, e)
                        print(f"Cảnh báo: {error_msg} - Bỏ qua trang này")
                        continue
            
            if not text.strip():
                error_msg = f"No text could be extracted from PDF '{pdf_path}' - file may be image-based or corrupted"
                if error_logger:
                    log_error(error_logger, error_msg)
                raise ValueError(error_msg)
            
            return text
            
    except Exception as e:
        error_msg = f"Failed to extract text from PDF '{pdf_path}'"
        if error_logger:
            log_error(error_logger, error_msg, e)
        raise

def extract_table_text_from_pdf(pdf_path, error_logger=None):
    """Extract table text from PDF with error handling"""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        row_count = 0
        header = ""
        
        for page_num in range(doc.page_count):
            try:
                page = doc[page_num]
                tables = page.find_tables()
                
                for table_num, table in enumerate(tables):
                    try:
                        if page_num == 0 and table.header.external:
                            header = ";".join([name if name else "" for name in table.header.names]) + "\n"
                            text += header
                            row_count += 1
                        
                        for row in table.extract():
                            row_text = ";".join([cell if cell else "" for cell in row]) + "\n"
                            if row_text != header:
                                text += row_text
                                row_count += 1
                    except Exception as e:
                        error_msg = f"Error extracting table {table_num + 1} from page {page_num + 1}"
                        if error_logger:
                            log_error(error_logger, error_msg, e)
                        print(f"Cảnh báo: {error_msg} - Bỏ qua bảng này")
                        continue
                        
            except Exception as e:
                error_msg = f"Error processing page {page_num + 1} for table extraction"
                if error_logger:
                    log_error(error_logger, error_msg, e)
                print(f"Cảnh báo: {error_msg} - Bỏ qua bảng này")
                continue
        
        doc.close()
        
        if row_count == 0:
            error_msg = f"No table data could be extracted from PDF '{pdf_path}'"
            if error_logger:
                log_error(error_logger, error_msg)
            raise ValueError(error_msg)
        
        print(f"Loaded {row_count} table rows from file '{pdf_path}'.\n")
        return text
        
    except Exception as e:
        error_msg = f"Failed to extract table text from PDF '{pdf_path}'"
        if error_logger:
            log_error(error_logger, error_msg, e)
        raise

# === TEXT CHUNKING & EMBEDDING ===
def split_chunks_with_context(text, max_chars=1200, paragraph_mode=False):
    if paragraph_mode:
        paragraphs = re.split(r'\n\s*\n', text)
        chunks = []
        current_chunk = ""
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            while len(para) > max_chars:
                split_idx = para[:max_chars].rfind('\n')
                if split_idx == -1 or split_idx < max_chars * 0.5:
                    split_idx = max_chars
                chunk_part = para[:split_idx].strip()
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                chunks.append(chunk_part)
                para = para[split_idx:].strip()
            if len(current_chunk) + len(para) + 2 > max_chars:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para
            else:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks
    else:
        pattern = r'(?=^\d+(\.\d+)*\.\s.*$)'
        raw_sections = re.split(pattern, text, flags=re.MULTILINE)
        clean_sections = [c.strip() for c in raw_sections if c]
        final_chunks = []
        for section in clean_sections:
            lines = section.split('\n')
            heading = lines[0]
            is_heading_present = re.match(r'^\d+(\.\d+)*\.\s.*$', heading)
            current_chunk = section
            while len(current_chunk) > max_chars:
                split_idx = current_chunk[:max_chars].rfind('\n\n')
                if split_idx == -1:
                    split_idx = current_chunk[:max_chars].rfind('\n')
                if split_idx < 500:
                    split_idx = max_chars
                final_chunks.append(current_chunk[:split_idx].strip())
                remaining_text = current_chunk[split_idx:].strip()
                if is_heading_present:
                    current_chunk = f"{heading} (next)\n...\n{remaining_text}"
                else:
                    current_chunk = f"(next)\n...\n{remaining_text}"
            final_chunks.append(current_chunk.strip())
        return final_chunks

def add_chunk_to_database(chunk, error_logger=None):
    """Add chunk to database with error handling"""
    try:
        embedding = ollama.embed(model=EMBEDDING_MODEL, input=chunk)['embeddings'][0]
        VECTOR_DB.append((chunk, embedding))
        return True
    except Exception as e:
        error_msg = f"Failed to embed chunk (length: {len(chunk)})"
        if error_logger:
            log_error(error_logger, error_msg, e)
        print(f"Error: {error_msg}")
        return False

def cosine_similarity(a, b):
    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x ** 2 for x in a) ** 0.5
    norm_b = sum(x ** 2 for x in b) ** 0.5
    return dot_product / (norm_a * norm_b)

def retrieve(query, top_n=8, error_logger=None):
    """Retrieve relevant chunks with error handling"""
    try:
        query_embedding = ollama.embed(model=EMBEDDING_MODEL, input=query)['embeddings'][0]
        similarities = [(chunk, cosine_similarity(query_embedding, emb)) for chunk, emb in VECTOR_DB]
        return sorted(similarities, key=lambda x: x[1], reverse=True)[:top_n]
    except Exception as e:
        error_msg = f"Failed to retrieve relevant chunks for query: '{query}'"
        if error_logger:
            log_error(error_logger, error_msg, e)
        print(f"Error: {error_msg}")
        return []

# === LOGIC OF THOUGHTS (LoT) MODULE ===
def logic_extraction(context_chunks, user_query, error_logger=None):
    """Use the LLM to extract logical relations and propositions from the context."""
    context_text = "\n".join([chunk for chunk, _ in context_chunks])
    
    prompt = f"""Based on the context below, identify the key facts and relationships that are relevant to answering this question: "{user_query}"

Context:
{context_text[:2000]}

Please list the main facts and logical connections you can identify:"""

    try:
        #print(f"[DEBUG] Sending logic extraction prompt to model: {LANGUAGE_MODEL}")
        #print(f"[DEBUG] Prompt length: {len(prompt)} characters")
        
        response = ollama.chat(
            model=LANGUAGE_MODEL,
            messages=[{'role': 'user', 'content': prompt}],
            stream=False
        )
        
        #print(f"[DEBUG] Raw response object: {response}")
        
        if 'message' in response and 'content' in response['message']:
            content = response['message']['content'].strip()
            #print(f"[DEBUG] Extracted content length: {len(content)}")
            return content if content else "No logical propositions extracted."
        else:
            #print("[DEBUG] Unexpected response format")
            return "Error: Unexpected response format from model."
            
    except Exception as e:
        error_msg = f"Error in logic_extraction for query: '{user_query}'"
        if error_logger:
            log_error(error_logger, error_msg, e)
        #print(f"[DEBUG] Error in logic_extraction: {e}")
        return f"Error in logic extraction: {str(e)}"

def logic_extension(logic_statements, error_logger=None):
    """Use the LLM to expand the logical statements using formal rules."""
    if not logic_statements or logic_statements.startswith("Error") or logic_statements.startswith("No logical"):
        return "No logic to extend."
    
    prompt = f"""Given these facts and relationships, what additional conclusions can we draw using basic logical reasoning?

Original facts:
{logic_statements}

What additional conclusions follow logically from these facts?"""

    try:
        #print(f"[DEBUG] Sending logic extension prompt")
        
        response = ollama.chat(
            model=LANGUAGE_MODEL,
            messages=[{'role': 'user', 'content': prompt}],
            stream=False
        )
        
        #print(f"[DEBUG] Logic extension response: {response}")
        
        if 'message' in response and 'content' in response['message']:
            content = response['message']['content'].strip()
            return content if content else "No additional logic derived."
        else:
            return "Error: Unexpected response format from model."
            
    except Exception as e:
        error_msg = "Error in logic_extension"
        if error_logger:
            log_error(error_logger, error_msg, e)
        #print(f"[DEBUG] Error in logic_extension: {e}")
        return f"Error in logic extension: {str(e)}"

def logic_translation(extended_logic, error_logger=None):
    """Translate the formal logical consequences back into clear, natural language."""
    if not extended_logic or extended_logic.startswith("Error") or extended_logic.startswith("No"):
        return "No logic to translate."
    
    prompt = f"""Please rephrase the following information in clear, simple Vietnamese:

{extended_logic}

Vietnamese translation:"""

    try:
        #print(f"[DEBUG] Sending logic translation prompt")
        
        response = ollama.chat(
            model=LANGUAGE_MODEL,
            messages=[{'role': 'user', 'content': prompt}],
            stream=False
        )
        
        #print(f"[DEBUG] Logic translation response: {response}")
        
        if 'message' in response and 'content' in response['message']:
            content = response['message']['content'].strip()
            return content if content else "No translation available."
        else:
            return "Error: Unexpected response format from model."
            
    except Exception as e:
        error_msg = "Error in logic_translation"
        if error_logger:
            log_error(error_logger, error_msg, e)
        #print(f"[DEBUG] Error in logic_translation: {e}")
        return f"Error in logic translation: {str(e)}"

def lot_augmented_answer(context_chunks, user_query, error_logger=None):
    """Generate augmented answer using Logic of Thoughts with error handling"""
    bot_response = ""
    
    try:
        # 1. Logic Extraction
        extracted_logic = logic_extraction(context_chunks, user_query, error_logger)
        
        # 2. Logic Extension  
        extended_logic = logic_extension(extracted_logic, error_logger)
        
        # 3. Logic Translation
        translated_logic = logic_translation(extended_logic, error_logger)
        
        # 4. Compose augmented instruction for final answer
        original_context = "\n".join([chunk for chunk, _ in context_chunks])
        instruction_prompt = f"""You are an AI assistant that responds in Vietnamese based on the provided knowledge.
Use the original information and any extended logical inferences to answer the question. Do not answer unrelated 

Original Information:
{original_context[:3000]}

Extended Logical Inferences:
{translated_logic[:3000]}

Question: {user_query}

Please provide a comprehensive answer in Vietnamese:"""

        stream = ollama.chat(
            model=LANGUAGE_MODEL,
            messages=[{'role': 'user', 'content': instruction_prompt}],
            stream=True,
        )
        print('\n\nChatbot response:')
        for chunk in stream:
            content = chunk['message']['content']
            print(content, end='', flush=True)
            bot_response += content
        
        print()  # Add newline after response
            
    except Exception as e:
        error_msg = f"Error in final response generation for query: '{user_query}'"
        if error_logger:
            log_error(error_logger, error_msg, e)
        bot_response = f"Xin lỗi, đã xảy ra lỗi khi tạo câu trả lời: {str(e)}"
        print(f"\n{bot_response}")
    
    return bot_response

# === MAIN EXECUTION ===
def main():
    # Setup logging
    conversation_logger, error_logger = setup_logging()
    print(f"Đã tạo logs, xem thư mục 'logs' để theo dõi cuộc trò chuyện và lỗi.")
    
    try:
        pdf_files = list_pdf_files()
        if not pdf_files:
            error_msg = "No PDF files found in the current directory."
            print(error_msg)
            log_error(error_logger, error_msg)
            return

        selected_pdf = select_pdf_file(pdf_files)
        
        # Validate PDF file before processing
        is_valid, validation_msg = validate_pdf_file(selected_pdf, error_logger)
        if not is_valid:
            print(f"Error: {validation_msg}")
            print("Vui lòng chọn tệp PDF khác hoặc sửa tệp bị lỗi.")
            return
        
        mode = ''
        while mode not in ['text', 'table']:
            mode = input("Chọn chế độ trích xuất ('text' hoặc 'table'): ").strip().lower()
        
        para_mode = ''
        while para_mode not in ['p', 'n']:
            para_mode = input("Chọn chế độ trích xuất paragraph|number_heading(2.2, 3.3.2, etc)? (p/n): ").strip().lower()
        use_paragraph = True if para_mode == 'p' else False

        print(f"\nĐang trích xuất từ: {selected_pdf} với chế độ: {mode}")
        
        try:
            dataset = extract_text_from_pdf(selected_pdf, mode=mode, error_logger=error_logger)
        except Exception as e:
            print(f"Failed to extract text from PDF: {e}")
            return

        if mode == 'table':
            chunks = [line.strip() for line in dataset.split('\n') if line.strip()]
        else:
            chunks = split_chunks_with_context(dataset, paragraph_mode=use_paragraph)

        print(f"Tổng số đoạn văn bản xử lý: {len(chunks)}")
        successful_chunks = 0
        
        for i, chunk in enumerate(chunks):
            #print(f"\nProcessing chunk {i+1}/{len(chunks)} (length {len(chunk)}):\n{chunk}\n{'-'*40}")
            
            if add_chunk_to_database(chunk, error_logger):
                successful_chunks += 1
                #print(f'Added chunk {i + 1}/{len(chunks)} to the database')
            else:
                print(f'Không thể thêm đoạn văn bản vào Agent{i + 1}/{len(chunks)}')

        if successful_chunks == 0:
            error_msg = "No chunks were successfully embedded. Cannot proceed with Q&A."
            print(f"Error: {error_msg}")
            log_error(error_logger, error_msg)
            return

        print(f"Thêm thành công {successful_chunks}/{len(chunks)} đoạn văn bản.")

        # Main conversation loop
        while True:
            input_query = input('\n\nTôi có thể giúp gì? (/bye để thoát): ')
            
            if input_query.strip().lower() == '/bye':
                farewell_msg = "Session ended"
                print("Bye!")
                log_conversation(conversation_logger, input_query, farewell_msg)
                break
            
            try:
                retrieved_knowledge = retrieve(input_query, error_logger=error_logger)

                if not retrieved_knowledge:
                    bot_response = "Xin lỗi, không thể tìm thấy thông tin liên quan để trả lời câu hỏi của bạn."
                    print(f"\n\n{bot_response}")
                    log_conversation(conversation_logger, input_query, bot_response)
                    continue

                #print('\n\nRetrieved knowledge:\n')
                #for chunk, similarity in retrieved_knowledge:
                    #print(f' - (similarity: {similarity:.4f}) {chunk}\n')

                # LoT pipeline answer
                bot_response = lot_augmented_answer(retrieved_knowledge, input_query, error_logger)
                
                # Log the conversation
                log_conversation(conversation_logger, input_query, bot_response)
                
            except Exception as e:
                error_msg = f"Error processing query: '{input_query}'"
                log_error(error_logger, error_msg, e)
                bot_response = f"Xin lỗi, đã xảy ra lỗi khi xử lý câu hỏi của bạn: {str(e)}"
                print(f"\n\n{bot_response}")
                log_conversation(conversation_logger, input_query, bot_response)

    except Exception as e:
        error_msg = "Critical error in main execution"
        log_error(error_logger, error_msg, e)
        print(f"Critical error: {e}")

if __name__ == "__main__":
    main()