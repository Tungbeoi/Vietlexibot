import ollama
# Load the dataset
import re
def split_chunks_with_context(text, max_chars=1000):

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

with open('RAR_test.txt', 'r', encoding='utf-8') as f:
    dataset = f.read()

chunks = split_chunks_with_context(dataset)

for i, chunk in enumerate(chunks):
    print(f"--- Chunk {i+1} ---\n{chunk}\n")


# Implement the retrieval system

EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-large-en-v1.5-gguf'
LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF'

# Each element in the VECTOR_DB will be a tuple (chunk, embedding)
# The embedding is a list of floats, for example: [0.1, 0.04, -0.34, 0.21, ...]
VECTOR_DB = []

def add_chunk_to_database(chunk):
  embedding = ollama.embed(model=EMBEDDING_MODEL, input=chunk)['embeddings'][0]
  VECTOR_DB.append((chunk, embedding))

for i, chunk in enumerate(chunks):
    try:
        add_chunk_to_database(chunk)
        print(f'Added chunk {i+1}/{len(chunks)} to the database')
    except Exception as e:
        print(f'Failed to embed chunk {i+1}: {e}')

def cosine_similarity(a, b):
  dot_product = sum([x * y for x, y in zip(a, b)])
  norm_a = sum([x ** 2 for x in a]) ** 0.5
  norm_b = sum([x ** 2 for x in b]) ** 0.5
  return dot_product / (norm_a * norm_b)

def retrieve(query, top_n=6):
  query_embedding = ollama.embed(model=EMBEDDING_MODEL, input=query)['embeddings'][0]
  # temporary list to store (chunk, similarity) pairs
  similarities = []
  for chunk, embedding in VECTOR_DB:
    similarity = cosine_similarity(query_embedding, embedding)
    similarities.append((chunk, similarity))
  # sort by similarity in descending order, because higher similarity means more relevant chunks
  similarities.sort(key=lambda x: x[1], reverse=True)
  # finally, return the top N most relevant chunks
  return similarities[:top_n]



# Chatbot
while True:
  input_query = input('\nTôi có thể giúp gì?: ')
  if input_query.strip().lower() == '/exit':
        print("Bye!")
        break
  retrieved_knowledge = retrieve(input_query)
  
  print('Retrieved knowledge:')
  for chunk, similarity in retrieved_knowledge:
    print(f' - (similarity: {similarity:.2f}) {chunk}\n')

  instruction_prompt = f'''You are a helpful vietnamese chat agent specialized in RAR-EP application.
  Always use only the following pieces of context to answer the question. Focus on informations that has the same number and has (next) and answer only. Compare which info matched. 
  forbiden from creating new information outside of provided information, do not provide answer if information is outside the knowledge provided:
  {'\n'.join([f'{chunk}' for chunk, similarity in retrieved_knowledge])}
  '''
  #print(instruction_prompt)

  stream = ollama.chat(
    model=LANGUAGE_MODEL,
    messages=[
      {'role': 'system', 'content': instruction_prompt},
      {'role': 'user', 'content': input_query},
    ],
    stream=True,
  )

# print the response from the chatbot in real-time
  print('Chatbot response:')
  for chunk in stream:
    print(chunk['message']['content'], end='', flush=True)