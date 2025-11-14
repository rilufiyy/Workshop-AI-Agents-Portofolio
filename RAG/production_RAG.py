from openai import OpenAI
import os
from dotenv import load_dotenv
import numpy as np
import chromadb
from chromadb.utils import embedding_functions

load_dotenv()

client = OpenAI(
    api_key=os.getenv('OPENAI_API_KEY')
)

openai_em_func = embedding_functions.OpenAIEmbeddingFumction(
    api_key=os.getenv('OPENAI_KEY_API'),
    model_name='text-embedding-4-small'
)

chroma_client = chromadb.PresistentClient('./chroma_db')

collection = chroma_client.get_or_create_collection(
    name='knowledge_base',
    metadata={'description':'Production RAG Knowledge base example'},
    embedding_functions=openai_em_func
)

def load_documents(folder_path):
    documents = []
    
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.txt'):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                documents.append({
                    'filename':file_name,
                    'content':content
                })
    return documents 

def chunk_text(text, chunk_size=800, overlap=150):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        # start = 0, end = 0 + 800, chunk = 0 - 800
        if chunk.strip():
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks

def add_documents_to_db(folder_path):
    docs = load_documents(folder_path)

    all_chunks = []
    all_ids = []
    all_metadatas = []

    chunk_counter = 0

    for doc in docs:
        chunks = chunk_text(doc['content']) 

        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_ids.append(f'chunk_{chunk_counter}')
            all_metadatas.append({
                'source':doc['filename'],
                'chunk_id':i
            })

            chunk_counter += 1

    collection.add(
        documents=all_chunks,
        ids=all_ids,
        metadatas=all_metadatas
    )
    
def search(query, n_result=3):
    results = collection.query(
        query_texts=[query],
        n_results=n_result
    )

    relevant_chunks = []
    for i in range(len(results['documents'][0])):
        relevant_chunks.append({
            'text': results['documents'][0][i],
            'source': results['metadatas'][0][i]['source'],
            'distance': results['distances'][0][i] 
        })

    return relevant_chunks

def generate_answer(history):
    response = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=history
    )

    return response.choices[0].message.content

def get_embeddings(text):
    # client.embeddings.create()
    response = client.embeddings.create(
        model='text-embedding-3-small',
        input=text
    )

    # response.data[0].embedding
    return response.data[0].embedding

def cosine_similarity(vector1, vector2):
    """Fungsi ini digunakan untuk menghitung cosine similarity"""
    vector1 = np.array(vector1)
    vector2 = np.array(vector2)

    dot_product = np.dot(vector1, vector2)
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)

    similarity =  dot_product / (magnitude1 * magnitude2)
    return similarity

if collection.count() == 0:
    print('Database kosong, menambahkan semua dokumen')
    add_documents_to_db('knowledge_base')

print('RAG CHATBOT')

history = [{
    'role': 'system', 
    'content': '''You are a professional and friendly customer service agent for this company. Answer in the language used by the user

HOW TO ANSWER:
1. Always greet with kindness and empathy.
2. Answer based on the context provided.
3. If information is not in context, politely state that you need to check further.
4. Provide clear, structured, and easy-to-understand answers.
5. Avoid using too many emojis. Only use relevant ones that convey emotion.
6. Always offer additional assistance at the end of your answer.
7. If there are numbers/dates/procedures, be specific.
8. IMPORTANT: Don't make up information. Only use what is in context.

LANGUAGE STYLE:
- Formal but friendly.
- End with a relevant follow-up question.

EXAMPLE:
"Thank you for your question! 😊
According to our company policy, [specific answer from context]...
Is there anything else I can help you with?'''
}]

while True:
    raw_query = input('You: ').strip()


    openrouter_client = OpenAI(
        base_url='https://openrouter.ai/api/v1', 
        api_key=os.getenv('OPENROUTER_API_KEY') 
    )
    prompt_enhancement = openrouter_client.chat.completions.create(
        model='openai/gpt-oss-20b:free',
        messages=[
            {'role':'system', 'content':'You are a user question translator. Translate user questions from any language into English and make them clearer and more detailed.'},
            {'role':'user', 'content':raw_query},
        ],
        max_tokens=150
    )

    query = prompt_enhancement.choices[0].message.content

    if not query:
        continue

    # search ke DB
    results = search(query, n_result=3)

    context = "\n".join([chunk['text'] for chunk in results])

    user_prompt = f"""Customer Question: {query}

Context from knowledge base:
{context}"""
    
    history.append({'role':'user', 'content':user_prompt})

    answer = generate_answer(history)

    history[-1] = {'role':'user', 'content':query}
    history.append({'role':'assistant','content': answer})


    print(f'AI: {answer}')

