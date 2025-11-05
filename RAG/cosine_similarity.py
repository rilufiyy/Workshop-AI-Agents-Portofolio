from openai import OpenAI
import os 
from dotenv import load_dotenv
import numpy as np

load_dotenv()

client = OpenAI(
    api_key=os.getenv('OPENAI_API_KEY')
)

def get_embeddings(text):
    response = client.embeddings.create(
        model='text-embedding-3-small',
        input=text
    )
    
    return response.data[0].embedding

def cosine_similarity(vector1, vector2):
    vector1 = np.array(vector1)
    vector2 = np.array(vector2)
    
    dot_product = np.dot(vector1, vector2)
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)
    
    simalirity = dot_product / (magnitude1 * magnitude2)
    return simalirity

text1 = 'I love cat'
text2 = 'I love cat'

emb1 = get_embeddings(text1)
emb2 = get_embeddings(text2)

similarity = cosine_similarity(emb1, emb2)
print(f'Text 1: {text1}')
print(f'Text 2: {text2}')
print(f'Cosine Similarity: {cosine_similarity}')