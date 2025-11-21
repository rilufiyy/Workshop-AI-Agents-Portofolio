from openai import OpenAI
import os
import json
from dotenv import load_dotenv
import requests
from ddgs import DDGS
import trafilatura


load_dotenv()

client = OpenAI(
    api_key=os.getenv('OPENAI_API_KEY')
)

MODEL = 'gpt-4o-mini'

def web_search(query_text: str, maximum_results: int = 5):
    search_results = []
    with DDGS() as search_engine:
        for result in search_engine.text(query_text, maximum_results=maximum_results):
            search_results.append({
                'title': result.get('title'),
                'url': result.get('href'),
                'snippet': result.get('body'),
                'source':'duckduckgo'
            })

    return search_results

def fetch_webpage_content(url: str, max_character: int = 4000):
    try:
        response = requests.get(
            url,
            timeout=15,
            headers={'User-Agent': 'Mozilla/5.0 (compatible: WebResearchBot/1.0)'}
        )
        response.raise_for_status()

        extracted_text = trafilatura.extract(
            response.text,
            include_comments=False,
            include_tables=False
        )

        if not extracted_text:
            return {'url': url, 'ok':False, 'reason':'Gagal mengekstrak teks dari webpage.'}
        
        clean_text = extracted_text.strip()
        if len(clean_text) > max_character: 
            clean_text = clean_text[:max_character]

        return {'url':url, 'ok':True, 'text':clean_text}

    except Exception as e:
        return {'url': url, 'ok':False, 'reason':f'Tidak bisa fetch: {e}'}
    

TOOLS = [
    {
        'type': 'function',
        'function': {
            'name': 'web_search',
            'description': 'Mencari informasi terkini di internet',
            'parameters': {
                'type': 'object',
                'properties': {
                    'query_text':{
                        'type':'string',
                        'description': 'Kata kunci pencarian'
                    },
                    'maximum_results':{
                        'type':'integer',
                        'minimum': 1,
                        'maximum':10,
                        'default': 5,
                        'description':'Seberapa banyak jumlah pencarian'
                    }
                },
                'required':['query_text']
            }
        }
    },
    {
        'type': 'function',
        'function': {
            'name': 'fetch_webpage_content',
            'description': 'Ekstrak konten dalam halaman website sesuai URL.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'url':{
                        'type':'string',
                        'description': 'Link website yang dituju'
                    },
                    'max_character':{
                        "type": "integer",
                        "minimum": 500,
                        "maximum": 20000,
                        "default": 4000,
                        'description':'Seberapa banyak jumlah character content'
                    }
                },
                'required':['url']
            }
        }
    }
]


PYTHON_FUNCTION = {
    'web_search':web_search,
    'fetch_webpage_content': fetch_webpage_content
}


def call_llm(message_history):
    response = client.chat.completions.create(
        model=MODEL,
        messages=message_history,
        tools=TOOLS,
        tool_choice='auto',
    )
    return response # model mau pakai tools apa response.final_reason

# response.choices[0]:
# {
#     "finish_reason": "tool_calls",  # ← Ini tanda LLM mau pakai tool
#     "message": {
#         "role": "assistant",
#         "content": None,  # ← Tidak ada jawaban text
#         "tool_calls": [
#             {
#                 "id": "call_abc123",  # ← ID unik untuk tool call ini
#                 "type": "function",
#                 "function": {
#                     "name": "web_search",  # ← Tool yang dipilih
#                     "arguments": '{"query_text": "harga coklat MrBeast", "maximum_results": 5}'
#                 }
#             }
#         ]
#     }
# }

def execute_tool_calls(message_history, assistant_message, tool_calls):
    message_history.append({
        'role':'assistant',
        'content':assistant_message.content,
        'tool_calls': [
            {
                'id': call.id,
                'type':'function',
                'function':{
                    'name': call.function.name,
                    'arguments': call.function.arguments
                }
            }
            
            for call in tool_calls
        ]
    })

    for call in tool_calls:
        tool_name = call.function.name
        tool_arguments = json.loads(call.function.arguments or '{}')
        python_function = PYTHON_FUNCTION.get(tool_name)
        result = python_function(**tool_arguments) # -> url, max_char -> {url: , max_char: }

        message_history.append({
            'role':'tool',
            'tool_call_id': call.id,
            'name':tool_name,
            'content': json.dumps(result, ensure_ascii=False)
        })


    return message_history


def start_chat_loop():
    print("\n=== Chatbot Web Research ===")
    print("Ketik 'exit' untuk keluar.\n")

    message_history = [
        {
            "role": "system",
            "content": (
                "Kamu adalah asisten riset yang sangat membantu. "
                "Jika kamu tidak tahu sesuatu, gunakan tools web_search dan fetch_webpage_content "
                "untuk mencari informasi terkini di internet, lalu berikan jawaban lengkap dengan sumbernya."
            )
        }
    ]

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("AI: Sampai jumpa!")
            break

        message_history.append({"role": "user", "content": user_input})

        # Panggilan pertama ke LLM
        response = call_llm(message_history)
        response_choice = response.choices[0]

        # Jika model memanggil tool, eksekusi dulu
        while response_choice.finish_reason == "tool_calls" and response_choice.message.tool_calls:
            # Pass assistant_message dan tool_calls
            message_history = execute_tool_calls(
                message_history, 
                response_choice.message,
                response_choice.message.tool_calls
            )
            response = call_llm(message_history)
            response_choice = response.choices[0]

        assistant_reply = response_choice.message.content or ""
        print(f"AI: {assistant_reply}\n")

        message_history.append({"role": "assistant", "content": assistant_reply})

if __name__ == "__main__":
    start_chat_loop()
