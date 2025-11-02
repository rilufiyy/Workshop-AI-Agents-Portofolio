import os
import json
import time
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  

client = OpenAI(
    base_url='https://openrouter.ai/api/v1',   
    api_key=os.getenv('OPENROUTER_API_KEY') 
)

MODEL_NAME = "openai/gpt-4o-mini" 

chat_history = [
    { "role": "system", "content": "You are a helpful AI assistant for beginners. Jawab dengan bahasa sederhana, ramah, dan jangan terlalu teknis kalau tidak diminta."}
]

stream_mode = True 


def save_history_to_file(filename="chat_history.json"):
    data_to_save = chat_history[1:] if len(chat_history) > 1 else []

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data_to_save, f, indent=2, ensure_ascii=False)

    print(f"\n[System] Chat history saved to {filename}\n")


def get_response_normal(messages):

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.7,
    )

    ai_message = response.choices[0].message.content
    return ai_message

def get_response_stream(messages):
    full_answer = ""

    stream = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.7,
        stream=True
    )

    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            content = chunk.choices[0].delta.content
            print(content, end='', flush=True)
            full_answer += content


        elif chunk.type == "message.completed":
            print() 
        elif chunk.type == "error":
            print(f"\n[Error] {chunk.error}")

    return full_answer

def main():
    global stream_mode

    print(" Welcome to Your First AI Chatbot on Terminal\n")
    print("Commands:")
    print("  /stream on   -> aktifkan streaming mode")
    print("  /stream off  -> matikan streaming mode (jawab langsung)")
    print("  /save        -> simpan riwayat chat ke chat_history.json")
    print("  /exit        -> keluar\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() == "/exit":
            print("AI: Byee 👋")
            break

        if user_input.lower() == "/save":
            save_history_to_file()
            continue

        if user_input.lower() == "/stream on":
            stream_mode = True
            print("[System] Streaming mode: ON (jawaban akan muncul real-time)\n")
            continue

        if user_input.lower() == "/stream off":
            stream_mode = False
            print("[System] Streaming mode: OFF (jawaban akan muncul sekaligus)\n")
            continue

        if not user_input:
            print("[System] (kosong, ketik sesuatu atau /exit)")
            continue

    

        chat_history.append({
            "role": "user",
            "content": user_input
        })

        print("AI: ", end="", flush=True)

        if stream_mode:
            ai_reply = get_response_stream(chat_history)
        else:
            ai_reply = get_response_normal(chat_history)
            print(ai_reply)  

        chat_history.append({
            "role": "assistant",
            "content": ai_reply
        })

        print()  


if __name__ == "__main__":
    main()