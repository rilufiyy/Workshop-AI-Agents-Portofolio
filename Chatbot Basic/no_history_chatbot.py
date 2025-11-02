from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(
    api_key=os.getenv('OPENAI_API_KEY')
)

SYSTEM_PROMPT = "You're a helpful assistant"
MAX_HISTORY = 12 * 2 + 1

history = [
    {'role':'system', 'content': SYSTEM_PROMPT}
]

while True:
    user_chat = input('You: ')

    # /exit
    if user_chat == '/exit':
        break

    history.append({'role':'user', 'content':user_chat})

    # client.chat.completions.create()
    response = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=history
    )

    history.append({'role':'assistant', 'content': response.choices[0].message.content})

    # if len(history) > MAX_HISTORY:
    #     # summarize
    #     response = client.chat.completions.create(
    #         model='gpt-4o-mini',
    #         messages=[]# tolong summrry history ini
    #     )
    #     new_history = history[0]
    #     history = new_history

    print(f"AI: {response.choices[0].message.content}")