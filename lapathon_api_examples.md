# Developer Guide: Using the AI Models

This guide provides Python code examples for connecting to our hosted LLM API.

**Prerequisites:**
* Install the OpenAI Python library: `pip install openai`
* Obtain your Team API Key (`sk-...`) from the administrator.

## 1. General Chat (Lapa & Mamay)

Use this for standard instructions, reasoning, and chat.

**Available Models:**
* `lapa` (LapaLLM, 12B Instruct)
* `mamay` (MamayLM, 12B Gemma-3 Instruct)

```python
from openai import OpenAI

client = OpenAI(
    api_key="YOUR_TEAM_API_KEY",
    base_url="http://146.59.127.106:4000"
)

response_lapa = client.chat.completions.create(
    model="lapa",
    messages=[
        {"role": "user", "content": "Хто тримає цей район?"}
    ],
    temperature=0.7,
    max_tokens=1000
)

print("LapaLLM:", response_lapa.choices[0].message.content)

response_mamay = client.chat.completions.create(
    model="mamay",
    messages=[
        {"role": "user", "content": "Хто тримає цей район?"}
    ],
    temperature=0.7,
    max_tokens=1000
)

print("MamayLM:", response_mamay.choices[0].message.content)

```

## 2. Function Calling (Lapa LoRA)

Use this when you need the model to use tools or output structured data. You **must** specify the model name `lapa-function-calling`.

Приклад виклику моделі з вашим prompt для режиму function calling:

```python
from openai import OpenAI

client = OpenAI(
    api_key="YOUR_TEAM_API_KEY",
    base_url="http://146.59.127.106:4000"
)

prompt = """Ти модель штучного інтелекту з викликом функцій. Тобі надаються підписи функцій всередині тегів  XML. Ти можеш викликати одну або кілька функцій, щоб допомогти з запитом користувача. Не роби припущень щодо значень, які потрібно підставляти у функції.
Ось доступні інструменти: 
[{'type': 'function', 'function': {'name': 'search_internet', 'description': 'Пошук інформації в інтернеті', 'parameters': {'type': 'object', 'properties': {'query': {'type': 'string', 'description': 'Запит користувача на пошук'}, 'browser': {'type': 'string', 'description': 'браузер для використання'}}, 'required': []}}}, {'type': 'function', 'function': {'name': 'get_definition', 'description': 'Отримати визначення слова', 'parameters': {'type': 'object', 'properties': {'word': {'type': 'string', 'description': 'Слово, для якого отримати визначення'}}, 'required': ['word']}}}]

Використовуй наступну pydantic model json schema для кожного виклику інструменту, який ти зробиш:
{'title': 'FunctionCall', 'type': 'object', 'properties': {'arguments': {'title': 'Arguments', 'type': 'object'}, 'name': {'title': 'Name', 'type': 'string'}}, 'required': ['arguments', 'name']}
Для кожного виклику функції повертай JSON-об'єкт з ім'ям функції та аргументами всередині XML-тегів  таким чином:

{tool_call}

Також, перед тим як зробити виклик функції, витрать час на спланування виклику - помісти твої думки між тегами <think>{your thoughts}</think>.

Виклич пошук "funny dogs" в інтернеті з використанням Google. Я викличу функцію "search_internet" з такими параметрами:
"""

response = client.chat.completions.create(
    model="lapa-function-calling",
    messages=[{"role": "user", "content": prompt}],
    temperature=0
)

print(response.choices[0].message.content)
```

## 3. Embeddings (Qwen)

**Model:** `text-embedding-qwen`

```python
from openai import OpenAI

client = OpenAI(
    api_key="YOUR_TEAM_API_KEY",
    base_url="http://146.59.127.106:4000"
)

response = client.embeddings.create(
    input="Хто тримає цей район?",
    model="text-embedding-qwen",
    encoding_format="float"
)

vector = response.data[0].embedding
print(f"Vector dimensions: {len(vector)}")
print(f"First 5 values: {vector[:5]}")
