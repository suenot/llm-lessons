# Урок 7: Streaming и асинхронные возможности

## Введение в streaming и асинхронность

При работе с языковыми моделями (LLM) важно эффективно организовать взаимодействие с ними. В этом уроке мы рассмотрим два ключевых механизма, которые помогают оптимизировать это взаимодействие:

1. **Streaming** - получение ответа от модели частями в режиме реального времени
2. **Асинхронность** - параллельное выполнение запросов для повышения производительности

Эти возможности поддерживаются библиотеками LiteLLM и SmolaGents и могут значительно улучшить пользовательский опыт и эффективность ваших приложений.

## Работа со streaming-ответами в LiteLLM

Streaming позволяет получать ответ от модели по мере его генерации, а не ждать, пока будет сгенерирован весь ответ.

### Преимущества streaming

- Мгновенная обратная связь для пользователя
- Снижение воспринимаемой задержки
- Возможность начать обработку частей ответа до завершения генерации
- Более естественный диалоговый опыт

### Базовый пример streaming с LiteLLM

```python
from litellm import completion
import time

def print_chunk(chunk):
    # Извлекаем содержимое из ответа
    content = chunk.choices[0].delta.content
    if content:
        print(content, end="", flush=True)

# Отправляем запрос в режиме streaming
response = completion(
    model="openrouter/openai/gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Расскажи историю о космическом путешествии. Постарайся сделать её длинной."}],
    stream=True  # Включаем режим streaming
)

print("Ответ модели: ", end="")
# Обрабатываем поток частей ответа
for chunk in response:
    print_chunk(chunk)
    time.sleep(0.01)  # Небольшая задержка для наглядности

print("\nОтвет получен полностью.")
```

### Использование streaming с LiteLLMClient

В нашем проекте можно использовать streaming через класс `LiteLLMClient`:

```python
from module import Request, LiteLLMClient
from pydantic import BaseModel, Field

class StreamingResponse(BaseModel):
    content: str = Field(..., description="Сгенерированный текст")

# Создаем запрос с включенным streaming
request = Request(
    model="openrouter/openai/gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Расскажи длинную историю о приключениях пирата"}],
    stream=True,  # Включаем streaming
    answer_model=StreamingResponse
)

# Создаем клиент
client = LiteLLMClient(request)

# Обработка потока ответов
def process_stream():
    for response_chunk in client.stream_response():
        if hasattr(response_chunk, 'content') and response_chunk.content:
            print(response_chunk.content, end="", flush=True)
        else:
            # Обрабатываем случай, когда формат ответа отличается
            delta = response_chunk.choices[0].delta if hasattr(response_chunk, 'choices') else None
            content = delta.get('content', '') if isinstance(delta, dict) else getattr(delta, 'content', '')
            if content:
                print(content, end="", flush=True)

# Запускаем обработку
process_stream()
print("\nОтвет полностью получен")
```

### Обработка и отображение streaming-ответов

В веб-приложениях и интерфейсах можно использовать streaming для постепенного отображения ответа:

```python
import gradio as gr
from litellm import completion

def stream_response(message):
    # Генератор для потоковой передачи ответа
    response = completion(
        model="openrouter/openai/gpt-3.5-turbo",
        messages=[{"role": "user", "content": message}],
        stream=True
    )
    
    # Буфер для накопления частей ответа
    partial_response = ""
    
    # Возвращаем части ответа по мере их получения
    for chunk in response:
        content = chunk.choices[0].delta.content
        if content:
            partial_response += content
            # Возвращаем обновленный ответ и флаг, что генерация продолжается
            yield partial_response

# Создаем интерфейс Gradio с потоковой передачей
with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    
    def respond(message, chat_history):
        # Добавляем сообщение пользователя в историю
        chat_history.append((message, ""))
        
        # Обновляем ответ по частям
        for partial_response in stream_response(message):
            chat_history[-1] = (message, partial_response)
            yield chat_history
    
    msg.submit(respond, [msg, chatbot], [chatbot])

demo.queue().launch()
```

## Асинхронные запросы к моделям

Асинхронное программирование позволяет выполнять несколько операций параллельно, не блокируя основной поток выполнения. Это особенно полезно при работе с LLM, где запросы могут занимать значительное время.

### Базовый пример асинхронного запроса с LiteLLM

```python
import asyncio
from litellm import acompletion  # Асинхронная версия completion

async def get_llm_response(prompt):
    try:
        response = await acompletion(
            model="openrouter/openai/gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Ошибка: {str(e)}"

async def main():
    # Создаем несколько задач для параллельного выполнения
    prompts = [
        "Расскажи о квантовой физике",
        "Опиши процесс фотосинтеза",
        "Что такое искусственный интеллект?",
        "Как работает блокчейн?"
    ]
    
    # Запускаем все задачи параллельно
    tasks = [get_llm_response(prompt) for prompt in prompts]
    results = await asyncio.gather(*tasks)
    
    # Выводим результаты
    for i, result in enumerate(results):
        print(f"Ответ на запрос {i+1}:")
        print(result[:100] + "...\n")

# Запускаем асинхронную функцию
asyncio.run(main())
```

### Асинхронный подход в SmolaGents

В SmolaGents также можно использовать асинхронные инструменты:

```python
import asyncio
import aiohttp
from smolagents import ToolCallingAgent, LiteLLMModel, tool

# Асинхронный инструмент для получения данных о погоде
@tool
async def async_weather(city: str) -> str:
    """
    Асинхронно получает информацию о погоде в указанном городе.
    
    Args:
        city: Название города
    """
    async with aiohttp.ClientSession() as session:
        # В реальном приложении здесь будет запрос к API погоды
        await asyncio.sleep(1)  # Имитируем задержку сети
        return f"Погода в городе {city}: Солнечно, +25°C"

# Асинхронный инструмент для получения новостей
@tool
async def async_news(topic: str) -> str:
    """
    Асинхронно получает последние новости по указанной теме.
    
    Args:
        topic: Тема новостей
    """
    async with aiohttp.ClientSession() as session:
        # В реальном приложении здесь будет запрос к API новостей
        await asyncio.sleep(1.5)  # Имитируем задержку сети
        return f"Последние новости по теме '{topic}': ..."

# Создаем модель и агента
model = LiteLLMModel(model_id="openrouter/openai/gpt-3.5-turbo")
agent = ToolCallingAgent(
    tools=[async_weather, async_news],
    model=model
)

# Асинхронная функция для запуска агента
async def run_agent():
    response = await agent.arun("Какая погода в Москве и какие новости о технологиях?")
    print(response)

# Запускаем агента
asyncio.run(run_agent())
```

## Параллельная обработка запросов

Для обработки множества запросов можно использовать асинхронные подходы и параллельное выполнение.

### Параллельная обработка с asyncio

```python
import asyncio
import time
from litellm import acompletion

async def process_query(query, index):
    start_time = time.time()
    print(f"Запрос {index} начат: {query}")
    
    response = await acompletion(
        model="openrouter/openai/gpt-3.5-turbo",
        messages=[{"role": "user", "content": query}]
    )
    
    execution_time = time.time() - start_time
    result = response.choices[0].message.content
    
    print(f"Запрос {index} завершен за {execution_time:.2f} сек")
    return {
        "query": query,
        "result": result,
        "time": execution_time
    }

async def process_batch(queries):
    tasks = []
    for i, query in enumerate(queries):
        tasks.append(process_query(query, i+1))
    
    # Запускаем все задачи параллельно
    results = await asyncio.gather(*tasks)
    return results

# Список запросов для обработки
queries = [
    "Напиши короткое стихотворение о весне",
    "Объясни концепцию искусственного интеллекта",
    "Расскажи о достопримечательностях Парижа",
    "Как приготовить идеальный стейк?",
    "Опиши основные принципы квантовой механики"
]

# Замеряем общее время выполнения всех запросов
start_time = time.time()
results = asyncio.run(process_batch(queries))
total_time = time.time() - start_time

# Выводим информацию о результатах
print(f"\nВсе запросы обработаны за {total_time:.2f} сек")
print(f"Среднее время на запрос: {total_time/len(queries):.2f} сек")
print(f"Суммарное время выполнения запросов: {sum(r['time'] for r in results):.2f} сек")
print(f"Экономия времени благодаря параллельному выполнению: {sum(r['time'] for r in results) - total_time:.2f} сек")
```

### Семафоры для ограничения параллельных запросов

Иногда необходимо ограничить количество одновременных запросов, чтобы не превысить лимиты API:

```python
import asyncio
from litellm import acompletion

async def controlled_process(queries, max_concurrent=3):
    # Создаем семафор для ограничения количества одновременных запросов
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_with_semaphore(query):
        # Захватываем семафор перед выполнением запроса
        async with semaphore:
            print(f"Обработка запроса: {query}")
            response = await acompletion(
                model="openrouter/openai/gpt-3.5-turbo",
                messages=[{"role": "user", "content": query}]
            )
            return response.choices[0].message.content
    
    # Создаем задачи с использованием семафора
    tasks = [process_with_semaphore(query) for query in queries]
    results = await asyncio.gather(*tasks)
    
    return results

# Пример использования
queries = [f"Вопрос номер {i}" for i in range(1, 11)]
results = asyncio.run(controlled_process(queries, max_concurrent=3))
```

## Оптимизация производительности

### Стратегии оптимизации при работе с LLM

1. **Правильный выбор модели**:
   - Используйте более компактные модели для простых задач
   - Резервируйте мощные модели для сложных запросов
   
2. **Эффективное использование контекста**:
   - Минимизируйте размер сообщений для уменьшения токенов
   - Удаляйте ненужную информацию из контекста
   
3. **Кэширование**:
   - Кэшируйте одинаковые или похожие запросы
   - Используйте Redis для распределенного кэширования

4. **Асинхронные и параллельные запросы**:
   - Используйте асинхронные функции для неблокирующего выполнения
   - Группируйте запросы, где это возможно

### Пример оптимизации производительности

```python
import asyncio
import hashlib
import json
from litellm import acompletion
from functools import lru_cache

# Кэш запросов в памяти
@lru_cache(maxsize=100)
def get_cache_key(model, messages):
    # Создаем стабильный хэш для запроса
    message_str = json.dumps(messages, sort_keys=True)
    return hashlib.md5(f"{model}:{message_str}".encode()).hexdigest()

# Хранилище кэша
cache = {}

async def optimized_completion(model, messages, use_cache=True):
    if use_cache:
        # Проверяем кэш
        cache_key = get_cache_key(model, json.dumps(messages))
        if cache_key in cache:
            print("Используем кэшированный ответ")
            return cache[cache_key]
    
    # Выполняем запрос к API
    response = await acompletion(
        model=model,
        messages=messages
    )
    
    result = response.choices[0].message.content
    
    if use_cache:
        # Сохраняем в кэш
        cache[cache_key] = result
    
    return result

async def batch_process(queries, model, batch_size=5):
    results = []
    
    # Обрабатываем запросы пакетами
    for i in range(0, len(queries), batch_size):
        batch = queries[i:i + batch_size]
        
        # Формируем задачи для пакета
        tasks = []
        for query in batch:
            task = optimized_completion(
                model=model,
                messages=[{"role": "user", "content": query}]
            )
            tasks.append(task)
        
        # Выполняем пакет задач
        batch_results = await asyncio.gather(*tasks)
        results.extend(batch_results)
    
    return results

# Пример использования
async def main():
    queries = [
        "Что такое искусственный интеллект?",
        "Как работает нейронная сеть?",
        "Что такое искусственный интеллект?",  # Дублирующийся запрос для демонстрации кэша
        "Объясни концепцию машинного обучения",
        "Что такое глубокое обучение?",
        "Как работает нейронная сеть?",  # Еще один дублирующийся запрос
    ]
    
    model = "openrouter/openai/gpt-3.5-turbo"
    results = await batch_process(queries, model, batch_size=3)
    
    for i, result in enumerate(results):
        print(f"Результат {i+1}: {result[:50]}...\n")

asyncio.run(main())
```

## Асинхронные агенты в SmolaGents

SmolaGents позволяет создавать полностью асинхронных агентов, которые могут эффективно выполнять задачи в асинхронном режиме.

### Создание асинхронного агента

```python
import asyncio
import aiohttp
from smolagents import ToolCallingAgent, LiteLLMModel, tool

# Определение асинхронных инструментов
@tool
async def async_search(query: str) -> str:
    """
    Выполняет асинхронный поиск информации.
    
    Args:
        query: Поисковый запрос
    """
    await asyncio.sleep(1)  # Имитация сетевого запроса
    return f"Результаты поиска по запросу '{query}': [...]"

@tool
async def async_translate(text: str, target_lang: str) -> str:
    """
    Асинхронно переводит текст на указанный язык.
    
    Args:
        text: Текст для перевода
        target_lang: Целевой язык перевода
    """
    await asyncio.sleep(0.5)  # Имитация сетевого запроса
    return f"Перевод на {target_lang}: '[перевод текста]'"

# Создаем асинхронный агент
async def create_async_agent():
    # Создаем модель
    model = LiteLLMModel(model_id="openrouter/openai/gpt-3.5-turbo")
    
    # Создаем агента с асинхронными инструментами
    agent = ToolCallingAgent(
        tools=[async_search, async_translate],
        model=model
    )
    
    return agent

# Асинхронная функция для использования агента
async def process_queries(queries):
    # Создаем агента
    agent = await create_async_agent()
    
    # Обрабатываем запросы параллельно
    async def process_query(query):
        return await agent.arun(query)
    
    tasks = [process_query(query) for query in queries]
    results = await asyncio.gather(*tasks)
    
    return results

# Пример использования
async def main():
    queries = [
        "Найди информацию о Париже и переведи её на испанский",
        "Найди данные о солнечной системе и переведи на немецкий",
        "Что такое искусственный интеллект? Переведи на французский"
    ]
    
    results = await process_queries(queries)
    
    for i, result in enumerate(results):
        print(f"Результат {i+1}: {result}\n")

# Запускаем асинхронную функцию
asyncio.run(main())
```

## Практическое задание

Создайте систему для параллельного анализа новостей, которая:

1. Асинхронно получает новости из нескольких источников
2. Использует streaming для отображения результатов анализа по мере их поступления
3. Оптимизирует производительность с использованием кэширования
4. Обрабатывает ошибки без остановки всего процесса

Пример решения:

```python
import asyncio
import aiohttp
import time
import hashlib
import json
from litellm import acompletion
from functools import lru_cache

# Имитация источников новостей
NEWS_SOURCES = {
    "tech": "https://api.example.com/news/tech",
    "business": "https://api.example.com/news/business",
    "science": "https://api.example.com/news/science",
    "sports": "https://api.example.com/news/sports",
    "politics": "https://api.example.com/news/politics"
}

# Кэш для хранения результатов запросов
cache = {}

# Функция для создания ключа кэша
def get_cache_key(data):
    data_str = json.dumps(data, sort_keys=True)
    return hashlib.md5(data_str.encode()).hexdigest()

# Асинхронная функция для получения новостей из источника
async def fetch_news(session, source_name, url):
    try:
        # Эмуляция запроса к API новостей
        await asyncio.sleep(1 + (hash(source_name) % 5) / 10)
        
        # В реальном приложении здесь был бы запрос к API
        # async with session.get(url) as response:
        #     return await response.json()
        
        # Генерируем тестовые данные
        return {
            "source": source_name,
            "title": f"Новости раздела {source_name}",
            "articles": [
                {"title": f"Статья {i} из {source_name}", "summary": f"Краткое содержание статьи {i} из раздела {source_name}"} 
                for i in range(1, 4)
            ]
        }
    except Exception as e:
        print(f"Ошибка при получении новостей из {source_name}: {str(e)}")
        return {
            "source": source_name,
            "error": str(e),
            "articles": []
        }

# Асинхронная функция для анализа новостной статьи
async def analyze_article(article, source):
    # Проверяем кэш
    cache_key = get_cache_key({"article": article, "source": source})
    if cache_key in cache:
        print(f"Используем кэшированный анализ для статьи: {article['title']}")
        return cache[cache_key]
    
    # Формируем запрос к модели
    prompt = f"Проанализируй короткую новостную статью и определи её основную тему, тональность и ключевые факты:\n\nИсточник: {source}\nЗаголовок: {article['title']}\nСодержание: {article['summary']}"
    
    try:
        # Выполняем анализ с помощью LLM
        response = await acompletion(
            model="openrouter/openai/gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        
        analysis = response.choices[0].message.content
        
        # Сохраняем результат в кэш
        cache[cache_key] = analysis
        
        return analysis
    except Exception as e:
        print(f"Ошибка при анализе статьи {article['title']}: {str(e)}")
        return f"Не удалось проанализировать статью: {str(e)}"

# Асинхронная функция для обработки потока новостей
async def process_news_stream():
    async with aiohttp.ClientSession() as session:
        # Получаем новости из всех источников параллельно
        fetch_tasks = [
            fetch_news(session, source_name, url) 
            for source_name, url in NEWS_SOURCES.items()
        ]
        
        # Запускаем все задачи параллельно
        news_data = await asyncio.gather(*fetch_tasks)
        
        # Обрабатываем каждый источник новостей
        for news in news_data:
            source = news["source"]
            print(f"\n=== Новости из {source} ===")
            
            if "error" in news:
                print(f"Ошибка: {news['error']}")
                continue
            
            # Обрабатываем статьи из этого источника параллельно
            articles = news["articles"]
            analysis_tasks = [analyze_article(article, source) for article in articles]
            
            # Обрабатываем результаты анализа по мере их поступления
            for i, article_task in enumerate(asyncio.as_completed(analysis_tasks)):
                analysis_result = await article_task
                print(f"\nСтатья {i+1}:")
                print(f"Заголовок: {articles[i]['title']}")
                print(f"Анализ: {analysis_result[:100]}...")
                
                # Эмулируем отображение в реальном времени
                await asyncio.sleep(0.2)

# Основная асинхронная функция
async def main():
    start_time = time.time()
    print("Начинаем анализ новостей...")
    
    # Устанавливаем таймаут для всего процесса
    try:
        await asyncio.wait_for(process_news_stream(), timeout=30)
    except asyncio.TimeoutError:
        print("Превышено время ожидания. Процесс остановлен.")
    
    execution_time = time.time() - start_time
    print(f"\nАнализ завершен за {execution_time:.2f} секунд")
    print(f"Размер кэша: {len(cache)} записей")

# Запускаем программу
if __name__ == "__main__":
    asyncio.run(main())
```

## Дополнительные ресурсы

- [Документация LiteLLM по асинхронным функциям](https://docs.litellm.ai/docs/completion/stream)
- [Документация Python по asyncio](https://docs.python.org/3/library/asyncio.html)
- [Руководство по асинхронному программированию в Python](https://realpython.com/async-io-python/)
- [Примеры streaming с различными моделями](https://github.com/BerriAI/litellm/tree/main/examples/streaming_examples)
- [Оптимизация производительности LLM-приложений](https://www.pinecone.io/learn/series/langchain/langchain-streaming/) 