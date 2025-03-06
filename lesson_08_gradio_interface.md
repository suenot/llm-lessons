# Урок 8: Создание веб-интерфейса с Gradio

## Введение в Gradio

Gradio — это библиотека Python, которая позволяет быстро создавать веб-интерфейсы для моделей машинного обучения и ИИ-приложений. Она особенно полезна для прототипирования, демонстрации и тестирования моделей без необходимости в создании сложных веб-приложений с нуля.

В этом уроке мы рассмотрим, как интегрировать наши агенты, созданные с помощью SmolaGents и LiteLLM, с веб-интерфейсом Gradio для создания полноценного интерактивного приложения.

### Особенности Gradio:

1. **Простота использования**: Создание интерфейса в несколько строк кода
2. **Разнообразные компоненты**: Поддержка различных типов ввода и вывода (текст, изображения, аудио и т.д.)
3. **Интерактивность**: Возможность взаимодействия с моделями в реальном времени
4. **Публикация**: Простой способ поделиться своим приложением с другими

## Установка Gradio

Начнем с установки Gradio:

```bash
pip install gradio
```

## Создание простого текстового интерфейса

Создадим простой интерфейс для работы с LiteLLM:

```python
import gradio as gr
import litellm

def generate_text(prompt):
    """Функция для генерации ответа с использованием LiteLLM"""
    response = litellm.completion(
        model="openrouter/openai/gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# Создаем интерфейс
with gr.Blocks() as demo:
    gr.Markdown("# Демонстрация работы LiteLLM")
    
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(label="Ваш запрос", placeholder="Введите ваш вопрос здесь...")
            submit_btn = gr.Button("Отправить запрос")
        
        with gr.Column():
            output_text = gr.Textbox(label="Ответ модели")
    
    # Связываем функцию с кнопкой
    submit_btn.click(generate_text, inputs=input_text, outputs=output_text)

# Запускаем приложение
demo.launch()
```

## Потоковая передача ответов (Streaming)

Для более удобного взаимодействия добавим потоковую передачу ответов:

```python
import gradio as gr
import litellm

def generate_with_streaming(prompt):
    """Генерация ответа с использованием streaming"""
    response = litellm.completion(
        model="openrouter/openai/gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )
    
    partial_response = ""
    
    # Генератор для потоковой передачи ответа
    for chunk in response:
        if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
            content = chunk.choices[0].delta.content
            if content:
                partial_response += content
                yield partial_response

# Создаем интерфейс с поддержкой streaming
with gr.Blocks() as demo:
    gr.Markdown("# Демонстрация работы LiteLLM со streaming")
    
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(label="Ваш запрос", placeholder="Введите ваш вопрос здесь...")
            submit_btn = gr.Button("Отправить запрос")
        
        with gr.Column():
            output_text = gr.Textbox(label="Ответ модели")
    
    # Связываем функцию с кнопкой
    submit_btn.click(generate_with_streaming, inputs=input_text, outputs=output_text)

# Запускаем приложение
demo.launch()
```

## Интеграция SmolaGents с Gradio

Теперь создадим интерфейс для работы с агентом из SmolaGents:

```python
import gradio as gr
from smolagents import ToolCallingAgent, LiteLLMModel, tool

# Определяем инструменты для агента
@tool
def калькулятор(выражение: str) -> str:
    """
    Вычисляет математическое выражение.
    
    Args:
        выражение: Математическое выражение для вычисления
    """
    try:
        result = eval(выражение)
        return f"Результат: {result}"
    except Exception as e:
        return f"Ошибка при вычислении: {str(e)}"

@tool
def погода(город: str) -> str:
    """
    Возвращает текущую погоду в указанном городе.
    
    Args:
        город: Название города
    """
    # В реальном приложении здесь был бы запрос к API погоды
    return f"Текущая погода в городе {город}: +20°C, солнечно."

# Создаем модель и агента
model = LiteLLMModel(model_id="openrouter/openai/gpt-3.5-turbo")
agent = ToolCallingAgent(
    tools=[калькулятор, погода],
    model=model,
    max_steps=5,
    verbosity_level=1
)

# Функция для обработки запроса
def process_request(query, history):
    # Обновляем историю с запросом пользователя
    history.append((query, ""))
    
    # Получаем ответ от агента
    response = agent(query)
    
    # Обновляем историю с ответом
    history[-1] = (query, response)
    
    return history, ""  # Возвращаем обновленную историю и очищаем поле ввода

# Создаем интерфейс чата
with gr.Blocks() as demo:
    gr.Markdown("# AI-ассистент с использованием SmolaGents")
    
    # История чата и поле ввода
    chatbot = gr.Chatbot()
    msg = gr.Textbox(placeholder="Задайте вопрос или выполните команду...", show_label=False)
    clear = gr.Button("Очистить диалог")
    
    # Обработчики событий
    msg.submit(process_request, [msg, chatbot], [chatbot, msg])
    clear.click(lambda: None, None, chatbot, queue=False)

# Запускаем приложение
demo.launch()
```

## Создание интерфейса с несколькими вкладками

Создадим более сложный интерфейс с несколькими вкладками для разных функций:

```python
import gradio as gr
from smolagents import ToolCallingAgent, CodeAgent, LiteLLMModel, tool
import time

# Создаем модели
chat_model = LiteLLMModel(model_id="openrouter/openai/gpt-3.5-turbo")
code_model = LiteLLMModel(model_id="openrouter/openai/gpt-4")

# Инструменты для агентов
@tool
def калькулятор(выражение: str) -> str:
    """Вычисляет математическое выражение."""
    try:
        return f"Результат: {eval(выражение)}"
    except Exception as e:
        return f"Ошибка: {str(e)}"

@tool
def время() -> str:
    """Возвращает текущую дату и время."""
    return f"Текущее время: {time.strftime('%Y-%m-%d %H:%M:%S')}"

@tool
def выполнить_код(код: str) -> str:
    """Выполняет код Python и возвращает результат."""
    try:
        # Создаем изолированное пространство имен
        local_dict = {}
        
        # Выполняем код
        exec(код, {"__builtins__": __builtins__}, local_dict)
        
        # Собираем результаты
        result = []
        for var_name, var_value in local_dict.items():
            if not var_name.startswith('_'):
                result.append(f"{var_name} = {var_value}")
        
        return "\n".join(result) if result else "Код выполнен успешно"
    except Exception as e:
        return f"Ошибка при выполнении кода: {str(e)}"

# Создаем агентов
chat_agent = ToolCallingAgent(
    tools=[калькулятор, время],
    model=chat_model,
    max_steps=5,
    verbosity_level=1
)

code_agent = CodeAgent(
    tools=[выполнить_код],
    model=code_model,
    max_steps=10,
    verbosity_level=1
)

# Функции для обработки запросов
def chat_with_agent(query, history):
    history.append((query, ""))
    response = chat_agent(query)
    history[-1] = (query, response)
    return history, ""

def generate_code(query, history):
    history.append((query, ""))
    response = code_agent(query)
    history[-1] = (query, response)
    return history, ""

def generate_simple_text(prompt):
    response = litellm.completion(
        model="openrouter/openai/gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# Создаем многостраничный интерфейс
with gr.Blocks() as demo:
    gr.Markdown("# Многофункциональный AI-ассистент")
    
    with gr.Tabs():
        with gr.Tab("Чат с агентом"):
            chatbot_ui = gr.Chatbot()
            chat_input = gr.Textbox(placeholder="Задайте вопрос...", show_label=False)
            chat_clear = gr.Button("Очистить чат")
            
            chat_input.submit(chat_with_agent, [chat_input, chatbot_ui], [chatbot_ui, chat_input])
            chat_clear.click(lambda: None, None, chatbot_ui, queue=False)
        
        with gr.Tab("Генерация кода"):
            code_chatbot = gr.Chatbot()
            code_input = gr.Textbox(placeholder="Опишите, какой код нужно создать...", show_label=False)
            code_clear = gr.Button("Очистить чат")
            
            code_input.submit(generate_code, [code_input, code_chatbot], [code_chatbot, code_input])
            code_clear.click(lambda: None, None, code_chatbot, queue=False)
        
        with gr.Tab("Генерация текста"):
            with gr.Row():
                with gr.Column():
                    text_input = gr.Textbox(placeholder="Введите запрос...", label="Запрос")
                    text_button = gr.Button("Сгенерировать")
                
                with gr.Column():
                    text_output = gr.Textbox(label="Сгенерированный текст")
            
            text_button.click(generate_simple_text, inputs=text_input, outputs=text_output)

# Запускаем приложение
demo.launch()
```

## Настройка компонентов интерфейса

Gradio предлагает множество компонентов для создания богатых интерфейсов. Рассмотрим некоторые из них:

### Дополнительные компоненты и настройки

```python
import gradio as gr
from smolagents import ToolCallingAgent, LiteLLMModel, tool

# Создаем модель и агент
model = LiteLLMModel(model_id="openrouter/openai/gpt-3.5-turbo")

@tool
def калькулятор(выражение: str) -> str:
    """Вычисляет математическое выражение."""
    try:
        return f"Результат: {eval(выражение)}"
    except Exception as e:
        return f"Ошибка: {str(e)}"

agent = ToolCallingAgent(
    tools=[калькулятор],
    model=model,
    max_steps=5
)

# Создаем интерфейс с различными компонентами
with gr.Blocks() as demo:
    gr.Markdown("# Расширенный интерфейс для работы с AI")
    
    with gr.Row():
        # Левая колонка с настройками
        with gr.Column(scale=1):
            gr.Markdown("## Настройки")
            
            model_choice = gr.Dropdown(
                choices=["openrouter/openai/gpt-3.5-turbo", "openrouter/openai/gpt-4", "openrouter/anthropic/claude-3-opus"],
                value="openrouter/openai/gpt-3.5-turbo",
                label="Выберите модель"
            )
            
            max_steps = gr.Slider(
                minimum=1, 
                maximum=10, 
                value=5, 
                step=1, 
                label="Максимальное количество шагов"
            )
            
            temperature = gr.Slider(
                minimum=0.1, 
                maximum=1.0, 
                value=0.7, 
                step=0.1, 
                label="Температура (креативность)"
            )
            
            log_level = gr.Radio(
                choices=["Минимальный", "Стандартный", "Подробный", "Отладка"],
                value="Стандартный",
                label="Уровень логирования"
            )
            
            append_log = gr.Checkbox(label="Добавлять логи в ответ", value=False)
        
        # Правая колонка с чатом
        with gr.Column(scale=2):
            gr.Markdown("## Диалог с агентом")
            
            chatbot = gr.Chatbot(height=400)
            
            with gr.Row():
                msg = gr.Textbox(placeholder="Задайте вопрос...", show_label=False)
                submit_btn = gr.Button("Отправить")
            
            with gr.Row():
                clear_btn = gr.Button("Очистить чат")
                save_btn = gr.Button("Сохранить диалог")
    
    # Выводим логи
    with gr.Accordion("Логи выполнения", open=False):
        logs = gr.Textbox(label="Логи", lines=10)
    
    # Обработчики событий
    def update_agent_config(model_name, steps, temp, log):
        """Обновляет конфигурацию агента"""
        # Сопоставление уровней логирования
        log_map = {
            "Минимальный": 0,
            "Стандартный": 1,
            "Подробный": 2,
            "Отладка": 3
        }
        
        # Создаем новую модель с обновленными параметрами
        updated_model = LiteLLMModel(
            model_id=model_name,
            temperature=temp
        )
        
        # Обновляем агента
        global agent
        agent = ToolCallingAgent(
            tools=[калькулятор],
            model=updated_model,
            max_steps=steps,
            verbosity_level=log_map[log]
        )
        
        return f"Настройки обновлены: модель={model_name}, шаги={steps}, температура={temp}, логи={log}"
    
    def process_request(message, chat_history, include_logs):
        """Обрабатывает запрос и возвращает ответ"""
        chat_history.append((message, ""))
        
        # Получаем ответ от агента
        response = agent(message)
        
        # Если нужно добавить логи
        if include_logs:
            if hasattr(agent, 'last_logs') and agent.last_logs:
                logs_text = "\n\n--- Логи выполнения ---\n" + agent.last_logs
                response += logs_text
        
        chat_history[-1] = (message, response)
        
        # Обновляем логи
        logs_output = agent.last_logs if hasattr(agent, 'last_logs') else "Логи недоступны"
        
        return chat_history, "", logs_output
    
    def save_conversation(chat_history):
        """Сохраняет разговор в файл"""
        if not chat_history:
            return "Нет данных для сохранения"
        
        filename = f"conversation_{time.strftime('%Y%m%d-%H%M%S')}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            for user, bot in chat_history:
                f.write(f"Пользователь: {user}\n")
                f.write(f"Ассистент: {bot}\n\n")
        
        return f"Разговор сохранен в файл: {filename}"
    
    # Связываем обработчики с компонентами
    submit_btn.click(
        process_request,
        inputs=[msg, chatbot, append_log],
        outputs=[chatbot, msg, logs]
    )
    
    msg.submit(
        process_request,
        inputs=[msg, chatbot, append_log],
        outputs=[chatbot, msg, logs]
    )
    
    clear_btn.click(
        lambda: (None, "", "Логи очищены"),
        None,
        [chatbot, msg, logs]
    )
    
    save_btn.click(
        save_conversation,
        inputs=[chatbot],
        outputs=[logs]
    )
    
    # Обновление конфигурации агента при изменении настроек
    model_choice.change(
        update_agent_config,
        inputs=[model_choice, max_steps, temperature, log_level],
        outputs=[logs]
    )
    
    max_steps.change(
        update_agent_config,
        inputs=[model_choice, max_steps, temperature, log_level],
        outputs=[logs]
    )
    
    temperature.change(
        update_agent_config,
        inputs=[model_choice, max_steps, temperature, log_level],
        outputs=[logs]
    )
    
    log_level.change(
        update_agent_config,
        inputs=[model_choice, max_steps, temperature, log_level],
        outputs=[logs]
    )

# Запускаем приложение
demo.launch()
```

## Публикация приложения

Gradio позволяет легко делиться своими приложениями с другими пользователями. Для этого можно использовать Hugging Face Spaces или временные публичные ссылки.

### Временные ссылки

```python
import gradio as gr
from smolagents import ToolCallingAgent, LiteLLMModel, tool

# Простой агент для демонстрации
model = LiteLLMModel(model_id="openrouter/openai/gpt-3.5-turbo")
agent = ToolCallingAgent(tools=[], model=model)

# Создаем простой интерфейс
with gr.Blocks() as demo:
    gr.Markdown("# Демонстрационный чат-бот")
    
    chatbot = gr.Chatbot()
    msg = gr.Textbox(placeholder="Задайте вопрос...")
    clear = gr.Button("Очистить")
    
    def respond(message, chat_history):
        response = agent(message)
        chat_history.append((message, response))
        return "", chat_history
    
    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)

# Запускаем с созданием публичной ссылки на 72 часа
demo.launch(share=True)  # Создаст временную ссылку вида https://xxxxx.gradio.app
```

### Размещение на Hugging Face Spaces

1. Создайте аккаунт на [Hugging Face](https://huggingface.co/)
2. Создайте новый Space и выберите Gradio
3. Загрузите ваш код и необходимые файлы
4. Пример файла `app.py`:

```python
import gradio as gr
import litellm
import os

# Получаем API ключ из переменной окружения
api_key = os.environ.get("OPENROUTER_API_KEY", "")
os.environ["OPENROUTER_API_KEY"] = api_key

def generate_text(prompt):
    response = litellm.completion(
        model="openrouter/openai/gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        api_key=api_key
    )
    return response.choices[0].message.content

with gr.Blocks() as demo:
    gr.Markdown("# AI-ассистент")
    
    input_text = gr.Textbox(label="Ваш запрос")
    output_text = gr.Textbox(label="Ответ модели")
    submit_btn = gr.Button("Отправить")
    
    submit_btn.click(generate_text, inputs=input_text, outputs=output_text)

demo.launch()
```

5. Создайте файл `requirements.txt`:

```
gradio
litellm
smolagents
```

## Интеграция с другими компонентами веб-приложения

Gradio можно интегрировать с другими веб-фреймворками, такими как Flask или FastAPI.

### Интеграция с FastAPI

```python
import gradio as gr
from fastapi import FastAPI
from smolagents import ToolCallingAgent, LiteLLMModel
import uvicorn

# Создаем модель и агента
model = LiteLLMModel(model_id="openrouter/openai/gpt-3.5-turbo")
agent = ToolCallingAgent(tools=[], model=model)

# Создаем приложение FastAPI
app = FastAPI()

# API-эндпоинт для прямого доступа к агенту
@app.post("/api/chat")
async def chat_endpoint(query: str):
    response = agent(query)
    return {"response": response}

# Создаем интерфейс Gradio
def chat_with_agent(message, history):
    history.append((message, ""))
    response = agent(message)
    history[-1] = (message, response)
    return history, ""

# Определяем интерфейс
chat_interface = gr.Interface(
    fn=chat_with_agent,
    inputs=[gr.Textbox(placeholder="Введите сообщение..."), gr.State([])],
    outputs=[gr.Chatbot(), gr.Textbox(placeholder="Введите сообщение...")],
    title="AI-ассистент",
    description="Задайте вопрос искусственному интеллекту"
)

# Интегрируем Gradio с FastAPI
app = gr.mount_gradio_app(app, chat_interface, path="/")

# Запускаем приложение
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## Практическое задание

Создайте веб-интерфейс для мультифункционального агента, который может:

1. Отвечать на общие вопросы
2. Анализировать тексты
3. Выполнять математические вычисления
4. Генерировать изображения (с помощью текстового описания)

Пример решения:

```python
import gradio as gr
import litellm
from smolagents import ToolCallingAgent, LiteLLMModel, tool
from PIL import Image
import numpy as np
import requests
from io import BytesIO
import os

# Настройка API ключей
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
STABILITY_API_KEY = os.environ.get("STABILITY_API_KEY", "")  # Для генерации изображений

# Создаем модель и инструменты
model = LiteLLMModel(model_id="openrouter/openai/gpt-3.5-turbo")

@tool
def калькулятор(выражение: str) -> str:
    """
    Вычисляет математическое выражение.
    
    Args:
        выражение: Математическое выражение для вычисления
    """
    try:
        result = eval(выражение)
        return f"Результат: {result}"
    except Exception as e:
        return f"Ошибка при вычислении: {str(e)}"

@tool
def анализ_текста(текст: str) -> str:
    """
    Анализирует текст и возвращает информацию о нем.
    
    Args:
        текст: Текст для анализа
    """
    # Расчет базовых статистик
    words = текст.split()
    num_words = len(words)
    num_chars = len(текст)
    num_sentences = текст.count('.') + текст.count('!') + текст.count('?')
    
    # Анализ тональности через LLM
    response = litellm.completion(
        model="openrouter/openai/gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Вы - система анализа текстов. Проанализируйте текст и определите его тональность, стиль и основные темы. Дайте краткий анализ в 2-3 предложениях."},
            {"role": "user", "content": текст}
        ]
    )
    analysis = response.choices[0].message.content
    
    return f"""Статистика текста:
- Количество слов: {num_words}
- Количество символов: {num_chars}
- Количество предложений: {num_sentences}

Анализ содержания:
{analysis}
"""

@tool
def генерация_изображения(описание: str) -> str:
    """
    Генерирует изображение по текстовому описанию.
    
    Args:
        описание: Текстовое описание изображения
    """
    try:
        if not STABILITY_API_KEY:
            return "API ключ для Stability API не настроен"
        
        # Запрос к Stability AI API
        url = "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image"
        headers = {
            "Authorization": f"Bearer {STABILITY_API_KEY}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        payload = {
            "text_prompts": [{"text": описание}],
            "cfg_scale": 7,
            "height": 1024,
            "width": 1024,
            "samples": 1
        }
        
        response = requests.post(url, headers=headers, json=payload)
        
        if response.status_code != 200:
            return f"Ошибка API: {response.text}"
        
        # Путь для сохранения изображения
        image_path = "generated_image.png"
        
        # Сохраняем изображение
        with open(image_path, "wb") as f:
            f.write(response.content)
        
        return f"Изображение успешно сгенерировано и сохранено как {image_path}"
    except Exception as e:
        return f"Ошибка при генерации изображения: {str(e)}"

# Создаем агента
agent = ToolCallingAgent(
    tools=[калькулятор, анализ_текста, генерация_изображения],
    model=model,
    max_steps=5,
    verbosity_level=1
)

# Создаем интерфейс
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Мультифункциональный AI-ассистент")
    
    with gr.Tabs():
        # Вкладка с чатом
        with gr.Tab("Чат с ассистентом"):
            chat_history = gr.Chatbot(height=400)
            chat_input = gr.Textbox(placeholder="Задайте вопрос или выполните команду...", show_label=False)
            chat_clear = gr.Button("Очистить чат")
            
            # Обработчик запросов в чате
            def chat_response(message, history):
                history.append((message, ""))
                response = agent(message)
                history[-1] = (message, response)
                return history, ""
            
            chat_input.submit(chat_response, [chat_input, chat_history], [chat_history, chat_input])
            chat_clear.click(lambda: None, None, chat_history, queue=False)
        
        # Вкладка с анализом текста
        with gr.Tab("Анализ текста"):
            with gr.Row():
                with gr.Column():
                    text_input = gr.Textbox(placeholder="Введите текст для анализа...", lines=10, label="Исходный текст")
                    analyze_btn = gr.Button("Анализировать")
                
                with gr.Column():
                    text_analysis = gr.Textbox(label="Результаты анализа", lines=10)
            
            # Обработчик анализа текста
            def analyze_text(text):
                return анализ_текста(text)
            
            analyze_btn.click(analyze_text, inputs=text_input, outputs=text_analysis)
        
        # Вкладка с калькулятором
        with gr.Tab("Калькулятор"):
            with gr.Row():
                calc_input = gr.Textbox(placeholder="Введите математическое выражение...", label="Выражение")
                calc_btn = gr.Button("Вычислить")
                calc_result = gr.Textbox(label="Результат")
            
            # Обработчик вычислений
            def calculate(expression):
                return калькулятор(expression)
            
            calc_btn.click(calculate, inputs=calc_input, outputs=calc_result)
        
        # Вкладка с генерацией изображений
        with gr.Tab("Генерация изображений"):
            with gr.Row():
                with gr.Column():
                    image_prompt = gr.Textbox(placeholder="Опишите изображение...", label="Описание изображения")
                    generate_btn = gr.Button("Сгенерировать")
                
                with gr.Column():
                    generated_image = gr.Image(label="Сгенерированное изображение")
                    image_status = gr.Textbox(label="Статус")
            
            # Обработчик генерации изображений
            def generate_image(prompt):
                response = генерация_изображения(prompt)
                
                if "успешно" in response:
                    return "generated_image.png", response
                else:
                    return None, response
            
            generate_btn.click(generate_image, inputs=image_prompt, outputs=[generated_image, image_status])
    
    # Подвал с информацией
    with gr.Accordion("О приложении", open=False):
        gr.Markdown("""
        # Мультифункциональный AI-ассистент
        
        Это демонстрационное приложение, созданное с использованием:
        - SmolaGents для агентов
        - LiteLLM для работы с языковыми моделями
        - Gradio для веб-интерфейса
        
        ## Возможности
        
        - Общение с AI-ассистентом
        - Анализ текстов
        - Выполнение математических вычислений
        - Генерация изображений по описанию
        
        ## Требуемые API ключи
        
        - OpenRouter API для работы с моделями
        - Stability API для генерации изображений
        """)

# Запускаем приложение
demo.launch()
```

## Дополнительные ресурсы

- [Официальная документация Gradio](https://www.gradio.app/docs/)
- [Примеры приложений на Hugging Face Spaces](https://huggingface.co/spaces)
- [Руководство по интеграции Gradio с FastAPI](https://www.gradio.app/guides/running-gradio-on-your-web-server-with-apache)
- [Галерея компонентов Gradio](https://www.gradio.app/docs/components)
- [Репозиторий с примерами Gradio](https://github.com/gradio-app/gradio/tree/main/demo) 