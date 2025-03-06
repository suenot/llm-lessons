# Урок 3: Введение в SmolaGents

## Что такое SmolaGents

**SmolaGents** - это библиотека для создания интеллектуальных агентов на основе языковых моделей (LLM). Библиотека предоставляет простой интерфейс для создания агентов, которые могут выполнять различные инструменты (tools) на основе пользовательских запросов.

Ключевые возможности SmolaGents:
- Создание агентов с использованием различных LLM моделей
- Декоратор `@tool` для легкого определения инструментов
- Встроенные агенты разных типов (ToolCallingAgent, CodeAgent и др.)
- Интеграция с Gradio для быстрого создания веб-интерфейсов
- Гибкая настройка выполнения и логирования

## Установка и базовая настройка

Для установки SmolaGents используйте pip:

```bash
pip install smolagents
```

Для работы с веб-интерфейсом Gradio:

```bash
pip install smolagents[gradio]
```

## Ключевые компоненты SmolaGents

### Модели LLM

SmolaGents поддерживает различные типы моделей через адаптеры:

1. **LiteLLMModel** - для работы с любыми моделями через LiteLLM
2. **HfApiModel** - для работы с моделями через HuggingFace Inference API
3. **TransformersModel** - для локального запуска моделей с помощью библиотеки transformers

Пример инициализации модели:

```python
from smolagents import LiteLLMModel, HfApiModel, TransformersModel

# Использование LiteLLM (OpenAI, Anthropic и др.)
litellm_model = LiteLLMModel(model_id="openrouter/openai/gpt-4o-mini")

# Использование HuggingFace Inference API
hf_model = HfApiModel(model_id="meta-llama/Llama-3-8B-Instruct")

# Локальная модель через transformers
local_model = TransformersModel(
    model_id="HuggingFaceTB/SmolLM2-1.7B-Instruct", 
    device_map="auto",
    max_new_tokens=1000
)
```

### Агенты

SmolaGents предлагает несколько типов агентов:

1. **ToolCallingAgent** - основной агент для вызова инструментов
2. **CodeAgent** - агент, специализирующийся на написании кода
3. **В нашем проекте** - `SimpleAgent` - обертка над ToolCallingAgent с упрощенным интерфейсом

Пример создания ToolCallingAgent:

```python
from smolagents import ToolCallingAgent, LiteLLMModel

model = LiteLLMModel(model_id="openrouter/openai/gpt-4o-mini")

agent = ToolCallingAgent(
    tools=[],  # Пустой список инструментов
    model=model,
    max_steps=5,       # Максимальное количество шагов
    verbosity_level=1  # Уровень логирования
)

result = agent("Какая сегодня погода в Москве?")
print(result)
```

### Инструменты (Tools)

Инструменты - это функции, которые агент может вызывать для выполнения конкретных задач. В SmolaGents инструменты определяются с помощью декоратора `@tool`:

```python
from smolagents import tool

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
```

## Класс SimpleAgent в проекте

В нашем проекте библиотека SmolaGents используется через класс `SimpleAgent`, который находится в файле `ai/module/agents/simple_agent.py`. Этот класс предоставляет упрощенный интерфейс для работы с SmolaGents:

```python
from module.agents import SimpleAgent, tool

# Определение инструмента
@tool
def приветствие(имя: str) -> str:
    """
    Приветствует пользователя по имени.
    
    Args:
        имя: Имя пользователя
    """
    return f"Привет, {имя}! Рад тебя видеть!"

# Создание агента
agent = SimpleAgent()

# Добавление инструмента
agent.add_tools([приветствие])

# Запуск агента с запросом
response = agent.run("Поздоровайся с Иваном")
print(response)
```

Преимущества SimpleAgent:
- Упрощенный интерфейс для создания и использования агентов
- Автоматическая настройка LiteLLMModel с поддержкой OpenRouter
- Встроенная поддержка веб-интерфейса через Gradio
- Обработка ошибок и логирование

## Создание простого агента

Рассмотрим пример создания простого агента с использованием SmolaGents:

```python
from smolagents import LiteLLMModel, ToolCallingAgent, tool

# Создаем модель
model = LiteLLMModel(model_id="openrouter/openai/gpt-4o-mini")

# Определяем инструменты
@tool
def время() -> str:
    """
    Возвращает текущее время и дату.
    """
    from datetime import datetime
    now = datetime.now()
    return f"Текущее время: {now.strftime('%H:%M:%S')}, дата: {now.strftime('%d.%m.%Y')}"

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

# Создаем агента
agent = ToolCallingAgent(
    tools=[время, калькулятор],
    model=model,
    max_steps=5,
    verbosity_level=1
)

# Запускаем агента с запросом
result = agent("Сколько будет 2+2 и какое сейчас время?")
print(result)
```

## Настройка агента

Агенты в SmolaGents можно настраивать с помощью различных параметров:

```python
agent = ToolCallingAgent(
    tools=[...],
    model=model,
    max_steps=5,           # Максимальное количество шагов
    verbosity_level=1,     # Уровень логирования (0-3)
    max_execution_time=60, # Максимальное время выполнения в секундах
    name="math_agent",     # Имя агента
    description="Агент для математических вычислений", # Описание агента
)
```

## Практическое задание

Создайте простого агента с двумя инструментами:

1. Инструмент для генерации случайных чисел
2. Инструмент для преобразования текста (например, перевод в верхний регистр или преобразование в slug)

Затем протестируйте агента с различными запросами, которые требуют использования этих инструментов.

Пример решения:

```python
import random
import re
from smolagents import LiteLLMModel, ToolCallingAgent, tool

# Настройка модели
model = LiteLLMModel(model_id="openrouter/openai/gpt-3.5-turbo")

# Определение инструментов
@tool
def случайное_число(мин: int = 1, макс: int = 100) -> str:
    """
    Генерирует случайное число в заданном диапазоне.
    
    Args:
        мин: Минимальное значение (по умолчанию 1)
        макс: Максимальное значение (по умолчанию 100)
    """
    number = random.randint(мин, макс)
    return f"Сгенерированное случайное число: {number}"

@tool
def преобразовать_текст(текст: str, тип_преобразования: str = "верхний") -> str:
    """
    Преобразует текст в соответствии с указанным типом преобразования.
    
    Args:
        текст: Текст для преобразования
        тип_преобразования: Тип преобразования (верхний, нижний, slug)
    """
    if тип_преобразования == "верхний":
        return текст.upper()
    elif тип_преобразования == "нижний":
        return текст.lower()
    elif тип_преобразования == "slug":
        # Преобразование в slug (URL-friendly строку)
        text = текст.lower()
        text = re.sub(r'[^\w\s-]', '', text)  # Удаление специальных символов
        text = re.sub(r'\s+', '-', text)      # Замена пробелов на дефисы
        return text
    else:
        return f"Неизвестный тип преобразования: {тип_преобразования}"

# Создание агента
agent = ToolCallingAgent(
    tools=[случайное_число, преобразовать_текст],
    model=model,
    max_steps=5,
    verbosity_level=1
)

# Тестирование агента
test_queries = [
    "Сгенерируй случайное число от 1 до 10",
    "Преобразуй фразу 'Привет, мир!' в верхний регистр",
    "Создай slug из текста 'Это тестовая строка для URL'"
]

for query in test_queries:
    print(f"\nЗапрос: {query}")
    result = agent(query)
    print(f"Результат: {result}")
```

## Дополнительные ресурсы

- [GitHub репозиторий SmolaGents](https://github.com/smol-ai/smolagents)
- [Примеры использования SmolaGents](https://github.com/smol-ai/smolagents/tree/main/examples) 