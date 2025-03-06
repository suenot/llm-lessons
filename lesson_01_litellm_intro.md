# Урок 1: Введение в LiteLLM и его роль в проекте

## Что такое LiteLLM

**LiteLLM** - это унифицированный интерфейс для работы с различными провайдерами языковых моделей (LLM). Библиотека позволяет использовать единый API для взаимодействия с моделями от разных поставщиков, таких как OpenAI, Anthropic, Mistral, HuggingFace и других.

Ключевые возможности LiteLLM:
- Унифицированный API в стиле OpenAI для всех LLM
- Поддержка множества провайдеров и моделей
- Встроенное кэширование запросов
- Механизмы отказоустойчивости и резервных моделей
- Поддержка прокси-серверов и балансировки нагрузки

## Установка и базовая настройка

Для установки LiteLLM используйте pip:

```bash
pip install litellm
```

## Настройка окружения

Для работы с большинством LLM-провайдеров требуются API ключи. В нашем проекте используется OpenRouter, который предоставляет доступ к различным моделям через единый API:

```bash
export OPENROUTER_API_KEY="ваш_api_ключ"
```

Вы можете получить ключ API на сайте [OpenRouter](https://openrouter.ai/).

## Базовое использование LiteLLM

Простой пример использования LiteLLM для отправки запроса к модели:

```python
from litellm import completion

response = completion(
    model="openrouter/openai/gpt-4o-mini",  # Модель от OpenAI через OpenRouter
    messages=[{"role": "user", "content": "Привет, как дела?"}]
)

print(response.choices[0].message.content)
```

Этот код отправляет простой запрос к модели GPT-4o Mini через OpenRouter и выводит полученный ответ.

## Класс LiteLLMClient в проекте

В нашем проекте библиотека LiteLLM используется через класс `LiteLLMClient`, который находится в файле `ai/module/client/litellm_client.py`. Этот класс инкапсулирует функциональность LiteLLM и предоставляет удобный интерфейс для работы с моделями:

```python
from module import Request, LiteLLMClient
from pydantic import BaseModel, Field

class CustomAnswer(BaseModel):
    content: str = Field(..., description="Основное содержание ответа")
    keywords: list = Field(default_factory=list, description="Ключевые слова из ответа")

# Создаем запрос
request = Request(
    model="openrouter/openai/gpt-4o-mini",
    messages=[{"role": "user", "content": "Расскажи о погоде в Москве"}],
    answer_model=CustomAnswer
)

# Создаем клиент и получаем ответ
client = LiteLLMClient(request)
response = client.generate_response()

print(response.content)
print("Ключевые слова:", response.keywords)
```

Класс `LiteLLMClient` добавляет следующую функциональность поверх базового API LiteLLM:
- Валидация запросов и ответов с помощью Pydantic
- Обработка ошибок и автоматические повторные попытки
- Поддержка переключения на резервные модели при ошибках
- Встроенное логирование и сбор метрик

## Практическое задание

Создайте простой скрипт, который использует базовый API LiteLLM для отправки запроса к модели и обработки ответа:

1. Установите LiteLLM и настройте API ключ для OpenRouter
2. Создайте скрипт, который отправляет запрос к модели gpt-3.5-turbo
3. Попросите модель ответить на вопрос о том, что такое LiteLLM
4. Выведите ответ модели и измерьте время выполнения запроса

Пример решения:

```python
import time
from litellm import completion

# Засекаем время начала выполнения
start_time = time.time()

response = completion(
    model="openrouter/openai/gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Что такое LiteLLM? Ответь кратко."}]
)

# Вычисляем время выполнения
execution_time = time.time() - start_time

print("Ответ:", response.choices[0].message.content)
print(f"Время выполнения запроса: {execution_time:.2f} секунд")
```

## Дополнительные ресурсы

- [Официальная документация LiteLLM](https://docs.litellm.ai/)
- [GitHub репозиторий](https://github.com/BerriAI/litellm)
- [Список поддерживаемых провайдеров](https://docs.litellm.ai/docs/providers) 