# Урок 2: Продвинутые возможности LiteLLM

## Кэширование запросов

Одной из мощных возможностей LiteLLM является встроенное кэширование запросов, которое позволяет существенно сократить время ответа и снизить расходы на API при повторных запросах.

Для включения кэширования:

```python
from litellm.caching.caching import LiteLLMCacheType
import litellm

# Настройка дискового кэша
litellm.enable_cache(
    type=LiteLLMCacheType.DISK,
    disk_cache_dir="./cache_dir",
)
```

В нашем проекте кэширование настраивается в файле `ai/module/config/config.py`:

```python
def setup_caching():
    litellm.enable_cache(
        type=LiteLLMCacheType.DISK,
        disk_cache_dir=os.path.join(ROOT_DIR, ".cache"),
    )
    logger.debug("LiteLLM cache configured - using disk cache")
```

Доступны различные типы кэширования:
- `REDIS` - использует Redis для кэширования (хорошо подходит для распределенных систем)
- `DISK` - локальный дисковый кэш (простое решение для одного сервера)
- `MONGODB` - использует MongoDB (для долгосрочного хранения)
- `DYNAMODB` - использует AWS DynamoDB (для AWS-инфраструктуры)
- `CLOUDFLARE` - использует Cloudflare KV (для глобального быстрого доступа)

## Обработка ошибок и отказоустойчивость

LiteLLM предоставляет несколько типов исключений, которые позволяют эффективно обрабатывать различные ошибки при работе с LLM:

```python
from litellm.exceptions import RateLimitError, Timeout, APIConnectionError, BudgetExceededError, AuthenticationError

try:
    response = litellm.completion(
        model="openrouter/openai/gpt-4o",
        messages=[{"role": "user", "content": "Привет!"}]
    )
except RateLimitError:
    print("Превышен лимит запросов")
except Timeout:
    print("Истекло время ожидания ответа")
except APIConnectionError:
    print("Проблема с подключением к API")
except BudgetExceededError:
    print("Превышен бюджет")
except AuthenticationError:
    print("Ошибка аутентификации")
except Exception as e:
    print(f"Другая ошибка: {str(e)}")
```

В нашем проекте в файле `ai/module/client/litellm_client.py` можно увидеть, как обрабатываются эти исключения:

```python
try:
    # ... код отправки запроса ...
except AuthenticationError:
    logger.error("Authentication failed: Invalid API key.")
    raise ValueError("Invalid API key")
except (RateLimitError, Timeout, APIConnectionError, BudgetExceededError) as e:
    logger.warning(f"Error with model {model}: {e}. Trying next fallback if available...")
    attempt += 1
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    continue
```

## Резервные модели (fallbacks)

LiteLLM позволяет настроить резервные модели, которые будут использоваться в случае ошибок с основной моделью. Это особенно полезно для обеспечения бесперебойной работы приложения:

```python
models_to_try = ["openrouter/openai/gpt-4o-mini", "openrouter/openai/gpt-3.5-turbo"]

for model in models_to_try:
    try:
        response = litellm.completion(model=model, messages=messages)
        # Если запрос успешен, выходим из цикла
        break
    except Exception as e:
        print(f"Ошибка с моделью {model}: {e}")
```

В нашем проекте это реализовано через параметр `fallbacks` в классе `Request`:

```python
request = Request(
    model="openrouter/openai/gpt-4o",
    fallbacks=["openrouter/openai/gpt-3.5-turbo", "openrouter/anthropic/claude-3-haiku"],
    messages=[{"role": "user", "content": "Привет!"}],
    answer_model=CustomAnswer
)
```

## Настройка пользовательских провайдеров

LiteLLM позволяет создавать пользовательские провайдеры для работы с нестандартными API или с локальными моделями. Для этого используется класс `CustomLLM`:

```python
from litellm.llms.custom_llm import CustomLLM
import litellm
import requests

class MyCustomProvider(CustomLLM):
    def completion(self, model: str, messages: list, **kwargs):
        # Преобразуем сообщения в один промпт
        prompt = "\n".join([m["content"] for m in messages])
        
        # Отправляем запрос к нашему API
        response = requests.post(
            "https://my-custom-api.example.com/generate",
            json={"prompt": prompt, "max_tokens": kwargs.get("max_tokens", 100)}
        )
        
        # Форматируем ответ в структуру, ожидаемую LiteLLM
        result = response.json()
        return {
            "choices": [{
                "message": {
                    "content": result["generated_text"],
                    "role": "assistant"
                }
            }]
        }

# Регистрируем провайдер
custom_provider = MyCustomProvider()
litellm.custom_provider_map = [
    {"provider": "my_custom_provider", "custom_handler": custom_provider}
]

# Теперь можно использовать этот провайдер
response = litellm.completion(
    model="my_custom_provider/my-model",
    messages=[{"role": "user", "content": "Привет!"}]
)
```

В нашем проекте есть пример пользовательского провайдера в файле `ai/module/providers/custom_provider.py`.

## Балансировка нагрузки и роутинг

LiteLLM предоставляет функциональность для балансировки нагрузки между моделями и роутинга запросов:

```python
import litellm
from litellm import Router

model_list = [
    {
        "model_name": "gpt-3.5-turbo",
        "litellm_params": {
            "model": "openrouter/openai/gpt-3.5-turbo",
            "api_key": os.getenv("OPENROUTER_API_KEY"),
        },
        "tpm": 100000,
        "rpm": 10000,
    },
    {
        "model_name": "gpt-3.5-turbo",
        "litellm_params": {
            "model": "openai/gpt-3.5-turbo",
            "api_key": os.getenv("OPENAI_API_KEY"),
        },
        "tpm": 100000,
        "rpm": 10000,
    }
]

# Настройка роутера
router = Router(model_list=model_list)

# Отправка запроса через роутер
response = router.completion(
    model="gpt-3.5-turbo", 
    messages=[{"role": "user", "content": "Привет!"}]
)
```

## Бюджетирование и контроль расходов

LiteLLM позволяет устанавливать бюджетные ограничения на использование API:

```python
import litellm

# Установка максимального бюджета
litellm.max_budget = 10.0  # В долларах

try:
    response = litellm.completion(
        model="openrouter/openai/gpt-4o",
        messages=[{"role": "user", "content": "Напиши длинный рассказ"}]
    )
except litellm.exceptions.BudgetExceededError:
    print("Превышен установленный бюджет")
```

## Практическое задание

Создайте скрипт, который демонстрирует использование кэширования и резервных моделей:

1. Настройте кэширование запросов на диск
2. Создайте список моделей с основной и резервными моделями
3. Отправьте один и тот же запрос дважды и измерьте время ответа для каждого запроса
4. Проверьте, что второй запрос выполняется быстрее из-за кэширования
5. Искусственно вызовите ошибку для основной модели и убедитесь, что запрос обрабатывается резервной моделью

Пример решения:

```python
import time
import litellm
from litellm.caching.caching import LiteLLMCacheType

# Настраиваем кэширование
litellm.enable_cache(
    type=LiteLLMCacheType.DISK,
    disk_cache_dir="./cache_dir",
)

# Список моделей для тестирования
models = ["openrouter/openai/gpt-4o-mini", "openrouter/openai/gpt-3.5-turbo"]
messages = [{"role": "user", "content": "Что такое Python?"}]

# Первый запрос (без кэша)
start_time = time.time()
try:
    response = litellm.completion(model=models[0], messages=messages)
    first_model = models[0]
    print("Ответ от первой модели:", response.choices[0].message.content[:100] + "...")
except Exception as e:
    print(f"Ошибка с первой моделью: {e}")
    response = litellm.completion(model=models[1], messages=messages)
    first_model = models[1]
    print("Ответ от резервной модели:", response.choices[0].message.content[:100] + "...")

first_request_time = time.time() - start_time
print(f"Время первого запроса: {first_request_time:.2f} секунд (модель: {first_model})")

# Второй запрос (должен использовать кэш)
start_time = time.time()
response = litellm.completion(model=models[0], messages=messages)
second_request_time = time.time() - start_time

print(f"Время второго запроса: {second_request_time:.2f} секунд (модель: {models[0]})")
print(f"Ускорение благодаря кэшированию: {first_request_time / second_request_time:.2f}x")

# Проверка работы с некорректной моделью
try:
    # Искусственно создаем ошибку, используя несуществующую модель
    litellm.completion(model="несуществующая-модель", messages=messages)
except Exception as e:
    print(f"Ожидаемая ошибка с некорректной моделью: {e}")
```

## Дополнительные ресурсы

- [Документация по кэшированию в LiteLLM](https://docs.litellm.ai/docs/caching)
- [Обработка ошибок в LiteLLM](https://docs.litellm.ai/docs/exception_mapping)
- [Балансировка нагрузки](https://docs.litellm.ai/docs/routing) 