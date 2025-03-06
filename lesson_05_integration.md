# Урок 5: Интеграция SmolaGents и LiteLLM

## Совместное использование SmolaGents и LiteLLM

Одним из главных преимуществ нашего проекта является тесная интеграция библиотек SmolaGents и LiteLLM. Такая комбинация позволяет создавать мощные агенты с доступом к различным языковым моделям и с возможностью выполнения инструментов.

В этом уроке мы рассмотрим, как SmolaGents и LiteLLM взаимодействуют друг с другом и как можно использовать их вместе для создания продвинутых приложений на основе LLM.

## Использование LiteLLMModel в SmolaGents

SmolaGents предоставляет класс `LiteLLMModel`, который является адаптером для использования моделей LiteLLM в агентах SmolaGents:

```python
from smolagents import LiteLLMModel, ToolCallingAgent

# Создание модели с использованием LiteLLM
model = LiteLLMModel(
    model_id="openrouter/openai/gpt-4o-mini",
    api_base="https://openrouter.ai/api/v1",
    api_key="YOUR_API_KEY"  # Лучше брать из переменных окружения
)

# Создание агента с этой моделью
agent = ToolCallingAgent(
    tools=[],
    model=model
)
```

Класс `LiteLLMModel` принимает те же параметры, что и функция `litellm.completion`, поэтому вы можете настраивать его так же гибко:

```python
model = LiteLLMModel(
    model_id="openrouter/openai/gpt-4o",
    temperature=0.7,
    max_tokens=2000,
    api_base="https://openrouter.ai/api/v1",
    fallbacks=["openrouter/anthropic/claude-3-haiku"]
)
```

## SimpleAgent из нашего проекта

В нашем проекте мы создали класс `SimpleAgent`, который упрощает интеграцию SmolaGents и LiteLLM. Давайте рассмотрим его реализацию в файле `ai/module/agents/simple_agent.py`:

```python
def _init_llm_agent(self):
    """
    Initialize or reinitialize the LLM agent with current tools
    """
    try:
        # Create LiteLLM model
        model = LiteLLMModel(
            model_id=self.model_id,
            api_base=self.api_base
        )

        # Create agent with tools and model
        self.llm_agent = ToolCallingAgent(
            tools=self.tools,
            model=model,
            max_steps=5,
            verbosity_level=1
        )
        logger.info(f"Agent initialized with {len(self.tools)} tools")
    except Exception as e:
        logger.error(f"Error initializing LLM agent: {e}")
        self.llm_agent = None
```

Класс SimpleAgent предоставляет следующие возможности:
1. Автоматическая инициализация модели LiteLLM
2. Простой интерфейс для добавления инструментов
3. Обработка ошибок при инициализации и выполнении
4. Встроенная интеграция с Gradio для создания веб-интерфейса

## Настройка параметров модели

При использовании LiteLLMModel в SmolaGents вы можете настраивать различные параметры модели:

```python
from smolagents import LiteLLMModel, ToolCallingAgent

# Расширенная настройка модели
model = LiteLLMModel(
    model_id="openrouter/openai/gpt-4o",
    temperature=0.2,            # Низкая температура для более детерминированных ответов
    max_tokens=1000,            # Ограничение длины ответа
    top_p=0.95,                 # Параметр top_p для семплирования
    frequency_penalty=0.5,      # Штраф за частое повторение токенов
    presence_penalty=0.5,       # Штраф за повторное использование токенов
    stop=["КОНЕЦ"],             # Токены для остановки генерации
    timeout=30,                 # Таймаут запроса в секундах
    cache_prompt=True,          # Включение кэширования
    metadata={"user_id": "123"} # Метаданные для логирования
)

# Создание агента с этой моделью
agent = ToolCallingAgent(tools=[], model=model)
```

## Работа с несколькими моделями

Вы можете использовать разные модели для разных агентов:

```python
from smolagents import LiteLLMModel, ToolCallingAgent, tool

# Определение инструментов
@tool
def погода(город: str) -> str:
    """Возвращает информацию о погоде."""
    return f"В городе {город} сейчас солнечно, температура +25°C"

# Модель для быстрых и недорогих запросов
fast_model = LiteLLMModel(model_id="openrouter/openai/gpt-3.5-turbo")

# Модель для сложных запросов, требующих глубокого понимания
smart_model = LiteLLMModel(model_id="openrouter/openai/gpt-4o")

# Создаем агентов с разными моделями
fast_agent = ToolCallingAgent(tools=[погода], model=fast_model)
smart_agent = ToolCallingAgent(tools=[погода], model=smart_model)

# Функция для выбора агента в зависимости от сложности запроса
def process_query(query: str):
    if len(query) < 50 and "сложн" not in query.lower():
        print("Используем быстрого агента")
        return fast_agent(query)
    else:
        print("Используем продвинутого агента")
        return smart_agent(query)

# Тестирование
simple_query = "Какая погода в Москве?"
complex_query = "Проанализируй климатические изменения в Москве за последний год и предскажи погоду на завтра"

print(process_query(simple_query))
print(process_query(complex_query))
```

## Обработка ошибок и отказоустойчивость

При интеграции SmolaGents и LiteLLM важно обеспечить обработку ошибок и отказоустойчивость:

```python
import time
from smolagents import LiteLLMModel, ToolCallingAgent
from litellm.exceptions import RateLimitError, Timeout, APIConnectionError

class RobustAgent:
    def __init__(self, models):
        """
        Инициализирует агента с несколькими резервными моделями.
        
        Args:
            models: Список строк с именами моделей
        """
        self.model_names = models
        self.agents = {}
        
        # Инициализация агентов с разными моделями
        for model_name in models:
            try:
                model = LiteLLMModel(model_id=model_name)
                self.agents[model_name] = ToolCallingAgent(tools=[], model=model)
                print(f"Успешно инициализирован агент с моделью {model_name}")
            except Exception as e:
                print(f"Ошибка при инициализации модели {model_name}: {e}")
    
    def run(self, query):
        """
        Выполняет запрос, используя доступные модели с автоматическим переключением
        при ошибках.
        """
        for model_name in self.model_names:
            if model_name not in self.agents:
                continue
                
            agent = self.agents[model_name]
            
            try:
                print(f"Попытка выполнения запроса с моделью {model_name}")
                start_time = time.time()
                response = agent(query)
                print(f"Время ответа: {time.time() - start_time:.2f} секунд")
                return response
            except (RateLimitError, Timeout, APIConnectionError) as e:
                print(f"Ошибка с моделью {model_name}: {e}")
                continue
            except Exception as e:
                print(f"Неожиданная ошибка с моделью {model_name}: {e}")
                continue
        
        return "Не удалось выполнить запрос ни с одной из доступных моделей."

# Пример использования
robust_agent = RobustAgent([
    "openrouter/openai/gpt-4o",
    "openrouter/anthropic/claude-3-haiku",
    "openrouter/openai/gpt-3.5-turbo"
])

response = robust_agent.run("Какая сегодня погода в Москве?")
print(f"Ответ: {response}")
```

## Мониторинг и логирование

Для эффективного отслеживания работы агентов важно настроить мониторинг и логирование:

```python
import logging
import time
from smolagents import LiteLLMModel, ToolCallingAgent

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("agent.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("agent")

class MonitoredAgent:
    def __init__(self, model_id):
        self.model_id = model_id
        self.request_count = 0
        self.total_time = 0
        self.error_count = 0
        
        # Инициализация модели и агента
        try:
            self.model = LiteLLMModel(model_id=model_id)
            self.agent = ToolCallingAgent(tools=[], model=self.model)
            logger.info(f"Агент инициализирован с моделью {model_id}")
        except Exception as e:
            logger.error(f"Ошибка при инициализации агента: {e}")
            raise
    
    def run(self, query):
        """
        Выполняет запрос с логированием и сбором метрик.
        """
        self.request_count += 1
        logger.info(f"Запрос #{self.request_count}: {query[:50]}...")
        
        try:
            start_time = time.time()
            response = self.agent(query)
            execution_time = time.time() - start_time
            
            self.total_time += execution_time
            avg_time = self.total_time / self.request_count
            
            logger.info(f"Запрос #{self.request_count} выполнен за {execution_time:.2f} сек")
            logger.info(f"Средние метрики: Время: {avg_time:.2f} сек, Ошибки: {self.error_count}/{self.request_count}")
            
            return response
        except Exception as e:
            self.error_count += 1
            logger.error(f"Запрос #{self.request_count} завершился с ошибкой: {e}")
            raise
    
    def get_metrics(self):
        """
        Возвращает метрики производительности агента.
        """
        success_rate = ((self.request_count - self.error_count) / self.request_count) * 100 if self.request_count > 0 else 0
        avg_time = self.total_time / self.request_count if self.request_count > 0 else 0
        
        return {
            "model": self.model_id,
            "requests": self.request_count,
            "errors": self.error_count,
            "success_rate": f"{success_rate:.1f}%",
            "avg_time": f"{avg_time:.2f} сек"
        }

# Пример использования
agent = MonitoredAgent("openrouter/openai/gpt-3.5-turbo")

for query in ["Привет, как дела?", "Расскажи о погоде", "Что такое Python?"]:
    try:
        print(f"\nЗапрос: {query}")
        response = agent.run(query)
        print(f"Ответ: {response}")
    except Exception as e:
        print(f"Ошибка: {e}")

print("\nМетрики производительности:")
print(agent.get_metrics())
```

## Интеграция с другими библиотеками

SmolaGents и LiteLLM могут быть интегрированы с другими библиотеками для создания более мощных решений:

### Интеграция с базами знаний (RAG)

```python
from smolagents import LiteLLMModel, ToolCallingAgent, tool
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Создаем векторную базу данных
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma(embedding_function=embeddings)

# Загружаем данные в базу (в реальном приложении вы бы загрузили свои документы)
texts = [
    "Искусственный интеллект (ИИ) — это область информатики, которая занимается разработкой интеллектуальных машин.",
    "Python — высокоуровневый язык программирования общего назначения с динамической строгой типизацией и автоматическим управлением памятью.",
    "Машинное обучение — класс методов искусственного интеллекта, характерной чертой которых является не прямое решение задачи, а обучение в процессе применения решений множества сходных задач.",
]
db.add_texts(texts)

# Создаем инструмент для поиска в базе знаний
@tool
def поиск_знаний(запрос: str, количество_результатов: int = 2) -> str:
    """
    Ищет информацию в базе знаний.
    
    Args:
        запрос: Текст запроса для поиска
        количество_результатов: Количество результатов для возврата
    """
    results = db.similarity_search(запрос, k=количество_результатов)
    if not results:
        return "Информация не найдена."
    
    return "\n\n".join([doc.page_content for doc in results])

# Создаем агента с инструментом поиска
model = LiteLLMModel(model_id="openrouter/openai/gpt-3.5-turbo")
agent = ToolCallingAgent(tools=[поиск_знаний], model=model)

# Тестируем
response = agent("Что такое искусственный интеллект?")
print(response)
```

### Интеграция с веб-API

```python
import requests
from smolagents import LiteLLMModel, ToolCallingAgent, tool

# Инструмент для поиска информации о фильмах
@tool
def информация_о_фильме(название: str) -> str:
    """
    Ищет информацию о фильме по названию.
    
    Args:
        название: Название фильма
    """
    api_key = "your_api_key"  # Замените на ваш ключ API
    url = f"http://www.omdbapi.com/?apikey={api_key}&t={название}"
    
    try:
        response = requests.get(url)
        data = response.json()
        
        if data.get("Response") == "False":
            return f"Фильм '{название}' не найден."
        
        result = f"Название: {data.get('Title')}\n"
        result += f"Год: {data.get('Year')}\n"
        result += f"Рейтинг: {data.get('imdbRating')}\n"
        result += f"Жанр: {data.get('Genre')}\n"
        result += f"Режиссер: {data.get('Director')}\n"
        result += f"Актеры: {data.get('Actors')}\n"
        result += f"Сюжет: {data.get('Plot')}"
        
        return result
    except Exception as e:
        return f"Ошибка при поиске информации о фильме: {str(e)}"

# Создаем агента с инструментом
model = LiteLLMModel(model_id="openrouter/openai/gpt-3.5-turbo")
agent = ToolCallingAgent(tools=[информация_о_фильме], model=model)

# Тестируем
response = agent("Расскажи мне о фильме 'Матрица'")
print(response)
```

## Практическое задание

Создайте агента, который объединяет возможности LiteLLM и SmolaGents для анализа данных о погоде:

1. Создайте инструмент для получения информации о погоде из публичного API
2. Добавьте инструменты для анализа погодных данных (например, расчет средней температуры)
3. Используйте LiteLLMModel с настройкой кэширования
4. Добавьте базовое логирование запросов и ответов
5. Протестируйте агента на различных запросах о погоде

Пример решения:

```python
import requests
import statistics
import logging
import time
from smolagents import LiteLLMModel, ToolCallingAgent, tool
from typing import List, Dict, Optional

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("weather_agent")

# Инструменты для работы с погодой
@tool
def текущая_погода(город: str) -> str:
    """
    Получает информацию о текущей погоде в указанном городе.
    
    Args:
        город: Название города
    """
    logger.info(f"Запрос погоды для города: {город}")
    
    try:
        # Используем публичное API OpenWeatherMap (требуется API ключ)
        api_key = "your_api_key"  # Замените на ваш ключ API
        url = f"http://api.openweathermap.org/data/2.5/weather?q={город}&appid={api_key}&units=metric&lang=ru"
        
        response = requests.get(url)
        data = response.json()
        
        if response.status_code != 200:
            return f"Ошибка при получении погоды: {data.get('message', 'Неизвестная ошибка')}"
        
        # Форматируем ответ
        temp = data["main"]["temp"]
        feels_like = data["main"]["feels_like"]
        description = data["weather"][0]["description"]
        humidity = data["main"]["humidity"]
        wind_speed = data["wind"]["speed"]
        
        result = f"Погода в городе {город}:\n"
        result += f"Температура: {temp}°C (ощущается как {feels_like}°C)\n"
        result += f"Описание: {description}\n"
        result += f"Влажность: {humidity}%\n"
        result += f"Скорость ветра: {wind_speed} м/с"
        
        logger.info(f"Успешно получена погода для города {город}")
        return result
        
    except Exception as e:
        logger.error(f"Ошибка при получении погоды: {str(e)}")
        return f"Произошла ошибка при получении данных о погоде: {str(e)}"

@tool
def прогноз_погоды(город: str, дни: int = 3) -> str:
    """
    Получает прогноз погоды на несколько дней для указанного города.
    
    Args:
        город: Название города
        дни: Количество дней для прогноза (1-5)
    """
    logger.info(f"Запрос прогноза погоды для города {город} на {дни} дней")
    
    # Ограничиваем количество дней
    дни = max(1, min(5, дни))
    
    try:
        # Используем публичное API OpenWeatherMap (требуется API ключ)
        api_key = "your_api_key"  # Замените на ваш ключ API
        url = f"http://api.openweathermap.org/data/2.5/forecast?q={город}&appid={api_key}&units=metric&lang=ru&cnt={дни*8}"
        
        response = requests.get(url)
        data = response.json()
        
        if response.status_code != 200:
            return f"Ошибка при получении прогноза погоды: {data.get('message', 'Неизвестная ошибка')}"
        
        # Группируем прогноз по дням
        forecast_by_day = {}
        for item in data["list"]:
            date = item["dt_txt"].split()[0]
            if date not in forecast_by_day:
                forecast_by_day[date] = []
            forecast_by_day[date].append(item)
        
        # Форматируем ответ
        result = f"Прогноз погоды для города {город} на {дни} дней:\n\n"
        
        for day, forecasts in list(forecast_by_day.items())[:дни]:
            # Рассчитываем средние значения для дня
            temps = [f["main"]["temp"] for f in forecasts]
            avg_temp = statistics.mean(temps)
            max_temp = max(temps)
            min_temp = min(temps)
            descriptions = [f["weather"][0]["description"] for f in forecasts]
            
            # Добавляем в результат
            result += f"Дата: {day}\n"
            result += f"Средняя температура: {avg_temp:.1f}°C (мин: {min_temp:.1f}°C, макс: {max_temp:.1f}°C)\n"
            result += f"Погодные условия: {', '.join(set(descriptions))}\n\n"
        
        logger.info(f"Успешно получен прогноз погоды для города {город}")
        return result
        
    except Exception as e:
        logger.error(f"Ошибка при получении прогноза погоды: {str(e)}")
        return f"Произошла ошибка при получении данных о прогнозе погоды: {str(e)}"

@tool
def анализ_температуры(город: str, дней: int = 5) -> str:
    """
    Анализирует тенденции изменения температуры в городе.
    
    Args:
        город: Название города
        дней: Количество дней для анализа (1-5)
    """
    logger.info(f"Запрос анализа температуры для города {город}")
    
    try:
        # Получаем прогноз погоды
        api_key = "your_api_key"  # Замените на ваш ключ API
        url = f"http://api.openweathermap.org/data/2.5/forecast?q={город}&appid={api_key}&units=metric&lang=ru"
        
        response = requests.get(url)
        data = response.json()
        
        if response.status_code != 200:
            return f"Ошибка при получении данных: {data.get('message', 'Неизвестная ошибка')}"
        
        # Извлекаем температуры
        temps = []
        dates = []
        
        for item in data["list"][:дней*8]:
            temps.append(item["main"]["temp"])
            dates.append(item["dt_txt"])
        
        # Анализируем тенденции
        avg_temp = statistics.mean(temps)
        min_temp = min(temps)
        max_temp = max(temps)
        temp_change = temps[-1] - temps[0]
        
        # Определяем тренд
        if temp_change > 3:
            trend = "существенное потепление"
        elif temp_change > 1:
            trend = "небольшое потепление"
        elif temp_change < -3:
            trend = "существенное похолодание"
        elif temp_change < -1:
            trend = "небольшое похолодание"
        else:
            trend = "стабильная температура"
        
        # Форматируем результат
        result = f"Анализ температуры для города {город}:\n"
        result += f"Средняя температура: {avg_temp:.1f}°C\n"
        result += f"Минимальная температура: {min_temp:.1f}°C\n"
        result += f"Максимальная температура: {max_temp:.1f}°C\n"
        result += f"Изменение температуры: {temp_change:.1f}°C\n"
        result += f"Тенденция: {trend}"
        
        logger.info(f"Успешно выполнен анализ температуры для города {город}")
        return result
        
    except Exception as e:
        logger.error(f"Ошибка при анализе температуры: {str(e)}")
        return f"Произошла ошибка при анализе температуры: {str(e)}"

# Создаем LiteLLM модель с кэшированием
model = LiteLLMModel(
    model_id="openrouter/openai/gpt-3.5-turbo",
    cache_prompt=True,
    temperature=0.3
)

# Создаем агента с инструментами
agent = ToolCallingAgent(
    tools=[текущая_погода, прогноз_погоды, анализ_температуры],
    model=model,
    max_steps=5,
    verbosity_level=1
)

# Тестируем агента
test_queries = [
    "Какая сейчас погода в Москве?",
    "Дай прогноз погоды в Санкт-Петербурге на 3 дня",
    "Проанализируй изменение температуры в Москве"
]

for query in test_queries:
    logger.info(f"Обработка запроса: {query}")
    start_time = time.time()
    
    try:
        response = agent(query)
        execution_time = time.time() - start_time
        
        logger.info(f"Запрос обработан за {execution_time:.2f} секунд")
        print(f"\nЗапрос: {query}")
        print(f"Ответ: {response}")
        
    except Exception as e:
        logger.error(f"Ошибка при обработке запроса: {str(e)}")
        print(f"\nЗапрос: {query}")
        print(f"Ошибка: {str(e)}")
```

## Дополнительные ресурсы

- [Документация LiteLLM по интеграции с другими библиотеками](https://docs.litellm.ai/docs/integrations)
- [Примеры использования SmolaGents](https://github.com/smol-ai/smolagents/tree/main/examples)
- [Документация по OpenRouter API](https://openrouter.ai/docs)
- [Примеры использования Tool calling в OpenAI](https://platform.openai.com/docs/guides/function-calling) 