# Урок 6: Типы агентов в SmolaGents

## Обзор типов агентов в SmolaGents

SmolaGents предлагает несколько типов агентов, каждый из которых имеет свои особенности и предназначение. В этом уроке мы подробно рассмотрим различные типы агентов, их возможности и сценарии использования.

Основные типы агентов в SmolaGents:
- **ToolCallingAgent** - базовый агент для вызова инструментов
- **CodeAgent** - специализированный агент для работы с кодом
- **MultiAgent** - система для координации нескольких агентов

## ToolCallingAgent и его возможности

`ToolCallingAgent` - это основной тип агента в SmolaGents, который специализируется на использовании инструментов на основе запросов пользователя.

### Создание ToolCallingAgent

```python
from smolagents import ToolCallingAgent, LiteLLMModel, tool

# Создаем модель
model = LiteLLMModel(model_id="openrouter/openai/gpt-4o-mini")

# Определяем инструмент
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
    tools=[калькулятор],
    model=model,
    max_steps=5,
    verbosity_level=1
)

# Запускаем агента с запросом
result = agent("Сколько будет 2 + 2 * 3?")
print(result)
```

### Параметры ToolCallingAgent

```python
agent = ToolCallingAgent(
    tools=[...],                 # Список инструментов
    model=model,                 # Модель LLM
    max_steps=5,                 # Максимальное количество шагов
    verbosity_level=1,           # Уровень логирования (0-3)
    max_execution_time=60,       # Максимальное время выполнения в секундах
    name="my_agent",             # Имя агента
    description="Описание агента" # Описание агента
)
```

### Особенности ToolCallingAgent

1. **Цепочки рассуждений** - агент может выполнять несколько шагов рассуждений перед вызовом инструмента
2. **Контроль выполнения** - возможность ограничить количество шагов и время выполнения
3. **Детальное логирование** - различные уровни логирования для отладки
4. **Интеграция с различными моделями** - работает с разными LLM через соответствующие адаптеры

## CodeAgent для работы с кодом

`CodeAgent` - это специализированный тип агента, оптимизированный для работы с кодом. Он имеет дополнительные возможности для анализа, генерации и модификации кода.

### Создание CodeAgent

```python
from smolagents import CodeAgent, LiteLLMModel, tool

# Создаем модель
model = LiteLLMModel(model_id="openrouter/openai/gpt-4o")

# Определяем инструмент для выполнения кода Python
@tool
def execute_python(code: str) -> str:
    """
    Выполняет код Python и возвращает результат.
    
    Args:
        code: Строка с кодом Python
    """
    try:
        # Создаем изолированное пространство выполнения
        local_vars = {}
        exec(code, {"__builtins__": __builtins__}, local_vars)
        
        # Собираем результаты
        result = []
        for var_name, var_value in local_vars.items():
            if not var_name.startswith("_"):
                result.append(f"{var_name} = {var_value}")
        
        return "\n".join(result) if result else "Код выполнен успешно, но не вернул результатов"
    except Exception as e:
        return f"Ошибка при выполнении кода: {str(e)}"

# Создаем агента
agent = CodeAgent(
    tools=[execute_python],
    model=model,
    max_steps=10,
    verbosity_level=2
)

# Запускаем агента с запросом
query = "Напиши функцию, которая находит все простые числа до заданного числа n, и покажи ее работу для n=20"
result = agent(query)
print(result)
```

### Особенности CodeAgent

1. **Специализированные промпты** - оптимизированы для работы с кодом
2. **Улучшенный парсинг кода** - корректно обрабатывает блоки кода в ответах
3. **Интеграция с инструментами разработки** - может работать с средами разработки и системами контроля версий
4. **Понимание контекста кода** - лучше работает с запросами, связанными с программированием

### Сценарии использования CodeAgent

- Генерация кода по спецификации
- Отладка и исправление ошибок
- Оптимизация кода
- Рефакторинг существующего кода
- Анализ и документирование кода

## Многоагентные системы

SmolaGents позволяет создавать системы, в которых несколько агентов взаимодействуют друг с другом для решения сложных задач.

### Создание многоагентной системы

```python
from smolagents import ToolCallingAgent, CodeAgent, LiteLLMModel, tool

# Создаем модели
model = LiteLLMModel(model_id="openrouter/openai/gpt-3.5-turbo")
code_model = LiteLLMModel(model_id="openrouter/openai/gpt-4o")

# Инструменты для поиска и анализа данных
@tool
def search_data(query: str) -> str:
    """
    Выполняет поиск данных по запросу.
    
    Args:
        query: Поисковый запрос
    """
    return f"Найденные данные по запросу '{query}': ..."

@tool
def analyze_data(data: str) -> str:
    """
    Анализирует данные и возвращает результаты.
    
    Args:
        data: Данные для анализа
    """
    return f"Результаты анализа данных: ..."

# Создаем специализированных агентов
search_agent = ToolCallingAgent(
    tools=[search_data],
    model=model,
    name="search_agent",
    description="Агент для поиска данных"
)

analysis_agent = ToolCallingAgent(
    tools=[analyze_data],
    model=model,
    name="analysis_agent",
    description="Агент для анализа данных"
)

code_agent = CodeAgent(
    tools=[],
    model=code_model,
    name="code_agent",
    description="Агент для написания кода"
)

# Создаем управляющего агента
manager_agent = CodeAgent(
    tools=[],
    model=code_model,
    managed_agents=[search_agent, analysis_agent, code_agent],
    name="manager_agent",
    description="Координирующий агент"
)

# Запускаем систему с запросом
result = manager_agent("Найди данные о погоде в Москве, проанализируй их и напиши код для построения графика температуры")
print(result)
```

### Преимущества многоагентных систем

1. **Разделение ответственности** - каждый агент специализируется на своей задаче
2. **Масштабируемость** - можно добавлять новых агентов для решения дополнительных задач
3. **Гибкость** - возможность комбинировать агентов с разными моделями и инструментами
4. **Эффективность** - сложные задачи разбиваются на более простые подзадачи

## Создание собственных типов агентов

SmolaGents позволяет создавать собственные типы агентов путем наследования от базовых классов.

### Пример создания пользовательского агента

```python
from smolagents import ToolCallingAgent, LiteLLMModel
from typing import List, Optional

class CustomAgent(ToolCallingAgent):
    """
    Пользовательский агент с дополнительными возможностями
    """
    
    def __init__(
        self,
        tools: List,
        model,
        memory_size: int = 10,
        **kwargs
    ):
        """
        Инициализация агента
        
        Args:
            tools: Список инструментов
            model: Модель LLM
            memory_size: Размер памяти (количество запомненных взаимодействий)
        """
        super().__init__(tools=tools, model=model, **kwargs)
        self.memory = []
        self.memory_size = memory_size
    
    def __call__(self, query: str) -> str:
        """
        Обработка запроса с сохранением в памяти
        
        Args:
            query: Запрос пользователя
        """
        # Добавляем запрос в память
        self.memory.append({"role": "user", "content": query})
        
        # Ограничиваем размер памяти
        if len(self.memory) > self.memory_size:
            self.memory = self.memory[-self.memory_size:]
        
        # Формируем контекст из памяти
        context = "\n".join([f"{item['role']}: {item['content']}" for item in self.memory])
        
        # Добавляем контекст к запросу
        enhanced_query = f"Контекст предыдущих взаимодействий:\n{context}\n\nЗапрос: {query}"
        
        # Обрабатываем запрос
        response = super().__call__(enhanced_query)
        
        # Сохраняем ответ в памяти
        self.memory.append({"role": "assistant", "content": response})
        
        return response
    
    def clear_memory(self):
        """Очистка памяти агента"""
        self.memory = []

# Пример использования
model = LiteLLMModel(model_id="openrouter/openai/gpt-3.5-turbo")
custom_agent = CustomAgent(tools=[], model=model, memory_size=5)

print(custom_agent("Привет, как тебя зовут?"))
print(custom_agent("Что я спрашивал в предыдущем сообщении?"))
```

## Конвейеры и последовательное выполнение

Для решения сложных задач может потребоваться последовательное выполнение нескольких агентов, где результат работы одного агента становится входными данными для другого.

### Создание конвейера агентов

```python
from smolagents import ToolCallingAgent, LiteLLMModel, tool

# Создаем модель
model = LiteLLMModel(model_id="openrouter/openai/gpt-3.5-turbo")

# Инструменты
@tool
def генерация_текста(тема: str, стиль: str) -> str:
    """
    Генерирует текст на заданную тему в определенном стиле.
    
    Args:
        тема: Тема текста
        стиль: Стиль написания (формальный, неформальный, поэтический и т.д.)
    """
    agent = ToolCallingAgent(
        tools=[],
        model=model,
        name="text_generator",
        description="Агент для генерации текста"
    )
    
    prompt = f"Сгенерируй текст на тему '{тема}' в {стиль} стиле"
    return agent(prompt)

@tool
def анализ_текста(текст: str) -> str:
    """
    Анализирует текст и возвращает информацию о его структуре.
    
    Args:
        текст: Текст для анализа
    """
    agent = ToolCallingAgent(
        tools=[],
        model=model,
        name="text_analyzer",
        description="Агент для анализа текста"
    )
    
    prompt = f"Проанализируй следующий текст и предоставь информацию о его структуре, стиле и ключевых идеях:\n\n{текст}"
    return agent(prompt)

@tool
def редактирование_текста(текст: str, рекомендации: str) -> str:
    """
    Редактирует текст на основе рекомендаций.
    
    Args:
        текст: Исходный текст
        рекомендации: Рекомендации по улучшению текста
    """
    agent = ToolCallingAgent(
        tools=[],
        model=model,
        name="text_editor",
        description="Агент для редактирования текста"
    )
    
    prompt = f"Отредактируй следующий текст в соответствии с рекомендациями:\n\nТекст: {текст}\n\nРекомендации: {рекомендации}"
    return agent(prompt)

# Создаем агент-конвейер
pipeline_agent = ToolCallingAgent(
    tools=[генерация_текста, анализ_текста, редактирование_текста],
    model=model,
    max_steps=10,
    verbosity_level=2
)

# Запускаем конвейер
result = pipeline_agent("Сгенерируй текст о космосе в научно-популярном стиле, проанализируй его, а затем отредактируй для улучшения")
print(result)
```

## Практические рекомендации по выбору типа агента

1. **ToolCallingAgent** - используйте для общих задач, требующих вызова инструментов
2. **CodeAgent** - выбирайте для задач, связанных с программированием и работой с кодом
3. **Многоагентные системы** - применяйте для сложных задач, требующих специализации и разделения ответственности
4. **Собственные агенты** - создавайте, когда стандартные агенты не обеспечивают нужной функциональности

## Практическое задание

Создайте многоагентную систему для анализа и визуализации данных, состоящую из:

1. Агента для получения данных
2. Агента для анализа данных
3. Агента для визуализации
4. Управляющего агента для координации работы

Система должна принимать запросы вида "Получи данные о продажах за последний месяц, проанализируй их и создай визуализацию".

Пример решения:

```python
from smolagents import ToolCallingAgent, CodeAgent, LiteLLMModel, tool
import pandas as pd
import random
from datetime import datetime, timedelta

# Создаем модели
model = LiteLLMModel(model_id="openrouter/openai/gpt-3.5-turbo")
code_model = LiteLLMModel(model_id="openrouter/openai/gpt-4o")

# Инструмент для генерации тестовых данных
@tool
def get_sales_data(days: int = 30) -> str:
    """
    Генерирует тестовые данные о продажах за указанное количество дней.
    
    Args:
        days: Количество дней для генерации данных
    """
    # Генерируем случайные данные
    data = []
    categories = ["Электроника", "Одежда", "Продукты", "Бытовая техника"]
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    current_date = start_date
    while current_date <= end_date:
        for category in categories:
            sales = random.randint(1000, 10000)
            items = random.randint(10, 100)
            data.append({
                "date": current_date.strftime("%Y-%m-%d"),
                "category": category,
                "sales": sales,
                "items": items
            })
        current_date += timedelta(days=1)
    
    # Преобразуем в DataFrame и в строку CSV
    df = pd.DataFrame(data)
    return df.to_csv(index=False)

# Инструмент для анализа данных
@tool
def analyze_sales_data(data: str) -> str:
    """
    Анализирует данные о продажах и возвращает статистику.
    
    Args:
        data: Данные в формате CSV
    """
    # Преобразуем строку CSV в DataFrame
    import io
    import pandas as pd
    
    df = pd.read_csv(io.StringIO(data))
    
    # Анализируем данные
    total_sales = df['sales'].sum()
    total_items = df['items'].sum()
    avg_sales_per_day = df.groupby('date')['sales'].sum().mean()
    top_category = df.groupby('category')['sales'].sum().idxmax()
    category_stats = df.groupby('category')['sales'].sum().to_dict()
    
    # Возвращаем результаты анализа
    results = {
        "total_sales": total_sales,
        "total_items": total_items,
        "avg_sales_per_day": avg_sales_per_day,
        "top_category": top_category,
        "category_stats": category_stats
    }
    
    return str(results)

# Инструмент для визуализации данных
@tool
def generate_visualization_code(data: str, analysis: str) -> str:
    """
    Генерирует код для визуализации данных о продажах.
    
    Args:
        data: Данные в формате CSV
        analysis: Результаты анализа данных
    """
    # Преобразуем строку CSV в DataFrame
    import io
    
    # Возвращаем код для визуализации
    visualization_code = """
import pandas as pd
import matplotlib.pyplot as plt
import io
import seaborn as sns
import json
from ast import literal_eval

# Загружаем данные
data = '''
""" + data + """
'''

df = pd.read_csv(io.StringIO(data))

# Анализ из предыдущего шага
analysis = literal_eval('''
""" + analysis + """
''')

# Создаем несколько визуализаций
plt.figure(figsize=(15, 10))

# График 1: Продажи по категориям
plt.subplot(2, 2, 1)
category_sales = df.groupby('category')['sales'].sum()
category_sales.plot(kind='bar', color='skyblue')
plt.title('Продажи по категориям')
plt.xlabel('Категория')
plt.ylabel('Продажи')
plt.xticks(rotation=45)

# График 2: Тренд продаж по времени
plt.subplot(2, 2, 2)
time_sales = df.groupby('date')['sales'].sum()
time_sales.plot(color='green')
plt.title('Тренд продаж по времени')
plt.xlabel('Дата')
plt.ylabel('Продажи')
plt.xticks(rotation=45)

# График 3: Распределение продаж
plt.subplot(2, 2, 3)
sns.histplot(df['sales'], kde=True, color='purple')
plt.title('Распределение продаж')
plt.xlabel('Продажи')
plt.ylabel('Частота')

# График 4: Отношение продаж к количеству проданных товаров
plt.subplot(2, 2, 4)
sns.scatterplot(data=df, x='items', y='sales', hue='category')
plt.title('Отношение продаж к количеству товаров')
plt.xlabel('Количество товаров')
plt.ylabel('Продажи')

plt.tight_layout()
plt.savefig('sales_analysis.png')
plt.show()

print("Визуализация создана и сохранена в файл sales_analysis.png")
"""
    return visualization_code

# Создаем агентов
data_agent = ToolCallingAgent(
    tools=[get_sales_data],
    model=model,
    name="data_agent",
    description="Агент для получения данных о продажах"
)

analysis_agent = ToolCallingAgent(
    tools=[analyze_sales_data],
    model=model,
    name="analysis_agent",
    description="Агент для анализа данных о продажах"
)

visualization_agent = CodeAgent(
    tools=[generate_visualization_code],
    model=code_model,
    name="visualization_agent",
    description="Агент для создания визуализации данных"
)

# Создаем управляющего агента
manager_agent = CodeAgent(
    tools=[],
    model=code_model,
    managed_agents=[data_agent, analysis_agent, visualization_agent],
    name="manager_agent",
    description="Управляющий агент для координации работы с данными о продажах"
)

# Запускаем систему
result = manager_agent("Получи данные о продажах за последний месяц, проанализируй их и создай визуализацию")
print(result)
```

## Дополнительные ресурсы

- [Документация по агентам в SmolaGents](https://github.com/smol-ai/smolagents/blob/main/docs/agents.md)
- [Примеры многоагентных систем](https://github.com/smol-ai/smolagents/tree/main/examples)
- [Сравнение разных типов агентов](https://github.com/smol-ai/smolagents/blob/main/docs/agent_comparison.md)
- [Оптимизация производительности агентов](https://github.com/smol-ai/smolagents/blob/main/docs/performance.md) 