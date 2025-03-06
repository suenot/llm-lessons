# Урок 9: Системы поиска информации (RAG) с SmolaGents

## Введение в RAG

**Retrieval-Augmented Generation (RAG)** — это мощный подход, объединяющий возможности поиска информации и генеративные возможности языковых моделей. RAG-системы позволяют агентам искать и использовать информацию из внешних источников данных, существенно расширяя контекст и знания, доступные модели при генерации ответов.

### Преимущества RAG-систем:

1. **Актуальность информации** — доступ к данным, которые не были включены в обучающую выборку модели
2. **Снижение галлюцинаций** — генерация ответов на основе фактической информации
3. **Предметная специализация** — возможность работать с узкоспециализированной информацией
4. **Прозрачность** — возможность отслеживать источники информации
5. **Масштабируемость** — возможность работать с большими объемами данных

## Архитектура RAG-систем

Типичная RAG-система состоит из следующих компонентов:

1. **Хранилище документов** — база для хранения текстовых данных
2. **Векторная база данных** — хранилище для векторных представлений (эмбеддингов) текстов
3. **Модель эмбеддингов** — преобразует тексты в векторные представления
4. **Поисковый движок** — находит релевантные документы по запросу
5. **LLM** — генерирует ответы на основе найденной информации и запроса

## Настройка окружения для работы с RAG

Для работы с RAG-системами в SmolaGents нам потребуются дополнительные библиотеки:

```bash
pip install langchain chroma-db sentence-transformers
```

## Создание базовой RAG-системы с SmolaGents

Рассмотрим, как создать простую RAG-систему с использованием SmolaGents:

```python
import os
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from smolagents import ToolCallingAgent, LiteLLMModel, tool

# Шаг 1: Загрузка документов
def load_documents(directory):
    loader = DirectoryLoader(directory, glob="**/*.txt", loader_cls=TextLoader)
    documents = loader.load()
    print(f"Загружено {len(documents)} документов")
    return documents

# Шаг 2: Разделение документов на чанки
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Создано {len(chunks)} текстовых фрагментов")
    return chunks

# Шаг 3: Создание векторного хранилища
def create_vector_store(chunks):
    # Инициализируем модель эмбеддингов
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-large"
    )
    
    # Создаем векторное хранилище
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    
    return vectorstore

# Шаг 4: Создание инструмента для поиска информации
@tool
def поиск_информации(запрос: str, k: int = 3) -> str:
    """
    Ищет информацию в базе данных документов по запросу.
    
    Args:
        запрос: Поисковый запрос
        k: Количество результатов для возврата
    """
    # Получаем доступ к векторному хранилищу
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-large"
    )
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )
    
    # Выполняем поиск
    results = vectorstore.similarity_search(запрос, k=k)
    
    # Форматируем результаты
    formatted_results = ""
    for i, doc in enumerate(results):
        formatted_results += f"### Фрагмент {i+1}:\n"
        formatted_results += f"{doc.page_content}\n\n"
        formatted_results += f"Источник: {doc.metadata.get('source', 'Неизвестен')}\n\n"
    
    return formatted_results

# Функция для подготовки базы знаний
def prepare_knowledge_base(directory):
    documents = load_documents(directory)
    chunks = split_documents(documents)
    vectorstore = create_vector_store(chunks)
    return vectorstore

# Создаем модель и агента
model = LiteLLMModel(model_id="openrouter/openai/gpt-4o")
agent = ToolCallingAgent(
    tools=[поиск_информации],
    model=model,
    max_steps=5,
    verbosity_level=1
)

# Пример использования: подготавливаем базу знаний и задаем вопрос
if __name__ == "__main__":
    # Подготавливаем базу знаний (выполняем только один раз)
    # prepare_knowledge_base("./documents")
    
    # Задаем вопрос агенту
    query = "Какие основные принципы работы солнечных батарей?"
    response = agent(query)
    print(f"Вопрос: {query}")
    print(f"Ответ: {response}")
```

## Работа с различными источниками данных

В RAG-системах можно использовать различные источники данных. Рассмотрим наиболее распространенные:

### Текстовые файлы

```python
from langchain.document_loaders import TextLoader, DirectoryLoader

# Загрузка одного текстового файла
loader = TextLoader("path/to/file.txt")
documents = loader.load()

# Загрузка всех текстовых файлов из директории
loader = DirectoryLoader("./documents", glob="**/*.txt", loader_cls=TextLoader)
documents = loader.load()
```

### PDF-документы

```python
from langchain.document_loaders import PyPDFLoader, DirectoryLoader

# Загрузка одного PDF-файла
loader = PyPDFLoader("path/to/document.pdf")
documents = loader.load()

# Загрузка всех PDF из директории
loader = DirectoryLoader("./documents", glob="**/*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()
```

### Веб-страницы

```python
from langchain.document_loaders import WebBaseLoader

# Загрузка одной веб-страницы
loader = WebBaseLoader("https://example.com")
documents = loader.load()

# Загрузка нескольких веб-страниц
urls = ["https://site1.com", "https://site2.com"]
loader = WebBaseLoader(urls)
documents = loader.load()
```

## Различные векторные базы данных

Для хранения векторных представлений документов можно использовать различные базы данных:

### Chroma DB

```python
from langchain.vectorstores import Chroma

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)
```

### FAISS

```python
from langchain.vectorstores import FAISS
import pickle

# Создание хранилища
vectorstore = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings
)

# Сохранение
faiss_index = vectorstore.index
pickle.dump(faiss_index, open("faiss_index.pkl", "wb"))

# Загрузка
loaded_index = pickle.load(open("faiss_index.pkl", "rb"))
```

### Pinecone (облачное решение)

```python
import pinecone
from langchain.vectorstores import Pinecone

pinecone.init(
    api_key="your-api-key",
    environment="us-west-2"
)

index_name = "your-index-name"
vectorstore = Pinecone.from_documents(
    documents=chunks,
    embedding=embeddings, 
    index_name=index_name
)
```

## Создание продвинутой RAG-системы с инструментами SmolaGents

Теперь создадим более продвинутую RAG-систему, которая включает:
- Загрузку документов из разных источников
- Оценку релевантности найденной информации
- Возможность добавления новых документов в базу знаний

```python
from smolagents import ToolCallingAgent, LiteLLMModel, tool
from langchain.document_loaders import TextLoader, PyPDFLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
import os
import requests
from datetime import datetime

# Инициализация модели эмбеддингов
embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-large"
)

# Функция для создания или загрузки векторного хранилища
def get_vector_store():
    if os.path.exists("./chroma_db"):
        return Chroma(
            persist_directory="./chroma_db",
            embedding_function=embeddings
        )
    else:
        return None

# Инструменты для работы с документами
@tool
def поиск_информации(запрос: str, k: int = 5, порог_релевантности: float = 0.3) -> str:
    """
    Ищет информацию в базе данных документов по запросу.
    
    Args:
        запрос: Поисковый запрос
        k: Количество результатов для возврата
        порог_релевантности: Минимальный порог релевантности результатов (от 0 до 1)
    """
    vectorstore = get_vector_store()
    if not vectorstore:
        return "База данных не проиндексирована. Используйте индексировать_документы для создания базы."
    
    # Выполняем поиск с указанием оценки релевантности
    results_with_scores = vectorstore.similarity_search_with_relevance_scores(запрос, k=k)
    
    # Фильтруем результаты по порогу релевантности
    filtered_results = [(doc, score) for doc, score in results_with_scores if score >= порог_релевантности]
    
    # Если нет релевантных результатов
    if not filtered_results:
        return f"Не найдено релевантных результатов по запросу '{запрос}' с порогом {порог_релевантности}."
    
    # Форматируем результаты
    formatted_results = f"Найдено {len(filtered_results)} релевантных результатов:\n\n"
    for i, (doc, score) in enumerate(filtered_results):
        formatted_results += f"### Фрагмент {i+1} (релевантность: {score:.2f}):\n"
        formatted_results += f"{doc.page_content}\n\n"
        formatted_results += f"Источник: {doc.metadata.get('source', 'Неизвестен')}\n\n"
    
    return formatted_results

@tool
def индексировать_текстовый_файл(путь_к_файлу: str) -> str:
    """
    Индексирует текстовый файл и добавляет его в базу знаний.
    
    Args:
        путь_к_файлу: Путь к текстовому файлу для индексации
    """
    try:
        # Проверяем существование файла
        if not os.path.exists(путь_к_файлу):
            return f"Файл {путь_к_файлу} не найден."
        
        # Загружаем документ
        loader = TextLoader(путь_к_файлу)
        documents = loader.load()
        
        # Разделяем на чанки
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)
        
        # Получаем или создаем векторное хранилище
        if os.path.exists("./chroma_db"):
            vectorstore = Chroma(
                persist_directory="./chroma_db",
                embedding_function=embeddings
            )
            # Добавляем документы в существующую базу
            vectorstore.add_documents(chunks)
        else:
            # Создаем новую базу
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory="./chroma_db"
            )
        
        return f"Файл {путь_к_файлу} успешно проиндексирован. Добавлено {len(chunks)} фрагментов."
    
    except Exception as e:
        return f"Ошибка при индексации файла: {str(e)}"

@tool
def индексировать_pdf(путь_к_файлу: str) -> str:
    """
    Индексирует PDF-файл и добавляет его в базу знаний.
    
    Args:
        путь_к_файлу: Путь к PDF-файлу для индексации
    """
    try:
        # Проверяем существование файла
        if not os.path.exists(путь_к_файлу):
            return f"Файл {путь_к_файлу} не найден."
        
        # Загружаем документ
        loader = PyPDFLoader(путь_к_файлу)
        documents = loader.load()
        
        # Разделяем на чанки
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)
        
        # Получаем или создаем векторное хранилище
        if os.path.exists("./chroma_db"):
            vectorstore = Chroma(
                persist_directory="./chroma_db",
                embedding_function=embeddings
            )
            # Добавляем документы в существующую базу
            vectorstore.add_documents(chunks)
        else:
            # Создаем новую базу
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory="./chroma_db"
            )
        
        return f"PDF-файл {путь_к_файлу} успешно проиндексирован. Добавлено {len(chunks)} фрагментов."
    
    except Exception as e:
        return f"Ошибка при индексации PDF-файла: {str(e)}"

@tool
def индексировать_веб_страницу(url: str) -> str:
    """
    Индексирует веб-страницу и добавляет ее в базу знаний.
    
    Args:
        url: URL веб-страницы для индексации
    """
    try:
        # Проверяем доступность URL
        response = requests.head(url)
        if response.status_code != 200:
            return f"Не удалось получить доступ к URL {url}, код ответа: {response.status_code}"
        
        # Загружаем документ
        loader = WebBaseLoader(url)
        documents = loader.load()
        
        # Добавляем метаданные
        for doc in documents:
            doc.metadata["source"] = url
            doc.metadata["date_indexed"] = datetime.now().isoformat()
        
        # Разделяем на чанки
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)
        
        # Получаем или создаем векторное хранилище
        if os.path.exists("./chroma_db"):
            vectorstore = Chroma(
                persist_directory="./chroma_db",
                embedding_function=embeddings
            )
            # Добавляем документы в существующую базу
            vectorstore.add_documents(chunks)
        else:
            # Создаем новую базу
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory="./chroma_db"
            )
        
        return f"Веб-страница {url} успешно проиндексирована. Добавлено {len(chunks)} фрагментов."
    
    except Exception as e:
        return f"Ошибка при индексации веб-страницы: {str(e)}"

@tool
def статистика_базы_знаний() -> str:
    """
    Возвращает статистику по базе знаний.
    """
    try:
        if not os.path.exists("./chroma_db"):
            return "База данных не создана."
        
        vectorstore = Chroma(
            persist_directory="./chroma_db",
            embedding_function=embeddings
        )
        
        collection = vectorstore._collection
        count = collection.count()
        
        return f"""Статистика базы знаний:
- Количество документов: {count}
- Путь к базе данных: {os.path.abspath('./chroma_db')}
- Размер базы: {sum(os.path.getsize(os.path.join('./chroma_db', f)) for f in os.listdir('./chroma_db') if os.path.isfile(os.path.join('./chroma_db', f))) / (1024*1024):.2f} MB
"""
    
    except Exception as e:
        return f"Ошибка при получении статистики: {str(e)}"

# Создаем модель и агента
model = LiteLLMModel(model_id="openrouter/openai/gpt-4")
rag_agent = ToolCallingAgent(
    tools=[
        поиск_информации,
        индексировать_текстовый_файл,
        индексировать_pdf,
        индексировать_веб_страницу,
        статистика_базы_знаний
    ],
    model=model,
    max_steps=5,
    verbosity_level=1
)

# Пример использования
if __name__ == "__main__":
    # Проверяем статистику базы знаний
    print(rag_agent("Какая статистика базы знаний?"))
    
    # Индексируем веб-страницу
    print(rag_agent("Проиндексируй веб-страницу https://ru.wikipedia.org/wiki/Солнечная_батарея"))
    
    # Задаем вопрос на основе проиндексированных данных
    print(rag_agent("Расскажи о принципах работы солнечных батарей"))
```

## Оптимизация RAG-систем

Существует несколько способов оптимизации RAG-систем:

### 1. Оптимальное разделение документов на чанки

Размер и перекрытие чанков значительно влияют на качество поиска:

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Общее правило: более короткие чанки для точного поиска,
# более длинные для сохранения контекста
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      # Размер чанка в символах
    chunk_overlap=100,   # Перекрытие чанков
    separators=["\n\n", "\n", " ", ""]  # Приоритетные разделители
)
```

### 2. Выбор оптимальной модели эмбеддингов

От выбора модели эмбеддингов зависит качество поиска:

```python
# Высокая точность, но требуется много ресурсов
embeddings_high_quality = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-large"
)

# Баланс между качеством и скоростью
embeddings_balanced = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-base"
)

# Быстрая, но менее точная модель
embeddings_fast = HuggingFaceEmbeddings(
    model_name="distiluse-base-multilingual-cased-v1"
)
```

### 3. Переранжирование результатов

Переранжирование позволяет улучшить релевантность результатов:

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# Создаем базовый ретривер
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# Создаем компрессор, использующий LLM для извлечения релевантных частей
llm = LiteLLMModel(model_id="openrouter/openai/gpt-3.5-turbo")
compressor = LLMChainExtractor.from_llm(llm)

# Создаем ретривер с компрессией
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)

# Поиск с переранжированием
compressed_docs = compression_retriever.get_relevant_documents(query)
```

### 4. Кэширование запросов

Кэширование результатов запросов может существенно повысить производительность:

```python
import hashlib
import pickle
import os

def cached_search(vectorstore, query, k=5, cache_dir="./cache"):
    # Создаем директорию для кэша, если она не существует
    os.makedirs(cache_dir, exist_ok=True)
    
    # Создаем хэш запроса
    query_hash = hashlib.md5(query.encode()).hexdigest()
    cache_file = os.path.join(cache_dir, f"{query_hash}.pkl")
    
    # Проверяем наличие кэша
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    
    # Выполняем поиск и сохраняем результаты в кэш
    results = vectorstore.similarity_search(query, k=k)
    with open(cache_file, "wb") as f:
        pickle.dump(results, f)
    
    return results
```

## Интеграция RAG с LangChain Chain

LangChain предоставляет готовые цепочки для работы с RAG-системами:

```python
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# Создаем ретривер из векторного хранилища
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Создаем цепочку для ответов на вопросы
qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    chain_type="stuff",  # Другие варианты: "map_reduce", "refine"
    retriever=retriever
)

# Используем цепочку для ответа на вопрос
result = qa_chain.run("Какие основные принципы работы солнечных батарей?")
print(result)
```

## Создание инструмента RAG с метаданными и фильтрацией

Для более продвинутых сценариев можно создать инструмент, который позволяет фильтровать результаты по метаданным:

```python
@tool
def поиск_с_фильтрацией(запрос: str, фильтры: dict = None, k: int = 5) -> str:
    """
    Ищет информацию в базе данных документов по запросу с фильтрацией по метаданным.
    
    Args:
        запрос: Поисковый запрос
        фильтры: Словарь с фильтрами по метаданным, например {"author": "Иванов", "category": "Физика"}
        k: Количество результатов для возврата
    """
    vectorstore = get_vector_store()
    if not vectorstore:
        return "База данных не проиндексирована."
    
    # Создаем фильтр для Chroma
    filter_dict = {}
    if фильтры:
        for key, value in фильтры.items():
            filter_dict[f"metadata.{key}"] = value
    
    # Выполняем поиск с фильтрацией
    if filter_dict:
        results = vectorstore.similarity_search(
            запрос, 
            k=k,
            filter=filter_dict
        )
    else:
        results = vectorstore.similarity_search(запрос, k=k)
    
    # Форматируем результаты
    formatted_results = ""
    for i, doc in enumerate(results):
        formatted_results += f"### Фрагмент {i+1}:\n"
        formatted_results += f"{doc.page_content}\n\n"
        formatted_results += "Метаданные:\n"
        for key, value in doc.metadata.items():
            formatted_results += f"- {key}: {value}\n"
        formatted_results += "\n"
    
    return formatted_results
```

## Практическое задание

Создайте собственную RAG-систему для ответов на вопросы по специализированной теме:

1. Выберите тему (например, медицина, право, технологии)
2. Соберите и проиндексируйте документы по этой теме
3. Создайте агента с инструментами для поиска информации и ответов на вопросы
4. Оптимизируйте систему для достижения наилучших результатов

Пример решения:

```python
import os
from smolagents import ToolCallingAgent, LiteLLMModel, tool
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
import requests
from datetime import datetime

# Тема: Искусственный интеллект
# Определяем URLs для индексации
AI_URLS = [
    "https://ru.wikipedia.org/wiki/Искусственный_интеллект",
    "https://ru.wikipedia.org/wiki/Машинное_обучение",
    "https://ru.wikipedia.org/wiki/Глубокое_обучение",
    "https://ru.wikipedia.org/wiki/Нейронная_сеть"
]

# Настройка модели эмбеддингов
embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-base"
)

# Функция для подготовки базы знаний
def prepare_knowledge_base():
    # Загружаем документы из URL
    documents = []
    for url in AI_URLS:
        try:
            loader = WebBaseLoader(url)
            docs = loader.load()
            # Добавляем метаданные
            for doc in docs:
                doc.metadata["source"] = url
                doc.metadata["topic"] = "Искусственный интеллект"
                doc.metadata["date_indexed"] = datetime.now().isoformat()
            documents.extend(docs)
            print(f"Загружен документ с {url}")
        except Exception as e:
            print(f"Ошибка при загрузке {url}: {str(e)}")
    
    # Разделяем на чанки
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Создано {len(chunks)} текстовых фрагментов")
    
    # Создаем векторное хранилище
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./ai_knowledge_base"
    )
    
    return vectorstore

# Создаем инструменты для агента
@tool
def поиск_информации_по_ии(запрос: str, k: int = 5) -> str:
    """
    Ищет информацию по искусственному интеллекту и машинному обучению.
    
    Args:
        запрос: Поисковый запрос по теме ИИ
        k: Количество результатов для возврата
    """
    # Загружаем хранилище
    vectorstore = Chroma(
        persist_directory="./ai_knowledge_base",
        embedding_function=embeddings
    )
    
    # Выполняем поиск
    results = vectorstore.similarity_search(запрос, k=k)
    
    # Форматируем результаты
    formatted_results = ""
    for i, doc in enumerate(results):
        formatted_results += f"### Фрагмент {i+1}:\n"
        formatted_results += f"{doc.page_content}\n\n"
        formatted_results += f"Источник: {doc.metadata.get('source', 'Неизвестен')}\n\n"
    
    return formatted_results

# Создаем модель и агента
model = LiteLLMModel(model_id="openrouter/openai/gpt-4o")
ai_expert_agent = ToolCallingAgent(
    tools=[поиск_информации_по_ии],
    model=model,
    max_steps=3,
    name="AI Expert",
    description="Эксперт по искусственному интеллекту и машинному обучению"
)

# Главная функция
if __name__ == "__main__":
    # Проверяем, существует ли база знаний
    if not os.path.exists("./ai_knowledge_base"):
        print("Создаем базу знаний по искусственному интеллекту...")
        prepare_knowledge_base()
    
    # Примеры вопросов для агента
    questions = [
        "Что такое искусственный интеллект?",
        "Чем отличается машинное обучение от глубокого обучения?",
        "Как работают нейронные сети?",
        "Какие проблемы существуют в современном ИИ?"
    ]
    
    # Задаем вопросы агенту
    for question in questions:
        print(f"\nВопрос: {question}")
        response = ai_expert_agent(question)
        print(f"Ответ: {response}")
```

## Дополнительные ресурсы

- [Документация LangChain по RAG-системам](https://python.langchain.com/docs/use_cases/question_answering/)
- [Документация ChromaDB](https://docs.trychroma.com/)
- [Репозиторий HuggingFace sentence-transformers](https://github.com/UKPLab/sentence-transformers)
- [LlamaIndex - альтернативная библиотека для RAG](https://docs.llamaindex.ai/en/stable/)
- [Статья "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"](https://arxiv.org/abs/2005.11401) 