# Урок 10: Продвинутые инструменты и API интеграции

## Создание сложных инструментов

В этом уроке мы рассмотрим, как создавать продвинутые инструменты и интегрировать внешние API в ваших агентов SmolaGents. Продвинутые инструменты могут существенно расширить возможности ваших агентов, позволяя им выполнять специализированные задачи.

### Композиция инструментов

Один из способов создания сложных инструментов — это композиция, то есть создание инструмента, который использует другие инструменты:

```python
from smolagents import tool

@tool
def get_temperature(city: str) -> float:
    """
    Получает текущую температуру в указанном городе.
    
    Args:
        city: Название города
    """
    # Реализация API запроса
    return 25.5  # Условный результат

@tool
def convert_temperature(celsius: float, system: str) -> float:
    """
    Конвертирует температуру из Цельсия в другую систему.
    
    Args:
        celsius: Температура в градусах Цельсия
        system: Целевая система (fahrenheit/kelvin)
    """
    if system.lower() == "fahrenheit":
        return celsius * 9/5 + 32
    elif system.lower() == "kelvin":
        return celsius + 273.15
    else:
        return celsius

@tool
def weather_with_conversion(city: str, temperature_system: str = "celsius") -> str:
    """
    Получает текущую погоду в указанном городе с конвертацией температуры.
    
    Args:
        city: Название города
        temperature_system: Система измерения температуры (celsius/fahrenheit/kelvin)
    """
    temp_c = get_temperature(city)
    
    if temperature_system.lower() == "celsius":
        result = f"{temp_c}°C"
    else:
        temp_converted = convert_temperature(temp_c, temperature_system)
        
        if temperature_system.lower() == "fahrenheit":
            result = f"{temp_converted}°F"
        elif temperature_system.lower() == "kelvin":
            result = f"{temp_converted}K"
        else:
            result = f"{temp_c}°C"
    
    return f"Температура в городе {city}: {result}"
```

### Инструменты с состоянием

Иногда требуется создать инструмент, который сохраняет состояние между вызовами:

```python
from smolagents import tool

class SessionManager:
    def __init__(self):
        self.sessions = {}
    
    def create_session(self, user_id):
        if user_id not in self.sessions:
            self.sessions[user_id] = {"history": [], "preferences": {}}
        return f"Сессия для пользователя {user_id} создана"
    
    def add_to_history(self, user_id, query):
        if user_id in self.sessions:
            self.sessions[user_id]["history"].append(query)
            return f"Запрос добавлен в историю пользователя {user_id}"
        return f"Ошибка: сессия для пользователя {user_id} не найдена"
    
    def get_history(self, user_id):
        if user_id in self.sessions:
            return self.sessions[user_id]["history"]
        return []
    
    def set_preference(self, user_id, key, value):
        if user_id in self.sessions:
            self.sessions[user_id]["preferences"][key] = value
            return f"Настройка {key}={value} сохранена для пользователя {user_id}"
        return f"Ошибка: сессия для пользователя {user_id} не найдена"

# Создаем глобальный экземпляр менеджера сессий
session_manager = SessionManager()

@tool
def create_session(user_id: str) -> str:
    """
    Создает новую пользовательскую сессию.
    
    Args:
        user_id: Уникальный ID пользователя
    """
    return session_manager.create_session(user_id)

@tool
def add_to_history(user_id: str, query: str) -> str:
    """
    Добавляет запрос в историю пользователя.
    
    Args:
        user_id: ID пользователя
        query: Запрос для добавления в историю
    """
    return session_manager.add_to_history(user_id, query)

@tool
def get_history(user_id: str) -> list:
    """
    Возвращает историю запросов пользователя.
    
    Args:
        user_id: ID пользователя
    """
    history = session_manager.get_history(user_id)
    return f"История запросов: {history}"
```

## Интеграция с внешними API

Интеграция с внешними API позволяет значительно расширить возможности агентов. Рассмотрим несколько примеров:

### Погодные API

```python
import requests
from smolagents import tool

# Получите свой ключ API на сайте weatherapi.com
WEATHER_API_KEY = "ваш_api_ключ"

@tool
def get_weather_forecast(city: str, days: int = 1) -> str:
    """
    Получает прогноз погоды для указанного города.
    
    Args:
        city: Название города
        days: Количество дней для прогноза (1-7)
    """
    try:
        url = f"http://api.weatherapi.com/v1/forecast.json?key={WEATHER_API_KEY}&q={city}&days={days}&lang=ru"
        response = requests.get(url)
        data = response.json()
        
        if "error" in data:
            return f"Ошибка получения прогноза: {data['error']['message']}"
        
        location = data["location"]["name"]
        country = data["location"]["country"]
        current = data["current"]
        forecast_days = data["forecast"]["forecastday"]
        
        result = f"Погода в {location}, {country}:\n\n"
        result += f"Сейчас: {current['temp_c']}°C, {current['condition']['text']}\n\n"
        
        for day in forecast_days:
            date = day["date"]
            day_data = day["day"]
            result += f"Прогноз на {date}:\n"
            result += f"- Мин. температура: {day_data['mintemp_c']}°C\n"
            result += f"- Макс. температура: {day_data['maxtemp_c']}°C\n"
            result += f"- Условия: {day_data['condition']['text']}\n"
            result += f"- Вероятность осадков: {day_data['daily_chance_of_rain']}%\n\n"
        
        return result
    except Exception as e:
        return f"Ошибка при получении прогноза погоды: {str(e)}"
```

### Новостные API

```python
import requests
from datetime import datetime, timedelta
from smolagents import tool

# Получите свой ключ API на newsapi.org
NEWS_API_KEY = "ваш_api_ключ"

@tool
def get_latest_news(query: str = None, language: str = "ru", count: int = 5) -> str:
    """
    Получает последние новости по запросу.
    
    Args:
        query: Поисковый запрос (опционально)
        language: Язык новостей (ru, en, de, ...)
        count: Количество новостей для отображения
    """
    try:
        # Формируем дату за последнюю неделю
        week_ago = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        
        url = f"https://newsapi.org/v2/everything?apiKey={NEWS_API_KEY}&language={language}&from={week_ago}&sortBy=publishedAt"
        
        if query:
            url += f"&q={query}"
        
        response = requests.get(url)
        data = response.json()
        
        if data["status"] != "ok":
            return f"Ошибка получения новостей: {data.get('message', 'Неизвестная ошибка')}"
        
        articles = data["articles"][:count]
        
        if not articles:
            return "По вашему запросу новостей не найдено."
        
        result = "Последние новости:\n\n"
        
        for i, article in enumerate(articles, 1):
            title = article["title"]
            source = article["source"]["name"]
            date = datetime.fromisoformat(article["publishedAt"].replace("Z", "+00:00"))
            formatted_date = date.strftime("%d.%m.%Y %H:%M")
            url = article["url"]
            
            result += f"{i}. {title}\n"
            result += f"   Источник: {source}, {formatted_date}\n"
            result += f"   Ссылка: {url}\n\n"
        
        return result
    except Exception as e:
        return f"Ошибка при получении новостей: {str(e)}"
```

### API поиска

```python
import requests
from smolagents import tool

# Получите свой ключ API на serpapi.com
SERP_API_KEY = "ваш_api_ключ"

@tool
def search_web(query: str, count: int = 5) -> str:
    """
    Выполняет поиск в интернете по указанному запросу.
    
    Args:
        query: Текст поискового запроса
        count: Количество результатов для отображения
    """
    try:
        url = f"https://serpapi.com/search?engine=google&q={query}&api_key={SERP_API_KEY}&num={count}"
        response = requests.get(url)
        data = response.json()
        
        if "error" in data:
            return f"Ошибка поиска: {data['error']}"
        
        organic_results = data.get("organic_results", [])
        
        if not organic_results:
            return "По вашему запросу ничего не найдено."
        
        result = f"Результаты поиска по запросу '{query}':\n\n"
        
        for i, item in enumerate(organic_results[:count], 1):
            title = item.get("title", "Без заголовка")
            snippet = item.get("snippet", "Описание отсутствует")
            link = item.get("link", "#")
            
            result += f"{i}. {title}\n"
            result += f"   {snippet}\n"
            result += f"   Ссылка: {link}\n\n"
        
        return result
    except Exception as e:
        return f"Ошибка при выполнении поиска: {str(e)}"
```

## Работа с аутентификацией

При интеграции с внешними API часто требуется аутентификация. Рассмотрим различные способы аутентификации и безопасной работы с API-ключами:

### Использование переменных окружения

```python
import os
import requests
from smolagents import tool
from dotenv import load_dotenv

# Загружаем переменные окружения из файла .env
load_dotenv()

# Получаем API-ключи из переменных окружения
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

@tool
def get_weather_safe(city: str) -> str:
    """
    Безопасно получает прогноз погоды с использованием ключа API из переменных окружения.
    
    Args:
        city: Название города
    """
    if not WEATHER_API_KEY:
        return "Ошибка: API-ключ для сервиса погоды не настроен."
    
    try:
        url = f"http://api.weatherapi.com/v1/current.json?key={WEATHER_API_KEY}&q={city}&lang=ru"
        response = requests.get(url)
        data = response.json()
        
        if "error" in data:
            return f"Ошибка получения прогноза: {data['error']['message']}"
        
        location = data["location"]["name"]
        country = data["location"]["country"]
        temp = data["current"]["temp_c"]
        condition = data["current"]["condition"]["text"]
        
        return f"Текущая погода в {location}, {country}: {temp}°C, {condition}"
    except Exception as e:
        return f"Ошибка при получении прогноза погоды: {str(e)}"
```

### Использование OAuth

```python
import requests
import base64
import json
from urllib.parse import urlencode
from smolagents import tool

# Конфигурация для OAuth
CLIENT_ID = "ваш_client_id"
CLIENT_SECRET = "ваш_client_secret"
REDIRECT_URI = "http://localhost:8000/callback"

# Базовый URL для Spotify API
SPOTIFY_API_BASE_URL = "https://api.spotify.com/v1"

class SpotifyAuthManager:
    def __init__(self):
        self.access_token = None
        self.token_type = None
        self.expires_in = None
    
    def get_auth_url(self):
        """Получает URL для авторизации пользователя."""
        params = {
            "client_id": CLIENT_ID,
            "response_type": "code",
            "redirect_uri": REDIRECT_URI,
            "scope": "user-read-private user-read-email playlist-read-private"
        }
        return f"https://accounts.spotify.com/authorize?{urlencode(params)}"
    
    def exchange_code_for_token(self, code):
        """Обменивает код авторизации на токен доступа."""
        auth_header = base64.b64encode(f"{CLIENT_ID}:{CLIENT_SECRET}".encode()).decode()
        
        headers = {
            "Authorization": f"Basic {auth_header}",
            "Content-Type": "application/x-www-form-urlencoded"
        }
        
        data = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": REDIRECT_URI
        }
        
        response = requests.post("https://accounts.spotify.com/api/token", headers=headers, data=data)
        token_info = response.json()
        
        self.access_token = token_info.get("access_token")
        self.token_type = token_info.get("token_type")
        self.expires_in = token_info.get("expires_in")
        
        return token_info
    
    def get_headers(self):
        """Возвращает заголовки для API-запросов."""
        if not self.access_token:
            return None
        
        return {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }

# Создаем менеджер авторизации Spotify
spotify_auth = SpotifyAuthManager()

@tool
def get_auth_url() -> str:
    """
    Возвращает ссылку для авторизации в Spotify.
    """
    return spotify_auth.get_auth_url()

@tool
def authorize_spotify(code: str) -> str:
    """
    Авторизуется в Spotify с использованием кода авторизации.
    
    Args:
        code: Код авторизации, полученный после перенаправления
    """
    try:
        token_info = spotify_auth.exchange_code_for_token(code)
        if "access_token" in token_info:
            return "Авторизация в Spotify успешно выполнена!"
        else:
            return f"Ошибка авторизации: {token_info.get('error_description', 'Неизвестная ошибка')}"
    except Exception as e:
        return f"Ошибка при авторизации в Spotify: {str(e)}"

@tool
def search_spotify(query: str, type: str = "track", limit: int = 5) -> str:
    """
    Выполняет поиск в Spotify.
    
    Args:
        query: Поисковый запрос
        type: Тип поиска (track, album, artist, playlist)
        limit: Количество результатов
    """
    headers = spotify_auth.get_headers()
    if not headers:
        return "Ошибка: необходимо авторизоваться в Spotify. Используйте инструмент get_auth_url."
    
    try:
        params = {
            "q": query,
            "type": type,
            "limit": limit
        }
        
        response = requests.get(f"{SPOTIFY_API_BASE_URL}/search", headers=headers, params=params)
        data = response.json()
        
        results = []
        
        if type == "track" and "tracks" in data:
            items = data["tracks"]["items"]
            for item in items:
                name = item["name"]
                artists = ", ".join([artist["name"] for artist in item["artists"]])
                album = item["album"]["name"]
                url = item["external_urls"]["spotify"]
                results.append(f"Трек: {name}\nИсполнитель(и): {artists}\nАльбом: {album}\nСсылка: {url}")
        
        elif type == "album" and "albums" in data:
            items = data["albums"]["items"]
            for item in items:
                name = item["name"]
                artists = ", ".join([artist["name"] for artist in item["artists"]])
                url = item["external_urls"]["spotify"]
                results.append(f"Альбом: {name}\nИсполнитель(и): {artists}\nСсылка: {url}")
        
        elif type == "artist" and "artists" in data:
            items = data["artists"]["items"]
            for item in items:
                name = item["name"]
                url = item["external_urls"]["spotify"]
                results.append(f"Исполнитель: {name}\nСсылка: {url}")
        
        elif type == "playlist" and "playlists" in data:
            items = data["playlists"]["items"]
            for item in items:
                name = item["name"]
                owner = item["owner"]["display_name"]
                url = item["external_urls"]["spotify"]
                results.append(f"Плейлист: {name}\nВладелец: {owner}\nСсылка: {url}")
        
        if not results:
            return f"По запросу '{query}' ничего не найдено."
        
        return "\n\n".join(results)
    except Exception as e:
        return f"Ошибка при поиске в Spotify: {str(e)}"
```

## Обработка структурированных данных

Инструменты могут работать со структурированными данными разных форматов:

### Обработка JSON

```python
import json
import requests
from smolagents import tool

@tool
def process_json_api(url: str) -> str:
    """
    Получает JSON-данные из указанного API и форматирует результат.
    
    Args:
        url: URL для API-запроса
    """
    try:
        response = requests.get(url)
        data = response.json()
        
        # Форматируем JSON для удобного чтения
        formatted_json = json.dumps(data, indent=2, ensure_ascii=False)
        
        return f"Полученные данные:\n{formatted_json}"
    except Exception as e:
        return f"Ошибка при обработке JSON: {str(e)}"

@tool
def filter_json(json_string: str, path: str) -> str:
    """
    Выбирает данные из JSON по указанному пути.
    
    Args:
        json_string: Строка с JSON-данными
        path: Путь к нужным данным (например: 'results.0.name')
    """
    try:
        data = json.loads(json_string)
        
        # Разбиваем путь на компоненты
        path_components = path.split('.')
        
        current = data
        for component in path_components:
            if component.isdigit():  # Если компонент - индекс массива
                component = int(component)
            
            try:
                current = current[component]
            except (KeyError, IndexError, TypeError):
                return f"Ошибка: путь '{path}' не найден в JSON"
        
        # Если результат - словарь или список, форматируем его
        if isinstance(current, (dict, list)):
            result = json.dumps(current, indent=2, ensure_ascii=False)
        else:
            result = str(current)
        
        return f"Результат:\n{result}"
    except json.JSONDecodeError:
        return "Ошибка: некорректный формат JSON"
    except Exception as e:
        return f"Ошибка при фильтрации JSON: {str(e)}"
```

### Работа с XML

```python
import requests
import xmltodict
from smolagents import tool

@tool
def process_xml_api(url: str) -> str:
    """
    Получает XML-данные из указанного API и преобразует их в удобный формат.
    
    Args:
        url: URL для API-запроса
    """
    try:
        response = requests.get(url)
        xml_data = response.text
        
        # Преобразуем XML в словарь
        dict_data = xmltodict.parse(xml_data)
        
        # Преобразуем результат в строку
        result = json.dumps(dict_data, indent=2, ensure_ascii=False)
        
        return f"Преобразованные XML данные:\n{result}"
    except Exception as e:
        return f"Ошибка при обработке XML: {str(e)}"
```

### Работа с CSV

```python
import csv
import io
import requests
from smolagents import tool

@tool
def process_csv(url: str, delimiter: str = ",") -> str:
    """
    Получает и обрабатывает CSV-данные из указанного URL.
    
    Args:
        url: URL для получения CSV-файла
        delimiter: Символ-разделитель в CSV (обычно запятая)
    """
    try:
        response = requests.get(url)
        content = response.text
        
        # Парсим CSV
        reader = csv.reader(io.StringIO(content), delimiter=delimiter)
        rows = list(reader)
        
        if not rows:
            return "CSV-файл пуст или некорректен."
        
        # Извлекаем заголовки
        headers = rows[0]
        
        # Форматируем результат
        result = "Данные CSV-файла:\n\n"
        
        # Выводим первые 5 строк
        max_rows = min(len(rows), 6)  # Заголовок + 5 строк данных
        
        # Определяем максимальную ширину каждого столбца
        col_widths = [len(h) for h in headers]
        for i in range(1, max_rows):
            row = rows[i]
            for j, cell in enumerate(row):
                if j < len(col_widths):
                    col_widths[j] = max(col_widths[j], len(cell))
        
        # Выводим заголовки
        header_row = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
        result += header_row + "\n"
        result += "-" * len(header_row) + "\n"
        
        # Выводим данные
        for i in range(1, max_rows):
            row = rows[i]
            result += " | ".join(cell.ljust(col_widths[j]) if j < len(col_widths) else cell 
                                for j, cell in enumerate(row)) + "\n"
        
        if len(rows) > max_rows:
            result += f"\n... и еще {len(rows) - max_rows} строк."
        
        result += f"\nВсего строк данных: {len(rows) - 1}"
        
        return result
    except Exception as e:
        return f"Ошибка при обработке CSV: {str(e)}"
```

## Создание инструментов для работы с файлами

### Чтение и запись файлов

```python
import os
from smolagents import tool

@tool
def read_file(file_path: str) -> str:
    """
    Читает содержимое файла и возвращает его.
    
    Args:
        file_path: Путь к файлу для чтения
    """
    try:
        if not os.path.exists(file_path):
            return f"Ошибка: файл {file_path} не существует."
        
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        return f"Содержимое файла {file_path}:\n\n{content}"
    except Exception as e:
        return f"Ошибка при чтении файла: {str(e)}"

@tool
def write_to_file(file_path: str, content: str, mode: str = "w") -> str:
    """
    Записывает содержимое в файл.
    
    Args:
        file_path: Путь к файлу для записи
        content: Текст для записи в файл
        mode: Режим открытия файла ('w' - перезапись, 'a' - дозапись)
    """
    try:
        with open(file_path, mode, encoding='utf-8') as file:
            file.write(content)
        
        return f"Содержимое успешно записано в файл {file_path}."
    except Exception as e:
        return f"Ошибка при записи в файл: {str(e)}"
```

### Работа с изображениями

```python
import requests
from PIL import Image
from io import BytesIO
from smolagents import tool

@tool
def download_image(url: str, save_path: str) -> str:
    """
    Скачивает изображение по URL и сохраняет его локально.
    
    Args:
        url: URL изображения
        save_path: Путь для сохранения изображения
    """
    try:
        response = requests.get(url)
        image = Image.open(BytesIO(response.content))
        image.save(save_path)
        
        return f"Изображение успешно скачано и сохранено как {save_path}"
    except Exception as e:
        return f"Ошибка при скачивании изображения: {str(e)}"

@tool
def resize_image(image_path: str, width: int, height: int, save_path: str = None) -> str:
    """
    Изменяет размер изображения.
    
    Args:
        image_path: Путь к исходному изображению
        width: Новая ширина
        height: Новая высота
        save_path: Путь для сохранения измененного изображения (если None, перезаписывает исходное)
    """
    try:
        if not os.path.exists(image_path):
            return f"Ошибка: файл {image_path} не существует."
        
        image = Image.open(image_path)
        resized_image = image.resize((width, height))
        
        if save_path is None:
            save_path = image_path
        
        resized_image.save(save_path)
        
        return f"Размер изображения успешно изменен и сохранен как {save_path}"
    except Exception as e:
        return f"Ошибка при изменении размера изображения: {str(e)}"
```

## Заключение

В этом уроке мы рассмотрели, как создавать продвинутые инструменты и интегрировать внешние API в агентов SmolaGents. Мы изучили:

1. Создание сложных составных инструментов
2. Инструменты с состоянием
3. Интеграцию с различными типами API (погода, новости, поиск)
4. Работу с аутентификацией (API-ключи, OAuth)
5. Обработку структурированных данных (JSON, XML, CSV)
6. Создание инструментов для работы с файлами и изображениями

Эти продвинутые возможности позволяют создавать многофункциональных агентов, способных решать широкий спектр задач и эффективно взаимодействовать с внешними сервисами и данными.

## Практические задания

1. Создайте инструмент, который объединяет возможности погодного API и генерации изображений, чтобы получать визуальное представление прогноза погоды.
2. Разработайте инструмент для анализа настроения текста с использованием внешнего API.
3. Создайте инструмент, который может скачивать, анализировать и визуализировать данные из открытых источников.
4. Реализуйте инструмент для работы с API социальных сетей, который позволяет получать и анализировать данные.
5. Разработайте систему с сохранением состояния, которая позволяет агенту "запоминать" предыдущие взаимодействия с пользователем.

В следующих уроках мы рассмотрим более сложные сценарии использования агентов и продвинутые техники их разработки и оптимизации. 