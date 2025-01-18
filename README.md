# Apartment Price Predictor <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="MIT-License image"></a>

Проект для прогнозирования стоимости аренды квартиры на основе ее площади с использованием методов линейной регрессии.


## Описание

Этот проект представляет собой реализацию модели машинного обучения для предсказания стоимости аренды квартиры в зависимости от ее площади. Модель основана на методе линейной регрессии и использует Python, а также библиотеки `numpy`, `pandas`, `scikit-learn` и `matplotlib`.


## Зависимости

-   Python 3.8+
-   `numpy`
-   `pandas`
-   `scikit-learn`
-   `matplotlib`

Все зависимости можно установить из файла `requirements.txt`.


## Установка и запуск

### Клонирование репозитория

```bash
git clone <ссылка_на_ваш_репозиторий>
cd <название_папки_репозитория>
```


### Создание и активация виртуального окружения

```bash
python -m venv .venv
# Для Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Для Linux и macOS:
source .venv/bin/activate
```


### Установка зависимостей

```bash
pip install -r requirements.txt
```


### Запуск юнит-тестов

```bash
python -m unittest tests/test_apartment_predictor.py
```


### Запуск скрипта main.py

```bash
python main.py
```


### Дополнительные команды

-  Для деактивации виртуального окружения используйте команду `deactivate`.

```bash
deactivate
```


## Структура проекта

```
apartment-price-predictor/
  ├── data_handling/  # Директория обработки данных
  │  └── data_preparation.py  # Класс для подготовки данных
  │
  ├── models/  # Директория моделей данных
  │  └── linear_regression_model.py  # Класс для линейной регрессии
  │
  ├── visualization  # Директория визуализации данных
  │  └── visualization.py  # Класс визуализации
  │
  ├── tests/  # Директория тестирования
  │  ├── test_apartment_predictor.py  # Файл юнит-тестирования
  │  └── test_data.py  # Файл с тестовыми данными
  │
  ├── requirements.txt  # Файл с зависимостями
  ├── LICENSE
  ├── main.py  # Основной скрипт
  └── README.md
```


## Как использовать

1. Следуйте инструкциям по установке и запуску.
2. После запуска `main.py` будет построена модель линейной регрессии на основе случайного набора данных и выведен график предсказаний.
3. Можете поменять используемые наборы данных в `test_data.py`


## Автор

MindlessMuse666 (https://github.com/MindlessMuse666)