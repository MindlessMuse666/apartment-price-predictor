# Прогноз Аренды Квартиры (Apartment Price Predictor) <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="MIT-License image"></a>

Проект для прогнозирования стоимости аренды квартиры на основе ее площади с использованием методов линейной регрессии. Этот проект является практической работой.
-  Тема практической работы: `Основы машинного обучения`
-  Дисциплина: `МДК 13.01: Основы применения методов искусственного интеллекта в программировании`


## Описание

Этот проект представляет собой реализацию модели машинного обучения для предсказания стоимости аренды квартиры в зависимости от ее площади. Модель основана на методе линейной регрессии и использует Python, а также библиотеки `numpy`, `pandas`, `scikit-learn` и `matplotlib`.


## Структура проекта

```
apartment-price-predictor/
  ├── data_handling/  # Директория обработки данных
  │  └── data_preparation.py  # Класс для подготовки данных
  ├── models/  # Директория моделей данных
  │  └── linear_regression_model.py  # Класс для линейной регрессии
  ├── visualization  # Директория визуализации данных
  │  └── visualization.py  # Класс визуализации
  ├── tests/  # Директория тестирования
  │  ├── test_apartment_predictor.py  # Файл юнит-тестирования
  │  └── test_data.py  # Файл с тестовыми данными
  ├── LICENSE
  ├── main.py  # Основной скрипт
  ├── README.md
  └── requirements.txt  # Файл с зависимостями
```

## Зависимости

-  `numpy`
-  `pandas`
-  `scikit-learn`
-  `matplotlib`

Все зависимости можно установить из файла `requirements.txt`.


## Установка и запуск

### Клонирование репозитория

```bash
git clone https://github.com/MindlessMuse666/apartment-price-predictor
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

Для деактивации виртуального окружения используйте команду `deactivate`:

```bash
deactivate
```


## Как использовать

1. Следуйте инструкциям по установке и запуску.
2. Настройте *Параметры обучения модели* в [`main.py`](main.py), если вам это необходимо:
  - `__MIN_TRAIN_COEF` - минимальный коэффициент обучения модели (минимальный шаг модели) (*оптимально >=10*)
  - `__MAX_TRAIN_COEF` - максимальный коэффициент обучения модели (максимальный шаг модели) (*оптимально >=10*) 
  - `__MIN_TRAIN_THRESHOLD` - минимальный порог обучения модели (*оптимально >=0*)
  - `__MAX_TRAIN_THRESHOLD` - максимальный порог обучения модели (*оптимально >=100*)
3. После запуска [`main.py`](main.py) будет построена модель линейной регрессии на основе случайного набора данных и выведен график предсказаний.
4. Можете поменять используемые наборы данных в [`test_data.py`](tests/test_data.py).


## Скриншоты проделанной работы

### Основной исполняемый скрипт - [main.py](main.py)
![image](https://github.com/user-attachments/assets/36c2e164-b7a7-4450-89e1-2a54be244edc)

### Класс для подготовки данных - [data_preparation.py](data_handling/data_preparation.py)
![image](https://github.com/user-attachments/assets/fbcf4d8f-dd00-4d52-9fa0-a469293f15aa)

### Класс для линейной регрессии - [linear_regression_model.py](models/linear_regression_model.py)
![image](https://github.com/user-attachments/assets/1b328bf0-cff7-412b-81a8-b7a015ba8e8a)

### Класс визуализации - [visualization.py](visualization/visualization.py) (Класс для подготовки данных)
![image](https://github.com/user-attachments/assets/dc044285-ce20-408d-81fb-7ffb699aba9d)

### Класс юнит-тестирования - [test_apartment_predictor.py](tests/test_apartment_predictor.py)
![image](https://github.com/user-attachments/assets/ab481f63-ca29-4acf-99a7-036b746de413)

### Класс с тестовыми данными - [test_data.py](tests/test_data.py)
![image](https://github.com/user-attachments/assets/ba587a51-f4fe-4e2e-b7f0-8e652879ad56)


## Лицензия

Этот проект распространяется под лицензией MIT - смотрите файл [LICENSE](LICENSE) для деталей.


## Автор

Бедин Владислав ([MindlessMuse666](https://github.com/MindlessMuse666))
  - GitHub: [MindlessMuse666](https://github.com/MindlessMuse666 "Владислав: https://github.com/MindlessMuse666")
  - Telegram: [@mindless_muse](t.me/mindless_muse)
  - Gmail: [mindlessmuse.666@gmail.com](mindlessmuse.666@gmail.com)
