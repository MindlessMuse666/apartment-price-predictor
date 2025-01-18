import numpy as np
from data_handling.data_preparation import DataPreparation
from models.linear_regression_model import LinearRegressionModel
from visualization.visualization import Visualization
from tests.test_data import *
import random


# Параметры обучения модели
__MIN_TRAIN_COEF: float = -50 
__MAX_TRAIN_COEF: float = 50
__MIN_TRAIN_THRESHOLD: float = 0
__MAX_TRAIN_THRESHOLD: float = 200

# Параметры отображения точек данных 
__POINTS_FREQUENCY: int = 10
__PREDICT_POINTS_COUNT: int = 50


def main():
    """
    Основная функция для запуска процесса обучения и прогнозирования.
    """
    # 1. Подготовка данных
    all_test_data: list = _get_all_test_data()
    random_test_data: dict = _select_random_test_data(all_test_data)
    X_train, y_train, min_train, max_train = _prepare_data(random_test_data)
    
    # 2. Тренировка модели
    model: LinearRegressionModel = _train_model(X_train, y_train)
    
    # 3. Прогнозирование
    area_range, predicted_prices = _predict_prices(model, X_train, min_train, max_train)
    coef, intercept = model.get_model_coefficients()
    
    # 4. Вывод отладочных данных
    _print_debug_info(min_train, max_train, random_test_data)
    
    # 5. Визуализация
    _visualize_results(X_train, y_train, area_range, predicted_prices, coef, intercept)


def _get_all_test_data() -> list:
    """
    Получает список всех доступных тестовых наборов данных.
    
    Returns:
         list: Список тестовых наборов данных.
    """
    return [value for name, value in globals().items() if name.startswith("TEST_DATA_") and isinstance(value, dict)]


def _select_random_test_data(all_test_data: list) -> dict:
    """
     Выбирает случайный набор тестовых данных.

    Args:
        all_test_data (list): Список всех тестовых данных.
    Returns:
         dict: Случайно выбранный тестовый набор данных.
     """
    return random.choice(all_test_data)


def _prepare_data(random_test_data: dict) -> tuple:
    """
    Подготавливает данные для обучения модели.
    
    Args:
        random_test_data (dict): Случайно выбранный набор тестовых данных.
    Returns:
        tuple: Кортеж, содержащий признаки, целевые переменные, минимальный и максимальный пороги для построения прогноза.
    """
    data_prep = DataPreparation(random_test_data)
    
    X_train, y_train = data_prep.get_features_and_target(feature_col="Площадь", target_col="Стоимость")
    
    min_train: float = __MIN_TRAIN_THRESHOLD if min(X_train) + __MIN_TRAIN_COEF < __MIN_TRAIN_THRESHOLD else min(X_train) - __MIN_TRAIN_COEF
    max_train: float = __MAX_TRAIN_THRESHOLD if max(X_train) + __MAX_TRAIN_COEF > __MAX_TRAIN_THRESHOLD else max(X_train) + __MAX_TRAIN_COEF
    
    return X_train, y_train, min_train, max_train


def _train_model(X_train: np.ndarray, y_train: np.ndarray) -> LinearRegressionModel:
    """
    Обучает модель линейной регрессии.
     
    Args:
        X_train (np.ndarray): Признаки обучающей выборки.
        y_train (np.ndarray): Целевые переменные обучающей выборки.
    Returns:
        LinearRegressionModel: Обученная модель линейной регрессии.
    """
    model = LinearRegressionModel()
    model.train(X_train, y_train)
    
    return model


def _predict_prices(model: LinearRegressionModel, X_train: np.ndarray, min_train: float, max_train: float) -> tuple:
    """
    Прогнозирует значения на основе обученной модели.
    
    Args:
        model (LinearRegressionModel): Обученная модель линейной регрессии.
        X_train (np.ndarray): Признаки обучающей выборки.
        min_train (float): Минимальный порог для построения прогноза.
        max_train (float): Максимальный порог для построения прогноза.

    Returns:
        tuple: Кортеж, содержащий диапазон прогнозирования и спрогнозированные значения.
    """
    min_predict = min_train
    max_predict = max_train
    area_range_left = np.linspace(min_predict, min(X_train), __PREDICT_POINTS_COUNT // 2).reshape(-1, 1) if min(X_train) != min_predict else np.array([])
    area_range_right = np.linspace(max(X_train), max_predict, __PREDICT_POINTS_COUNT // 2).reshape(-1, 1) if max(X_train) != max_predict else np.array([])

    if area_range_left.size > 0 and area_range_right.size > 0:
        area_range = np.concatenate((area_range_left, area_range_right), axis=0)
    elif area_range_left.size > 0:
        area_range = area_range_left
    elif area_range_right.size > 0:
        area_range = area_range_right
    else:
        area_range = np.array([])

    if area_range.size > 0:
        predicted_prices = model.predict(area_range)
    else:
         predicted_prices = np.array([])
    return area_range, predicted_prices


def _print_debug_info(min_train: float, max_train: float, random_test_data: dict) -> None:
    """
    Выводит отладочную информацию.
    Args:
        min_train (float): Минимальный порог для построения прогноза.
        max_train (float): Максимальный порог для построения прогноза.
        random_test_data (dict): Случайно выбранный набор тестовых данных.
    """
    print(f'Минимальный порог прогноз: {min_train}\nМаксимальный порог прогноза: {max_train}')
    print('\n', random_test_data, '\n')


def _visualize_results(X_train: np.ndarray, y_train: np.ndarray, area_range: np.ndarray, predicted_prices: np.ndarray,
                      coef: float, intercept: float) -> None:
    """
    Визуализирует результаты работы модели.
    
    Args:
        X_train (np.ndarray): Признаки обучающей выборки.
        y_train (np.ndarray): Целевые переменные обучающей выборки.
        area_range (np.ndarray): Диапазон для прогнозирования.
        predicted_prices (np.ndarray): Спрогнозированные значения.
        coef (float): Коэффициент модели линейной регрессии.
        intercept (float): Свободный член модели линейной регрессии.
    """
    visualizer = Visualization(feature_name="Площадь", target_name="Стоимость")
    visualizer.plot_results(X_train, y_train, area_range, predicted_prices, coef, intercept,
                            points_frequency=__POINTS_FREQUENCY)


if __name__ == "__main__":
    main()