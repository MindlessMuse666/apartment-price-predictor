import numpy as np
from data_handling.data_preparation import DataPreparation
from models.linear_regression_model import LinearRegressionModel
from visualization.visualization import Visualization
from tests.test_data import *
import random


def main():
    """
     Основная функция для запуска процесса обучения и прогнозирования.
    """
    # получаем все TEST_DATA
    all_test_data = [value for name, value in globals().items() if name.startswith("TEST_DATA_") and isinstance(value, dict)]
    # выбираем случайный набор данных
    random_test_data = random.choice(all_test_data)

    # 1. Подготовка данных
    data_prep = DataPreparation(random_test_data)
    X_train, y_train = data_prep.get_features_and_target(feature_col = "Площадь", target_col = "Стоимость")

    # 2. Обучение модели
    model = LinearRegressionModel()
    model.train(X_train, y_train)

    # 3. Прогноз
    area_range = np.linspace(min(X_train)-5, max(X_train)+5, 100).reshape(-1, 1)
    predicted_prices = model.predict(area_range)
    coef, intercept = model.get_model_coefficients()

    # 4. Визуализация
    visualizer = Visualization(feature_name="Площадь", target_name="Стоимость")
    visualizer.plot_results(X_train, y_train, area_range, predicted_prices, coef, intercept)


if __name__ == "__main__":
    main()