import numpy as np
from sklearn.linear_model import LinearRegression
from typing import Tuple


class LinearRegressionModel:
    """
    Класс для модели линейной регрессии.
    Инкапсулирует логику обучения модели и прогнозирования.
    """
    
    def __init__(self):
        self.model = LinearRegression()


    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Обучает модель линейной регрессии на предоставленных данных.

        Args:
            X (np.ndarray): Массив признаков.
            y (np.ndarray): Массив целевых переменных.
        """
        self.model.fit(X, y)


    def predict(self, X_predict: np.ndarray) -> np.ndarray:
        """
        Прогнозирует значения для заданных признаков.

        Args:
            X (np.ndarray): Массив признаков для прогнозирования.

        Returns:
            np.ndarray: Массив с прогнозируемыми значениями.
        """
        return self.model.predict(X_predict)


    def get_model_coefficients(self) -> Tuple[float, float]:
        """
        Возвращает коэффициенты обученной модели.
        
        Returns:
            Tuple[float, float]: Коэффициент и свободный член.
        """
        return self.model.coef_[0], self.model.intercept_