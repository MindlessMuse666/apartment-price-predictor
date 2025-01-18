import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple


class Visualization:
    """
    Класс для визуализации результатов работы модели.
    Отвечает за создание графиков и их отображение.
    """

    def __init__(self, feature_name: str, target_name: str):
        self.feature_name = feature_name
        self.target_name = target_name


    def plot_results(self, X_train: np.ndarray, y_train: np.ndarray, X_predict: np.ndarray, y_predict: np.ndarray,
                    coef: float, intercept: float) -> None:
        """
        Визуализирует исходные данные и линию регрессии.

        Args:
            X_train (np.ndarray): Исходные данные по признакам.
            y_train (np.ndarray): Исходные данные по целевым переменным.
            X_predict (np.ndarray): Диапазон признаков для прогнозирования.
            y_predict (np.ndarray): Прогнозируемые значения целевых переменных.
            coef (float): Коэффициент модели линейной регрессии.
            intercept (float): Свободный член модели линейной регрессии.
        """
        plt.figure(figsize=(10, 6))
        plt.scatter(X_train, y_train, color="blue", label="Исходные данные")
        plt.plot(X_predict, y_predict, color="red", label="Линейная регрессия")
        plt.title(f"Прогноз {self.target_name} по {self.feature_name}")
        plt.xlabel(self.feature_name)
        plt.ylabel(self.target_name)
        plt.legend()
        plt.grid(True)
        plt.show()

        print(f"Коэффициент: {coef:.2f}, Свободный член: {intercept:.2f}")