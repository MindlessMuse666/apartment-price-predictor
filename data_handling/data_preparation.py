import pandas as pd
from typing import Tuple
import numpy as np


class DataPreparation:
    """
    Класс для подготовки данных для модели линейной регрессии.
    Отвечает за загрузку данных и их преобразование в нужный формат.
    """

    def __init__(self, data: dict):
        self.data = data
        self.df = pd.DataFrame(self.data)


    def get_features_and_target(self, feature_col: str, target_col: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Извлекает признаки и целевую переменную из DataFrame.

        Args:
            feature_col (str): название столбца с признаком.
            target_col (str): название столбца с целевой переменной.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Кортеж из признаков и целевой переменной в виде массивов numpy.
        """
        X = self.df[feature_col].values.reshape(-1, 1)  # Независимая переменная
        y = self.df[target_col].values  # Зависимая переменная
        return X, y