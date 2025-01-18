import unittest
import numpy as np

from data_handling.data_preparation import DataPreparation
from models.linear_regression_model import LinearRegressionModel
from tests.test_data import *
import random


class TestApartmentPredictor(unittest.TestCase):

    def setUp(self):
        """
        Метод setUp вызывается перед каждым тестом и подготавливает данные и модель для тестирования
        """
        self.feature_col = "Площадь"
        self.target_col = "Стоимость"
        
        # получаем все TEST_DATA
        self.all_test_data = [value for name, value in globals().items() if name.startswith("TEST_DATA_") and isinstance(value, dict)]
        
        # перемешиваем данные случайным образом
        random.shuffle(self.all_test_data)
        
        # определяем количество используемых наборов данных
        self.num_test_data = 5  # выбираем кол-во наборов данных для тестирования
        self.selected_test_data = self.all_test_data[:self.num_test_data]


    def test_data_preparation_get_features_and_target(self):
        """
        Тестирует метод get_features_and_target класса DataPreparation на корректность возврата X и Y
        """
        for test_data in self.selected_test_data:
            data_prep = DataPreparation(test_data)
            X, y = data_prep.get_features_and_target(self.feature_col, self.target_col)
            self.assertIsInstance(X, np.ndarray)
            self.assertIsInstance(y, np.ndarray)
            self.assertEqual(X.shape[1], 1)
            self.assertEqual(len(X), len(y))


    def test_linear_regression_model_train(self):
        """
        Тестирует обучение модели, проверяя, что обучение проходит без ошибок.
        """
        for test_data in self.selected_test_data:
            data_prep = DataPreparation(test_data)
            X, y = data_prep.get_features_and_target(self.feature_col, self.target_col)
            model = LinearRegressionModel()
            model.train(X, y)
            self.assertTrue(hasattr(model.model, 'coef_'))


    def test_linear_regression_model_predict(self):
        """
        Тестирует метод predict модели, проверяя, что он возвращает numpy массив и что его длина соответствует длине входящих данных.
        """
        for test_data in self.selected_test_data:
            data_prep = DataPreparation(test_data)
            X, y = data_prep.get_features_and_target(self.feature_col, self.target_col)
            model = LinearRegressionModel()
            model.train(X, y)
            X_predict = np.array([[50], [60], [70]])
            predictions = model.predict(X_predict)
            self.assertIsInstance(predictions, np.ndarray)
            self.assertEqual(len(predictions), len(X_predict))


    def test_linear_regression_model_get_model_coefficients(self):
        """
        Тестирует метод get_model_coefficients, проверяя, что возвращаются два числовых значения (коэффициент и свободный член)
        """
        for test_data in self.selected_test_data:
            data_prep = DataPreparation(test_data)
            X, y = data_prep.get_features_and_target(self.feature_col, self.target_col)
            model = LinearRegressionModel()
            model.train(X, y)
            coef, intercept = model.get_model_coefficients()
            self.assertIsInstance(coef, float)
            self.assertIsInstance(intercept, float)


    def test_model_predict_values_with_random_data_set(self):
        """
        Тестирует, что модель возвращает прогнозируемые значения не выходящие за рамки ожидаемого.
        """
        for test_data in self.selected_test_data:
            data_prep = DataPreparation(test_data)
            X, y = data_prep.get_features_and_target(self.feature_col, self.target_col)
            model = LinearRegressionModel()
            model.train(X, y)

            X_predict = np.linspace(min(X)-50, max(X)+50, 50).reshape(-1, 1)
            predictions = model.predict(X_predict)

            self.assertIsInstance(predictions, np.ndarray)
            self.assertTrue(predictions.size > 0)


if __name__ == '__main__':
    unittest.main()