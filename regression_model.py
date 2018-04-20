
from inspect import getmembers
import numpy as np
import json

class RegressionModelPredictor:
    '''
    Las instancias de esta clase poseen toda la información necesaria para generar un modelo de regresión
    sin supervisión (sin aprendizaje previo)
    '''
    def __init__(self):
        pass

    @staticmethod
    def get_params():
        '''
        Debe devolver los nombres de los parámetros que definen al modelo de regresión.
        Cada parámetro debe estar definido como una propiedad de clase o como un atributo asignado en el
        constructor.
        e.g: LinearRegressionPredictor.get_params() devuelve la tupla ('bias', 'weights')
        :return:
        '''
        raise NotImplementedError()


    def to_string(self):
        '''
        Convierte esta instancia a formato de cadena de caracteres. Puede ser recuperado usando el método
        de clase .from_string()
        :return:
        '''
        members = dict(getmembers(self))
        params = dict([(param_name, members[param_name]) for param_name in self.get_params()])
        for param_name, param_value in params.items():
            if isinstance(param_value, np.ndarray):
                params[param_name] = param_value.tolist()

        return json.dumps(params)

    @classmethod
    def from_string(cls, s):
        '''
        Recupera una instancia de esta clase a partir de una cadena de caracteres generada previamente usando
        el método to_string()
        :param s:
        :return:
        '''
        data = json.loads(s)

        params = dict([(param_name, data[param_name]) for param_name in cls.get_params()])
        for param_name, param_value in params.items():
            if isinstance(param_value, (tuple, list)):
                params[param_name] = np.asarray(param_value)

        return cls(**params)



class RegressionModel:
    '''
    Representa un modelo de regresión. Es la clase base para LinearRegression, PolynomialRegression entre otras.
    '''
    def __init__(self):
        self.is_trained = False


    def train(self, X, Y):
        '''
        Entrena este modelo
        :param X: Debe ser una matriz numpy de tamaño nxm. n > 0, m > 0. Donde n es el número de ejemplos
        del conjunto de entrenamiento y m es el número de características.
        :param Y: Debe ser una matriz columna numpy o un iterable de tamaño n. Cada elemento indicará el valor
        del modelo esperado para cada ejemplo del conjunto de entrenamiento.
        :return:
        '''
        self.is_trained = True

    def predict(self, X):
        '''
        Predice el valor de salida de este modelo asociado a un ejemplo o varios.
        :param X: Es una matriz numpy de tamaño pxm, p > 0, m > 0. Donde m es el número de caracteristicas de
        cada ejemplo.
        :return: Devuelve una lista de p-elementos. El elemento i-ésimo es la predicción del modelo sobre el
        ejemplo i-ésimo del conjunto X pasado como parámetro.
        '''
        self.predictor.predict(X)


    @property
    def predictor(self):
        '''
        Debe devolver una instancia de la clase RegressionModelPredictor o una de sus subclases.
        :return:
        '''
        return NotImplementedError()


    def __str__(self):
        return 'Regression model'

    def __repr__(self):
        return str(self)