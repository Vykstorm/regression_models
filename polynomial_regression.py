
from linear_regression import LinearRegression, LinearRegressionPredictor, LinearRegressionCmd
from sklearn.preprocessing import PolynomialFeatures
import json
import numpy as np
from sys import argv

class PolynomialRegressionPredictor(LinearRegressionPredictor):
    def __init__(self, weights, bias, degrees, interaction_only):
        super().__init__(weights, bias)
        self.degree = degrees
        self.interaction_only = interaction_only

    def predict(self, X):
        return super().predict( PolynomialFeatures(self.degree, self.interaction_only).fit_transform(X) )


    def to_string(self):
        return json.dumps({
            'weights' : self.weights.tolist(),
            'bias' : self.bias,
            'degree' : self.degree,
            'interaction_only' : self.interaction_only
        })

    @classmethod
    def from_string(cls, s):
        data = json.loads(s)
        weights = np.asarray(data['weights'])
        bias = data['bias']
        degree = data['degree']
        interaction_only = data['interaction_only']
        return cls(weights, bias, degree, interaction_only)


    def __str__(self):
        '{}, max.degree = {}, interaction_only = {}'.format(super().__str__(), self.degree, self.interaction_only)


class PolynomialRegression(LinearRegression):
    '''
    Esta clase puede utilizarse para crear un modelo de regresión lineal para clasificar ejemplos que
    no siguen un modelo lineal.
    A partir de las características de los ejemplos: x1, x2, ...xn, se crean nuevas, de la forma xi ^ k
    y se creará un modelo de regresión lineal con todas estas.
    '''

    def __init__(self, model_type, degrees = 2, interaction_only = False, **kwargs):
        '''
        Inicializa una instancia de esta clase
        :param degree: Es el máximo exponente que pueda tener una característica de los ejemplos del modelo.
        Si por ejemplo, este parámetro se establece a 3. Se utilizarán las caracteristicas x1, x1^2, x1^3,
        x2, x2^2, x2^3, ....
        Debe ser un valor entero positivo mayor que cero. Por defecto es 2

        :param model_type: Es el tipo de regresión lineal a utilizar: 'normal', 'ridge', 'lasso', 'elasticnet'
        :param kwargs: Parámetros adicionales para el constructor de la superclase (LinearRegression)
        '''

        super().__init__(model_type, **kwargs)
        self.degree = degrees
        self.interaction_only = interaction_only

    def train(self, X, Y):
        super().train( PolynomialFeatures(self.degree, self.interaction_only).fit_transform(X), Y )


    @property
    def predictor(self):
        return PolynomialRegressionPredictor(self.weights, self.bias, self.degree, self.interaction_only)



class PolynomialRegressionCmd(LinearRegressionCmd):
    def __init__(self):
        super().__init__(description = 'Builds a polynomial regression model using the specified training data set'
                                       'and test it with some examples')
        self.add_argument('-d', '--degrees', help = 'Indicates maximum exponent of a generated characteristic that will be used for this model. By default is 2', type = int, metavar = 'DEGREES')
        self.add_argument('-i', '--interaction-only', help = 'Indicates that only products of at most degrees distinct features will be used as input for the model', action = 'store_true')

    def _process_args(self, args):
        model_type, model_params, train_data, test_data, output, plot_model, print_summary = super()._process_args(args)

        model_params['degrees'] = args.degrees if not args.degrees is None else 2
        model_params['interaction_only'] = args.interaction_only

        return model_type, model_params, train_data, test_data, output, plot_model, print_summary

    def _generate_model(self, model_type, model_params):
        return PolynomialRegression(model_type, **model_params)



if __name__ == '__main__':
    cmd = PolynomialRegressionCmd()
    cmd.exec(argv[1:])