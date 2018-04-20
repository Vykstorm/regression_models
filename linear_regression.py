
'''
Script con clases para crear modelos de regresión lineal.
Se usa internamente la librería sklearn
'''

from sklearn import linear_model
from sklearn.exceptions import NotFittedError
from pyvalid.validators import accepts
import json
import numpy as np

from sys import argv, stdout
from argparse import ArgumentParser
from csv_utils import csv_to_array
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from regression_model import RegressionModelPredictor, RegressionModel


class LinearRegressionPredictor(RegressionModelPredictor):
    def __init__(self, weights, bias):
        super().__init__()
        self.model = linear_model.LinearRegression()
        self.weights = weights
        self.bias = bias

    def predict(self, X):
        return self.model.predict(X)

    @property
    def weights(self):
        return self.model.coef_

    @weights.setter
    def weights(self, values):
        self.model.coef_ = values

    @property
    def bias(self):
        return self.model.intercept_

    @bias.setter
    def bias(self, value):
        self.model.intercept_ = value

    @staticmethod
    def get_params():
        return 'weights', 'bias'


    def __str__(self):
        return 'weights=[{}], bias={}'.format(','.join([str(value) for value in self.weights]), self.bias)

    def __repr__(self):
        return str(self)



class LinearRegression(RegressionModel):
    @accepts(object, ('normal', 'ridge', 'lasso', 'elasticnet'))
    def __init__(self, model_type, alpha = None, l1_ratio = None, **kwargs):
        super().__init__()

        '''
        Instancia un modelo de regresión lineal. En función del parámetro model_type, se usará una técnica distinta
        para ajustar el modelo con respecto al conjunto de entrenamiento.
        :param model_type: Puede ser normal, ridge, lasso, multitask-lasso, elasticnet, multitask-elasticnet

        :param alpha: Parámetro de regularización. Se usa para los modelos 'ridge', 'lasso', 'elasticnet'
        Puede ser un valor númerico o una lista (en ese caso, se llevara a cabo una validación cruzada para escoger el
        mejor valor para este parámetro) Por defecto será .05

        :param l1_ratio: Parámetro que controla la relación entre la penalización de las normas l1 y l2 en el modelo
        de regresión 'elasticnet'

        :param kwargs: Argumentos adicionales que se pasarán al constructor de una de las clases del módulo
        sklearn.linear_model: LinearRegression, Ridge, Lasso, MultiTaskLasso, ElasticNet, MultiTaskElasticNet
        en función de lo que se haya indicado en el parámetro "model_type"
        Pará más información, mirar la página http://scikit-learn.org/stable/modules/linear_model.html
        '''

        # Procesamos los parámetros opcionales
        if alpha is None and model_type != 'normal':
            alpha = .05

        if l1_ratio is None and model_type == 'elasticnet':
            l1_ratio = .5

        # Usamos validación cruzada (CV) ?
        use_cross_validation = not alpha is None and not isinstance(alpha, (float, int))
        if use_cross_validation:
            alpha = np.array(alpha)

        # Seleccionamos la clase del módulo sklearn que necesitamos para construir el modelo
        cls = {
            ('normal', False) : linear_model.LinearRegression,
            ('normal', True) : linear_model.LinearRegression,
            ('ridge', False) : linear_model.Ridge,
            ('ridge', True) : linear_model.RidgeCV,
            ('lasso', False) : linear_model.Lasso,
            ('lasso', True) : linear_model.LassoCV,
            ('elasticnet', False) : linear_model.ElasticNet,
            ('elasticnet', True) : linear_model.ElasticNetCV
        }
        model_cls = cls[(model_type, use_cross_validation)]


        # Convertimos los parámetros de este constructor a parámetros para el constructor de la clase sklearn seleccionada
        model_params = {
            'alpha' if not use_cross_validation else 'alphas' : alpha,
            'l1_ratio' : l1_ratio
        }
        model_params = dict([(key, value) for key, value in model_params.items() if not value is None])
        model_params.update(kwargs)

        # Instanciamos la clase del módulo sklearn.
        self.model = model_cls(**model_params)

        # Valores por defecto de algunos atributos
        self.model_type = model_type


    def train(self, X, Y):
        '''
        Entrena este modelo de regresión lineal.
        :param X: Debe ser una matriz numpy de tamaño nxm. n > 0, m > 0. Donde n es el número de ejemplos
        del conjunto de entrenamiento y m es el número de características.
        :param Y: Debe ser una matriz columna numpy o un iterable de tamaño n. Cada elemento indicará el valor
        del modelo esperado para cada ejemplo del conjunto de entrenamiento.
        :return:
        '''
        self.model.fit(X, Y)
        super().train(X, Y)



    @property
    def weights(self):
        '''
        Propiedad que puede usarse para devolver/modificar los pesos del modelo (w1, w2, ..., wm) asociados a cada característica
        :return:
        '''
        if not hasattr(self.model, 'coef_') or not hasattr(self.model,  'intercept_'):
            raise NotFittedError()
        return self.model.coef_


    @weights.setter
    def weights(self, values):
        self.model.coef_ = values
        if hasattr(self.model,  'intercept_'):
            self.is_trained = True

    @property
    def bias(self):
        '''
        Propiedad que puede usarse para devolver/modificar el termíno bias del modelo (w0)
        :return:
        '''
        if not hasattr(self.model, 'coef_') or not hasattr(self.model,  'intercept_'):
            raise NotFittedError()
        return self.model.intercept_


    @bias.setter
    def bias(self, value):
        self.model.intercept_ = value
        if hasattr(self.model, 'coef_'):
            self.is_trained = True


    @property
    def predictor(self):
        return LinearRegressionPredictor(self.weights, self.bias)


    def __str__(self):
        return '{} regression model. {}'.format(self.model_type, 'not trained yet' if not self.is_trained else self.predictor)


    def __repr__(self):
        return str(self)



class LinearRegressionCmd(ArgumentParser):
    def __init__(self, description = None):
        if description is None:
            description = 'Creates a linear regression model using a training data set ' \
                          'and test it with some examples'
        super().__init__(description = description)

        self.add_argument('-tr', '--train', help = 'File with training data set examples in .csv format (comma separated values). The expected model value for each example is the las column of the data', metavar = 'FILE', required = True)
        self.add_argument('-tst', '--test', help = 'File with test examples in .csv format (comma separated). The expected model value for each example is the las column of the data', metavar = 'FILE', required = True)
        self.add_argument('-o', '--output', help = 'Output file to print regression model parameters (default to stdout)', metavar = 'FILE')
        self.add_argument('-s', '--print-summary', help = 'Print accuracy rate and stats of the model on stdout', action = 'store_true')
        self.add_argument('-p', '--plot', help = 'Plot training data set examples and the line modelled by the regression model (only used when examples has 2 dimensions)', action = 'store_true')
        self.add_argument('-t', '--type', help = 'Indicates the model regression type.'
                                                 ' Can be \'normal\', \'ridge\', \'lasso\', \'elasticnet\'',
                          metavar = 'TYPE')
        self.add_argument('--alpha', help='Regularization parameter, used for some regression model. '
                                          'Can be a list of values (in this case, a cross-validation will be performed to choose the best value)')
        self.add_argument('--l1-ratio', help='Parameter that controls the relationship between l1 and l2 norm penalties in the elasticnet regression model')


    def _process_args(self, args):
        train_file = args.train
        test_file = args.test
        output = args.output
        print_summary = args.print_summary
        plot_model = args.plot
        model_type = args.type

        model_params = {}

        if not model_type is None and not model_type in ('normal', 'ridge', 'lasso', 'elasticnet'):
            self.error('Invalid regression model type specified')

        if output is None:
            output = stdout
        model_type = 'normal' if model_type is None else model_type.lower()

        if not model_type == 'normal' and not args.alpha is None:
            alpha = eval(args.alpha, {'__builtins__': {'range': range, 'arange': np.arange}})
            model_params['alpha'] = alpha

        if model_type == 'elasticnet' and not args.l1_ratio is None:
            l1_ratio = eval(args.l1_ratio, {'__builtins__': {'range': range, 'arange': np.arange}})
            model_params['l1_ratio'] = l1_ratio

        try:
            with open(train_file, 'r') as file:
                train_data = csv_to_array(file.read())

            if train_data.shape[0] == 0 or train_data.shape[1] < 2:
                raise Exception()
        except:
            self.error('Error reading training dataset examples')

        try:
            with open(test_file, 'r') as file:
                test_data = csv_to_array(file.read())

            if test_data.shape[0] == 0 or test_data.shape[1] < 2:
                raise Exception()
        except:
            self.error('Error reading test dataset examples')

        if train_data.shape[1] != test_data.shape[1]:
            self.error('Incompatible train and test datasets')

        if train_data.shape[1] > 2 and plot_model:
            self.error('Examples cannot be plotted (they have more than 1 attribute)')

        return model_type, model_params, train_data, test_data, output, plot_model, print_summary


    def _print_summary(self, model_type, test_data, test_predictions):
        print('{} regression'.format(model_type))
        print('MSE (Mean-Squared-Error): {}'.format(mean_squared_error(test_data[:, -1], test_predictions)))
        print('R2 Score (Coefficient of determination): {}'.format(r2_score(test_data[:, -1], test_predictions)))


    def _plot_data(self, train_data, test_data, test_predictions):
        plt.scatter(train_data[:, :-1], train_data[:, -1], linewidth=1, color='blue', s=1, label='Training data',
                    marker='.')
        plt.plot(test_data[:, :-1], test_predictions, linewidth=2, color='red', label='linear regressor')
        plt.xlabel('Input data')
        plt.ylabel('Response')
        plt.show()

    def _save_model(self, model, output):
        file = output
        try:
            if output != stdout:
                file = open(output, 'w')

            file.write(model.predictor.to_string())
        finally:
            if output != stdout:
                file.close()
            else:
                file.flush()

    def _generate_model(self, model_type, model_params):
        return LinearRegression(model_type, **model_params)


    def exec(self, args):
        args = super().parse_args(args)

        model_type, model_params, train_data, test_data, output, plot_model, print_summary = self._process_args(args)

        # Creamos el modelo
        model = self._generate_model(model_type, model_params)

        # Entrenamos el modelo
        model.train(train_data[:, :-1], train_data[:,-1])

        # Predecimos los valores para los ejemplos del conjunto de entrenamiento
        Y = model.predict(test_data[:, :-1])

        # Mostramos información resumida del modelo y resultados
        if print_summary:
            self._print_summary(model_type, test_data, Y)

        # Guardamos los datos del modelo de regresión.
        self._save_model(model, output)

        print()

        # Mostramos una gráfica con los datos del conjunto de entrenamiento y la línea que representan los valores predichos por el modelo.
        if plot_model:
            self._plot_data(train_data, test_data, Y)






if __name__ == '__main__':
     cmd = LinearRegressionCmd()
     cmd.exec(args = argv[1:])
