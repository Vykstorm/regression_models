
'''
Script que crea un programa con una interfaz por consola de comandos para generar datasets de prueba.
Se especifica una función y = f(x1, x2, ..., xn) y un rango a0, a1 tal que a0 <= xi <= a1 con 1 <= i <= n

El resultado es un dataset, una base de datos que contiene registros de la forma:
x01, x02, ...., x0n, y
x11, x12, ...., x1n, y
...
xm1, xm2, ...., xmn, y

El dataset puede exportarse en formato .csv
'''

from sys import argv
from argparse import ArgumentParser
from re import fullmatch, match
import numpy as np
from itertools import combinations
from sys import stdout


if __name__ == '__main__':
    parser = ArgumentParser(description = 'This program creates a dataset for testing purposes on regression models'
                                          ' using an arbitrary function y = f(x1, x2, ..., xn)')
    parser.add_argument('function', help = 'Must be a function that accepts an array X whose size is N and returns an arbitrary value. '
                                           'This argument will be evaluated using the built-in function eval(), which means that it will be interpreted as '
                                           'python code. All the functions on numpy library are avaliable', metavar = 'FUNCTION')
    parser.add_argument('-d', '--domain', help = 'Specifies the interval [a0, a1] to set the domain of the variables x1, x2, ..., xn.'
                                                 'a0 <= xi <= a1 will be true for each xi, 0 <= i <= N. By default a0 = 0 and a1 = 100. It must be a tuple of two or three values '
                                                 'separated by comma: "a0, a1, [step]". Step will indicate the step value to generate a sequence of all the '
                                                 'possible values for each variable x0, x1, ..., xN. By default is 1', metavar = 'RANGE')
    parser.add_argument('-n', help = 'Indicates the amount of variables (N) to be used. It must be an integer greater than 0. '
                                     'By default is 1', metavar = 'N', type = int)

    parser.add_argument('-o', '--output', help = 'Specifies a file to output the generated dataset. If any is specified, then stdout will be used', metavar = 'FILE')

    args = parser.parse_args(argv[1:])



    N = args.n
    if N is None:
        N = 1
    elif N <= 0:
        parser.error('Number of variables (N) must be greater than 0')


    domain = args.domain

    try:
        if not domain is None:
            result = match('^[ ]*([^, ]+)[ ]*,[ ]*([^, ]+)[ ]*([^ ].*)$', domain)
            a0, a1 = float(result.group(1)), float(result.group(2))

            result = fullmatch(',[ ]*([^, ]+)[ ]*', result.group(3))
            step = 1 if result is None else float(result.group(1))
        else:
            a0, a1, step = 0, 100, 1

        domain = np.arange(a0, a1, step)

    except Exception as e:
        parser.error('Invalid syntax specified for "domain" argument')

    if len(domain) == 0:
        parser.error('Domain interval is not valid')



    try:
        output = stdout if args.output is None else open(args.output, 'w')
    except:
        parser.error('Failed to open output file')

    I = combinations(range(0, len(domain)), N)
    F = lambda X: eval(args.function, { '__builtins__' : np.__dict__}, {'X' : X})

    try:
        first_iter = True
        while True:
            X = domain[np.asarray(next(I))]
            try:
                y = F(X)
                if not isinstance(y, (int, float)) and (not isinstance(y, np.ndarray) and len(y) == 1):
                    raise Exception('Return value must be a unique integer or float value')
                if isinstance(y, np.ndarray):
                    y = y[0]


                if not first_iter:
                    output.write('\n')
                first_iter = False
                output.write(','.join([str(value) for value in (list(X) + [y]) ]))


            except Exception as e:
                parser.error('Error calling the generator function specified: {}'.format(e))


    except StopIteration:
        pass

    try:
        pass
    finally:
        if output != stdout:
            output.close()
        else:
            print()
