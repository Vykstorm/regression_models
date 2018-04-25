
from setuptools import setup


setup(
    name = 'regression_models',
    version = '1.0.0',
    description = 'Python library wrapper utility around sklearn package to create simple regression models',
    url = 'https://github.com/Vykstorm/regression_models',
    author = 'Vykstorm',
    author_email = 'victorruizgomezdev@gmail.com',
    license = 'MIT',
    zip_safe = False,
    packages = [''],
    install_requires = ['numpy', 'scipy', 'sklearn', 'pyvalid', 'matplotlib'],
    include_package_data = True
)