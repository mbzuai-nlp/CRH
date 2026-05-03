from setuptools import setup

setup(
   name='steering_opt',
   version='0.1',
   description='A library for optimizing steering vectors in LLMs.',
   py_modules=['steering_opt'],
   install_requires=['mdmm', 'numpy', 'torch']
)
