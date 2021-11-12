from setuptools import setup

setup(
    name='gym-pomdps',
    version='1.0.0',
    description='Gym flat POMDP environments',
    author='Andrea Baisero',
    author_email='andrea.baisero@gmail.com',
    url='https://github.com/abaisero/gym-pomdps',
    packages=['gym_pomdps'],
    package_data={'': ['*.pomdp']},
    install_requires=['gym', 'numpy', 'one-to-one', 'rl-parsers',],
    license='MIT',
)
