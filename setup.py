from setuptools import setup

with open('requirements.txt') as requirements_file:
    requirements = requirements_file.read()


setup(
    name='gym-pomdps',
    version='1.0.0',
    description='Gym flat POMDP environments',
    author='Andrea Baisero',
    author_email='andrea.baisero@gmail.com',
    url='https://github.com/abaisero/gym-pomdps',
    packages=['gym_pomdps'],
    package_data={'': ['*.pomdp']},
    install_requires=requirements,
    license='MIT',
)
