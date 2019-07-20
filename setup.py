from setuptools import find_packages, setup

setup(
    name='gym_pomdps',
    version='0.1.0',
    packages=find_packages(),
    package_data={'': ['*.pomdp']},
    install_requires=['numpy', 'gym'],
    test_suite='tests',
)
