from setuptools import setup, find_packages


setup(
    name='gym_pomdps',
    version='0.1.0',
    packages=find_packages(),
    package_data={'': ['*.pomdp']},
    install_requires=['numpy', 'gym'],
    test_suite='tests',
)
