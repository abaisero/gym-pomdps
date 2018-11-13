from setuptools import setup, find_packages

setup(
    name='gym_pomdps',
    version='0.1.0',
    packages=find_packages(),
    package_data={'': ['*.pomdp']},
    # install_requires=[
    #     'gym',
    #     'numpy',
    #     'rl_parsers',
    # ],
)
