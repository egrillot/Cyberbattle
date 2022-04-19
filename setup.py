from setuptools import setup, find_packages

setup(
    name='cyberbattle',
    version='0.1.0',
    description='Cyber environment in Python with gym like API',
    url='https://github.com/egrillot/CyberBattleAgents/cyberbattle',
    author='Emilien GRILLOT',
    author_email='emilien.grillot@polytechnique.edu',
    license='new BSD',
    packages=find_packages(),
    install_requires=[
        'boolean.py',
        'gym',
        'matplotlib',
        'networkx',
        'numpy',
        'tensorflow',
        'keras'
    ]
)