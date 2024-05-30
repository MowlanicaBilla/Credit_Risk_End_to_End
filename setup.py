'''
This Setup.py is a Python file that is used to build and distribute Python packages. 
It contains information about the package, such as its name, version, and dependencies, as well as instructions for building and installing the package.

'''

from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = '-e .'

def get_requirements(file_path:str)->List[str]:
    '''
    this function will return the list of requirements
    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        # Since the requirements.txt contains all the packages in different lines, removing \n
        requirements=[req.replace("\n","") for req in requirements]
        
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements

setup(
    name='Credit Risk Modelling',
    version='0.0.1',
    author="Mowlanica Billa",
    author_email='b.mowlanica@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)