from setuptools import setup, find_packages
import sys

if sys.version_info.major != 3:
    print('This Python is only compatible with Python 3, but you are running '
          'Python {}. The installation will likely fail.'.format(sys.version_info.major))

print([package for package in find_packages()
                if package.startswith('cvar')])
setup(name='cvar-algorithms',
      packages=['cvar'],
      install_requires=[
          'matplotlib',
          'opencv-python',
          'pygame'
      ],
      description='Risk-Averse DistributionalReinforcement Learning',
      author='Silvestr Stanko',
      url='todo',
      author_email='silvicek@gmail.com',
      version='1.0.0')
