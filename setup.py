from setuptools import setup, find_packages
import sys

if sys.version_info.major != 3:
    print('This Python is only compatible with Python 3, but you are running '
          'Python {}. The installation will likely fail.'.format(sys.version_info.major))


setup(name='cvar-algorithms',
      packages=[package for package in find_packages('.')
                if package.startswith('cvar')],
      # install_requires=[
      #     'gym[mujoco,atari,classic_control,robotics]',
      #     'scipy',
      #     'tqdm',
      #     'joblib',
      #     'zmq',
      #     'dill',
      #     'progressbar2',
      #     'mpi4py',
      #     'cloudpickle',
      #     'tensorflow>=1.4.0',
      #     'click',
      # ],
      description='CVaR Algorithms',
      author='Silvestr Stanko',
      url='todo',
      author_email='silvicek@gmail.com',
      version='0.0.0')
