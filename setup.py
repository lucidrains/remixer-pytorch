from setuptools import setup, find_packages

setup(
  name = 'remixer-pytorch',
  packages = find_packages(exclude=[]),
  version = '0.0.2',
  license='MIT',
  description = 'Remixer - Pytorch',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  url = 'https://github.com/lucidrains/remixer-pytorch',
  keywords = [
    'artificial intelligence',
    'transformer',
    'feedforward',
    'mlp-mixer'
  ],
  install_requires=[
    'einops>=0.3',
    'torch>=1.6'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
