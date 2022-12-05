from setuptools import setup, find_packages

packages = find_packages('src')

for p in packages:
    assert p == 'chess_bot' or p.startswith('chess_bot.')

setup(
    name='chess_bot',
    version='1.0',
    packages=packages,
    package_dir={'': 'src'},
)