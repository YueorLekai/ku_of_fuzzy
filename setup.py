from setuptools import setup, find_packages

setup(
    name='ku_of_fuzzy',
    version='0.1.1',
    author='Yuekai',
    author_email='1977269004@qq.com',
    description='用于模糊数学计算的Python库',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/YueorLekai/ku_of_fuzzy',
    packages=find_packages(),
    install_requires=[
        'pandas>=2.2.1',
        'numpy>=1.26.4',
        'scipy>=1.13.0',
        'matplotlib>=3.8.3',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
