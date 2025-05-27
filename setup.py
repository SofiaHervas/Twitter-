from setuptools import setup, find_packages

setup(
    name='emotion_classifier',
    version='0.1',
    packages=find_packages(),  # <- this line finds the `inference` folder
    include_package_data=True,
    install_requires=[
        'torch',
        'pandas',
        'numpy',
        'scikit-learn'
    ],
    entry_points={
        'console_scripts': [
            'inference=inference.cli:main'
        ]
    },
)

