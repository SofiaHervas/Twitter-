from setuptools import setup, find_packages

setup(
    name='emotion_classifier',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
package_data={
    'inference': ['data/*.pt', 'data/*.pkl'],
},
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
