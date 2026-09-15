from setuptools import setup, find_packages

setup(
    name="MMLD",
    version="0.1.0",
    description="Microbial community causal analysis toolkit",
    packages=find_packages(),
    package_dir={"": "."},
    python_requires=">=3.9",
    install_requires=[
        "numpy",
        "pandas",
        "openpyxl",
        "scipy",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "networkx",
        "torch",
        "statsmodels",
    ],
    extras_require={
        "pcmci": ["tigramite"],
        "lingam": ["lingam"],
        "neural": ["hybridmetrics"],
        "pc": ["causal-learn"],
    },
    entry_points={
        "console_scripts": [
            "MMLD = src.cli:main",
        ],
    },
)
