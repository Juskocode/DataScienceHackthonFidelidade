"""Package setup"""

from setuptools import setup, find_packages


setup(
    name="AUT.SMASHIT.XSELL_DENTAL_EXEMPLO",
    version="0.1.0",
    description="Xsell Dental Exemplo 42Porto",
    long_description_content_type="text/markdown",
    setup_requires=["setuptools>=38.6.0"],
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=["pandas == 2.0.3",
        "dask == 2023.5.0",
        "scikit-learn == 1.3.0",
        "termcolor == 2.3.0",
        "category-encoders == 2.6.1",
        "feature-engine == 1.6.1",
        "ydata-profiling == 4.5.1",
        "factor-analyzer == 0.5.0",
        "scipy == 1.10.1",
        "imbalanced-learn == 0.11.0",
        "lightgbm == 4.2.0",
        "prettytable == 3.10.0",
        "azureml-mlflow == 1.56.0",
        "mlflow == 2.14.1", # Alterado de 2.14.1 para 2.22.0 devido a problemas de compatibilidades de versões com pydantic
        "xlsxwriter ==3.2.0",
        "pytimedinput==2.0.1",
        "pydantic==1.10.14", # Alterado de 2.8.2 para 1.10.14 devido a problemas de compatibilidades de versões com várias bibliotecas
        "shap==0.44.1",
        "seaborn==0.12.2", # Alterado de 0.13.2 para 0.12.2 devido a problemas de compatibilidades de versões com ydata-profiling
        "openpyxl==3.1.5"
        ],
    package_data={},
)
