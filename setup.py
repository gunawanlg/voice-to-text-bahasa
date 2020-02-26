from setuptools import setup

setup(
    name="gurih",
    version="0.1.0",
    description="""End to end Speech Recognition Library""",
    url="https://github.com/Arc-rendezvous/voice-to-text-bahasa",
    author="Gunawan Lumban Gaol, M Haries Ramdhani",
    author_email="gunimarbun@gmail.com, hydrolizedmaltose@gmail.com",
    license="Apache 2.0",
    packages=[
        "gurih",
        "gurih.data",
        "gurih.features",
        "gurih.models"
    ],
    install_requires=[
        "pyyaml==5.3",
        "setuptools>=41.0.0",
        "six>=1.12.0",
        "librosa==0.7.2",
        "llvmlite==0.31.0",
        "mlxtend==0.17.0",
        "numba==0.47.0",
        "resampy==0.2.2",
        "scikit-learn==0.21.3",
        "pandas==0.24.2",
        "numpy==1.16.3",
        "tensorflow==2.1.0",
        "tqdm==4.42.0",
        "matplotlib==3.1.2",
        "tinytag==1.2.2",
        "pydub==0.23.1"
    ],
    include_package_data=False,
    # zip_safe=False,
)
