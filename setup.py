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
        "pandas>=0.23.2",
        "numpy>=1.16.3"
    ],
    include_package_data=True,
    # zip_safe=False,
)
