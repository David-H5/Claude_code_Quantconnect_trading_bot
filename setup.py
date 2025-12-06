"""
Setup configuration for QuantConnect Trading Bot
"""

from setuptools import find_packages, setup


with open("README.md", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="quantconnect-trading-bot",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Custom trading algorithms for QuantConnect platform",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/Claude_code_Quantconnect_trading_bot",
    packages=find_packages(exclude=["tests", "research"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "flake8>=6.1.0",
            "mypy>=1.5.0",
        ],
        "research": [
            "jupyter>=1.0.0",
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
        ],
    },
)
