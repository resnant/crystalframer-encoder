from setuptools import setup, find_packages
import os

# パッケージの基本情報を読み込み
def read_requirements():
    """requirements.txtから依存関係を読み込み"""
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

def read_readme():
    """Read README.md for package description"""
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ""

setup(
    name="crystalframer-encoder",
    version="0.1.0",
    author="Yuta Suzuki",
    author_email="resnant@outlook.jp",
    description="Crystal structure encoder based on pre-trained CrystalFramer models",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/resnant/crystalframer-encoder",  # TODO: Update with your GitHub username
    packages=find_packages(exclude=["tests", "tests.*", "examples", "docs"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": ["pytest>=6.0", "black>=22.0", "flake8>=4.0", "mypy>=0.950", "isort>=5.0"],
        "docs": ["sphinx>=4.0", "sphinx-rtd-theme>=1.0"],
    },
    include_package_data=True,
    package_data={
        "crystalframer_encoder": [
            "configs/*.json", 
            "configs/*.yaml",
            "py.typed",  # For type hints
        ],
    },
    entry_points={
        "console_scripts": [
            "crystalframer-encode=crystalframer_encoder.cli:main",
        ],
    },
    keywords="crystal structure, materials science, machine learning, transformer, graph neural network, materials informatics",
    project_urls={
        "Bug Reports": "https://github.com/resnant/crystalframer-encoder/issues",
        "Source": "https://github.com/resnant/crystalframer-encoder",
        "Documentation": "https://github.com/yourusername/crystalframer-encoder#readme",
        "Paper": "https://openreview.net/forum?id=gzxDjnvBDa",
    },
)
