# -*- encoding: utf-8 -*-
from pathlib import Path
import setuptools
from setuptools import find_packages

def get_readme():
    root_dir = Path(__file__).resolve().parent
    readme_path = str(root_dir / 'README.md')
    print(readme_path)
    with open(readme_path, 'r', encoding='utf-8') as f:
        readme = f.read()
    return readme



setuptools.setup(
    name='torch_paraformer',
    version='0.0.1',
    platforms="Any",
    url="https://github.com/alibaba-damo-academy/FunASR.git",
    author="Speech Lab, Alibaba Group, China",
    author_email="funasr@list.alibaba-inc.com",
    description="FunASR: A Fundamental End-to-End Speech Recognition Toolkit",
    license="The MIT License",
    long_description=get_readme(),
    long_description_content_type='text/markdown',
    include_package_data=True,
    install_requires=["librosa", "onnxruntime>=1.7.0",
                      "scipy", "numpy>=1.19.3",
                      "typeguard", "kaldi-native-fbank",
                      "PyYAML>=5.1.2"],
    packages=find_packages(include=["torch_paraformer*"]),
    keywords=[
        'funasr,paraformer'
    ],
    classifiers=[
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
)
