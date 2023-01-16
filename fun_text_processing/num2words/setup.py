# -*- coding: utf-8 -*-

import re
from io import open

from setuptools import find_packages, setup

PACKAGE_NAME = "num2words"

CLASSIFIERS = [
    'Development Status :: 5 - Production/Stable',
    'Intended Audience :: Developers',
    'License :: OSI Approved :: GNU Library or Lesser General Public License '
    '(LGPL)',
    'Programming Language :: Python :: 2.7',
    'Programming Language :: Python :: 3',
    'Topic :: Software Development :: Internationalization',
    'Topic :: Software Development :: Libraries :: Python Modules',
    'Topic :: Software Development :: Localization',
    'Topic :: Text Processing :: Linguistic',
]

LONG_DESC = open('README.rst', 'rt', encoding="utf-8").read() + '\n\n' + \
            open('CHANGES.rst', 'rt', encoding="utf-8").read()


def find_version(fname):
    """Parse file & return version number matching 0.0.1 regex
    Returns str or raises RuntimeError
    """
    version = ''
    with open(fname, 'r', encoding="utf-8") as fp:
        reg = re.compile(r'__version__ = [\'"]([^\'"]*)[\'"]')
        for line in fp:
            m = reg.match(line)
            if m:
                version = m.group(1)
                break
    if not version:
        raise RuntimeError('Cannot find version information')
    return version


setup(
    name=PACKAGE_NAME,
    version=find_version("bin/num2words"),
    description='Modules to convert numbers to multilingual words.',
    long_description=LONG_DESC,
    license='Alibaba Group',
    author='Zhang Chong, Alibaba DAMO Academy',
    author_email='',
    maintainer='Zhang Chong',
    maintainer_email='',
    keywords=' number word numbers words convert conversion '
             'localisation localization internationalisation '
             'internationalization',
    url='https://github.com/num2words',
    packages=find_packages(exclude=['tests']),
    test_suite='tests',
    classifiers=CLASSIFIERS,
    scripts=['bin/num2words'],
    install_requires=["docopt>=0.6.2"],
    tests_require=['delegator.py'],
)
