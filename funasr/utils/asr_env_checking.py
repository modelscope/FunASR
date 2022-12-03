# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import shutil
import ssl

import nltk

# mkdir nltk_data dir if not exist
try:
    nltk.data.find('.')
except LookupError:
    dir_list = nltk.data.path
    for dir_item in dir_list:
        if not os.path.exists(dir_item):
            os.mkdir(dir_item)
        if os.path.exists(dir_item):
            break

# download one package if nltk_data not exist
try:
    nltk.data.find('.')
except:  # noqa: *
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    nltk.download('cmudict', halt_on_error=False, raise_on_error=True)

# deploy taggers/averaged_perceptron_tagger
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except:  # noqa: *
    data_dir = nltk.data.find('.')
    target_dir = os.path.join(data_dir, 'taggers')
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    src_file = os.path.join(os.path.dirname(__file__), '..', 'nltk_packages',
                            'averaged_perceptron_tagger.zip')
    shutil.copyfile(src_file,
                    os.path.join(target_dir, 'averaged_perceptron_tagger.zip'))
    shutil._unpack_zipfile(
        os.path.join(target_dir, 'averaged_perceptron_tagger.zip'), target_dir)

# deploy corpora/cmudict
try:
    nltk.data.find('corpora/cmudict')
except:  # noqa: *
    data_dir = nltk.data.find('.')
    target_dir = os.path.join(data_dir, 'corpora')
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    src_file = os.path.join(os.path.dirname(__file__), '..', 'nltk_packages',
                            'cmudict.zip')
    shutil.copyfile(src_file, os.path.join(target_dir, 'cmudict.zip'))
    shutil._unpack_zipfile(os.path.join(target_dir, 'cmudict.zip'), target_dir)

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except:  # noqa: *
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    nltk.download('averaged_perceptron_tagger',
                  halt_on_error=False,
                  raise_on_error=True)

try:
    nltk.data.find('corpora/cmudict')
except:  # noqa: *
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    nltk.download('cmudict', halt_on_error=False, raise_on_error=True)
