#!/bin/bash

# Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
# Licensed under the MIT license.
#
# BLSTM-based model experiment
./run.sh --train-config conf/blstm/train.yaml --average-start 20 --average-end 20 \
         --adapt-config conf/blstm/adapt.yaml --adapt-average-start 10 --adapt-average-end 10 \
         --infer-config conf/blstm/infer.yaml $*
