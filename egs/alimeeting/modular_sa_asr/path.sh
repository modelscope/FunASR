export FUNASR_DIR=$PWD/../../..

export KALDI_ROOT=/Your_Kaldi_root
export DATA_SOURCE=/Your_data_path
export DATA_NAME=Test_2023_Ali_far
export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C

# NOTE(kan-bayashi): Use UTF-8 in Python to avoid UnicodeDecodeError when LC_ALL=C
export PYTHONIOENCODING=UTF-8
export PATH=$FUNASR_DIR/funasr/bin:./utils:$FUNASR_DIR:$PATH
export PYTHONPATH=$FUNASR_DIR:$PYTHONPATH
