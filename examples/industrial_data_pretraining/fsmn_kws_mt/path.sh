export FUNASR_DIR=$PWD/../../..

# NOTE(kan-bayashi): Use UTF-8 in Python to avoid UnicodeDecodeError when LC_ALL=C
export PYTHONIOENCODING=UTF-8
export PATH=$FUNASR_DIR/funasr/bin:$PATH
