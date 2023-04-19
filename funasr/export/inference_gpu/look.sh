name=blade.fp16
rm -rf z_tmp/*
cp $name.pt z_tmp
cd z_tmp
unzip $name.pt
file=`find $name/code/__torch__/funasr/ -name "*.py"`
cat $file
