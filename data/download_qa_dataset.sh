#!/bin/bash

# Download korquad 2.1 datasets

mkdir -p {data file dir}/korquad_2/train
for var in {0..12}
do
    var=$(printf %02d $var)
    wget -P {data file dir}/korquad_2/train https://github.com/korquad/korquad.github.io/raw/master/dataset/KorQuAD_2.1/train/KorQuAD_2.1_train_${var}.zip
done

mkdir -p {data file dir}/korquad_2/dev
for var in {0..1}
do
    var=$(printf %02d $var)
    wget -P {data file dir}/korquad_2/dev https://github.com/korquad/korquad.github.io/raw/master/dataset/KorQuAD_2.1/dev/KorQuAD_2.1_dev_${var}.zip
done

cd {data file dir}/korquad_2
cd {data file dir}/train
unzip '*.zip'
rm *.zip
cd ..

cd dev
unzip '*.zip'
rm *.zip
cd ..
