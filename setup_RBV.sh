#!/bin/bash
rbv_model_num=2000
rbv_model_fold="./saved_models/RBV"
rbv_output_fold="./train/RBV_output"
dataset="./train/input/sample_RBV_dataset.xlsx"
batch_size=128
learning_rate=0.001
process_sum=32

if [ ! -d $rbv_model_fold ];then
    mkdir $rbv_model_fold
fi

if [ ! -d $rbv_output_fold ];then
    mkdir $rbv_output_fold
fi

output_file=$rbv_output_fold"/main.out"
nohup python -u ./src/setup_RBV.py -rmn $rbv_model_num -rmf $rbv_model_fold -ro $rbv_output_fold -d $dataset -bs $batch_size -lr $learning_rate -p $process_sum > $output_file 2>&1 &

echo "Submitting RBV models training tasks."
description="Training"
monitor=$rbv_output_fold"/pred_test"
python ./src/tqdmbar.py $rbv_model_num $monitor $description
echo "Finish."
