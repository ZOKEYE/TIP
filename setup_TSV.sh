#!/bin/bash
tsv_model_num=2000
tsv_model_fold="./saved_models/TSV"
tsv_output_fold="./train/TSV_output"
dataset="./train/input/sample_TSV_dataset.xlsx"
batch_size=128
learning_rate=0.001
process_sum=32

if [ ! -d $tsv_model_fold ];then
    mkdir $tsv_model_fold
fi

if [ ! -d $tsv_output_fold ];then
    mkdir $tsv_output_fold
fi

output_file=$tsv_output_fold"/main.out"
nohup python -u ./src/setup_TSV.py -rmn $tsv_model_num -rmf $tsv_model_fold -ro $tsv_output_fold -d $dataset -bs $batch_size -lr $learning_rate -p $process_sum > $output_file 2>&1 &

echo "Submitting TSV models training tasks."
description="Training"
monitor=$tsv_output_fold"/pred_test"
python ./src/tqdmbar.py $tsv_model_num $monitor $description
echo "Finish."
