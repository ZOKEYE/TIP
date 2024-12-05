#!/bin/bash
input_file="./examples/input/sample_TSV_predict.xlsx"
tsv_model_num=2000
tsv_model_fold="./saved_models/TSV"
tsv_output_fold="./examples/TSV_output"
process_sum=32

if [ ! -d $tsv_output_fold ];then
    mkdir $tsv_output_fold
fi

output_file=$tsv_output_fold"/main.out"
nohup python -u ./src/predict_TSV.py -i $input_file -rmn $tsv_model_num -rmf $tsv_model_fold -ro $tsv_output_fold -p $process_sum > $output_file 2>&1 &

# monitor predicting data
echo "Submitting TSV predicting tasks."
description="Predicting"
monitor=$tsv_output_fold"/monitor"
total_num=$(expr $(expr $tsv_model_num / 100) \* $process_sum)
python ./src/tqdmbar.py $total_num $monitor $description

echo "Saving predicted file."
