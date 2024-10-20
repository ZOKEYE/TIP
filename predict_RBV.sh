#!/bin/bash
input_file="./examples/input/sample_RBV_predict.xlsx"
rbv_model_num=2000
rbv_model_fold="./saved_models/RBV"
rbv_output_fold="./examples/RBV_output"
process_sum=32

if [ ! -d $rbv_output_fold ];then
    mkdir $rbv_output_fold
fi

output_file=$rbv_output_fold"/main.out"
nohup python -u ./src/predict_RBV.py -i $input_file -rmn $rbv_model_num -rmf $rbv_model_fold -ro $rbv_output_fold -p $process_sum > $output_file 2>&1 &

# monitor predicting data
echo "Submitting RBV predicting tasks."
description="Predicting"
monitor=$rbv_output_fold"/monitor"
total_num=$(expr $(expr $rbv_model_num / 100) \* $process_sum)
python ./src/tqdmbar.py $total_num $monitor $description

echo "Saving predicted file."
