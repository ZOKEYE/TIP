#!/bin/bash
input_file="./train/input/sample_TIP_trainset.xlsx"
scanned_file="./train/input/sample_TIP_scanned.csv"
rbv_model_num=2000
rbv_model_fold="./saved_models/RBV"
rbv_output_fold="./examples/RBV_output"
process_sum=32

tip_model_fold="./saved_models/TIP"
tip_output_fold="./train/TIP_output"

if [ ! -f $scanned_file ];then
    if [ ! -d $rbv_output_fold ];then
        mkdir $rbv_output_fold
    fi

    rbv_output_file=$rbv_output_fold"/main.out"
    nohup python -u ./src/predict_RBV.py -i $input_file -rmn $rbv_model_num -rmf $rbv_model_fold -ro $rbv_output_fold -rsf $scanned_file -p $process_sum > $rbv_output_file 2>&1 &

    # monitor predicting data
    echo "scanning and predicting input file with RBV model."
    description="Predicting"
    monitor=$rbv_output_fold"/monitor"
    total_num=$(expr $(expr $rbv_model_num / 100) \* $process_sum)
    python ./src/tqdmbar.py $total_num $monitor $description

    echo "Finish predicting scanned file."
fi

python ./src/train_TIP.py -i $input_file -sf $scanned_file -tmf $tip_model_fold -to $tip_output_fold -p $process_sum
