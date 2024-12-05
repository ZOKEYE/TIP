# Predicting details

## 1. TSV Model Predicting
This module is mainly responsible for scanning and predicting UTR sequences with a 25nt sliding window based on TSV model. Considering compatibility, regardless of whether the leng, we use the scanning approch for prediction. Since the TSV model uses an ensemble model prediction method based on random forest decision-making, each sequence is predicted by the models in the TSV model group. Ultimately, the median of predicted TIA values will be taken as the output.

### 1.1 input folder
The input file for predicting is placed in this folder by default, and we provide the template file ***[example/input/sample_TSV_predict.xlsx](./input/sample_TSV_predict.xlsx)*** as an example, including the "Gene Name" column, the "UTR Sequence" column, and the "Actual Activity" column. 
- The "Gene Name" column is used to uniquely specify each UTR sequence. Please notice that gene name cannot be underlined or duplicated.
- The "UTR Sequence" column is used for model training, must be not shorter than 25nt in length, and cannot contain any letters other than "ATUCG". In addition, it is permissible for each line to have a different UTR sequence length. 
- The "Actual Activity" column is used to provide the training label for each sequence. This column can be empty if it is only an activity prediction and does not need to be validated.

### 1.2 output file
After completing predicting, the output file will be saved as ***pred_tsv_sample_TSV_predict.csv*** in the examples folder. Additionally, if the "Actual Activity" column does not exist in the input file, that column in the prediction file will be populated with 0.

### 1.3 pred_tsv fold
This folder saves the prediction result file of each subprocess named like ***pred_TSV_0.csv***. The origin gene names, combined with the start position of the subsequence in each sliding window, make up the list of genes in the file.

### 1.4 log folder
The log folder recording the runtime status of each subprocess aids users in real-time monitoring.

### 1.5 PID folder
The PID folder records the process IDs of subprocesses, which helps users terminate program execution in real-time. If you want to kill the subprocess, please execute the following script and note two things: 
- this script will kill the main process and all subprocesses; 
- please pay attention to the pid_fold parameter in the file and make sure that the parameter points to the correct folder (please correspond to the folder under which the PID folder is located).
```
$ bash kill.sh
```

### 1.6 monitor folder
This folder is related to the progress bar and can be ignored.


## 2. TIP Model Predicting
Similar to TIP model training, you can directly provide the TIA scan file of the sequence to be predicted. Other than that, it is basically the same as the TSV model prediction approach.

### 2.1 input folder
The input file for predicting is placed in this folder by default, and we provide the template file ***[example/input/sample_TIP_testset.xlsx](./input/sample_TIP_testset.xlsx)*** as an example, including the "Gene Name" column, the "UTR Sequence" column, and the "Actual Activity" column. 
- The "Gene Name" column is used to uniquely specify each UTR sequence. Please notice that gene name cannot be underlined or duplicated.
- The "UTR Sequence" column is used for model training, must be 90nt in length, and cannot contain any letters other than "ATUCG".
- The "Actual Activity" column is used to provide the training label for each sequence. This column can be empty if it is only an activity prediction and does not need to be validated.

### 2.2 output file
After completing predicting, the output file will be saved as ***pred_tip_sample_TIP_testset.csv*** in the examples folder, and a scatterplot of actual activity and predicted TIP will be generated as ***draw_scatter.jpg***. Additionally, if the "Actual Activity" column does not exist in the input file, that column in the prediction file will be populated with 0.

### 2.3 pred_tip fold
The pred_tip folder saves the prediction result file of each subprocess named like ***pred_TIP_0.csv***. 

### 2.4 log folder
The log folder recording the runtime status of each subprocess aids users in real-time monitoring.

### 2.5 PID folder
The PID folder records the process IDs of subprocesses, which helps users terminate program execution in real-time.

### 2.6 monitor folder
This folder is related to the progress bar and can be ignored.

