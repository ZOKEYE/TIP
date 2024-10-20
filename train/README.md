# Training details

## 1. RBV Model Training

### 1.1 input folder
The input file for training is placed in this folder by default, and we provide the template file ***[train/input/sample_RBV_dataset.xlsx](./train/input)*** as an example, including the "Gene Name" column, the "UTR Sequence" column, and the "Actual Activity" column. 
- The "Gene Name" column is used to uniquely specify each UTR sequence. Please notice that gene name cannot be underlined or duplicated.
- The "UTR Sequence" column is used for model training, must be 25nt in length, and cannot contain any letters other than "ATUCG". 
- The "Actual Activity" column is used to provide the training label for each sequence.
> Please be careful not to modify the column names when using your own data! 

### 1.2 pred_test folder
During the training of each model, 10% of the dataset is randomly divided as a test set for accuracy evaluation. The prediction result file of the test set will be saved in this folder named like ***0_pred_test.csv***. The file saves, in addition to the input information, the actual activity and the predicted TIA of each UTR sequence, as well as the result of their logarithmic taking. 

### 1.3 scatter_test folder
The scatter_test folder includes scatter plots named like ***0_scatter_test.jpg*** corresponding to the predicted TIA values and the actual activity of the test set partitioned under each model.

### 1.4 log folder
The log folder recording the runtime status of each subprocess aids users in real-time monitoring. This folder is only used in multiprocess mode, i.e. when ***setup_RBV.sh*** is running.

### 1.5 PID folder
The PID folder records the process IDs of subprocesses, which helps users terminate program execution in real-time. This folder is only used in multiprocess mode, i.e. when ***setup_RBV.sh*** is running. If you want to kill the subprocess, please execute the following script and note two things: 
- this script will kill the main process and all subprocesses; 
- please pay attention to the pid_fold parameter in the file and make sure that the parameter points to the correct folder (please correspond to the folder under which the PID folder is located).
```
$ bash kill.sh
```

### 1.6 other important files
After completing all model training, cross-validation results and scatter plots will be saved as ***test_cross_validation_pred.csv*** and ***scatter_cross_validation.jpg***. Additionally, during batch processing with multiple processes, the main process's log file ***main.out*** will also be in train folder. 

### 1.7 Arguments
**-d, --dataset** 
- The **xlsx file** including the UTR samples for training. The default path used is ***[./train/input/sample_RBV_dataset.xlsx](./train/input)***.

**-rmn, --rbv_model_num**
- The RBV models involved in prediction. The default value is 2000.

**-rmf, --rbv_model_fold**
- The folder to save RBV model. The default path used is ***[/saved_models/RBV](/saved_models/RBV)***.

**-ro, --rbv_output_fold**
- The folder to save output data. The default path used is ***[./train/RBV_output](./train/RBV_output)***.

**-bs, --batch_size**
- The batch size used to train the model. The default value is 128.

**-lr, --learning_rate**
- The learning rate of the Adam optimizer used to optimize the model parameters. The default value is 1e-3. If 4 is provided, the learning rate will be 1e-4.

**-p, --process_sum**
- Maximum number of processes in multiprocess mode, which can be set manually by the user. The default value is based on the number of cores in the CPU.

## 2. TIP Model Training

### 2.1 input folder
The input file for training is placed in this folder by default, and we provide the template file ***[train/input/sample_TIP_trainset.xlsx](./train/input)*** as an example, including the "Gene Name" column, the "UTR Sequence" column, and the "Actual Activity" column, which is similar to RBV model training. Since the input to the TIP model is TIA matric of UTR sequence, you can choose either of the two methods under:
- Allow the programme to perform the scan automatically, without performing any additional actions, but you need to pay attention to the output path of the scan file in the script.
- Enter the scanned file of the input file as a parameter, at this point either ***train_TIP.sh*** script or ***setup_TIP.sh*** script, you need to change the scanned_file parameter to the absolute path of the scanned file.

### 2.2 output files
Execution of train_TIP.sh script and setup_TIP.sh script have separate output files. At the end of the run of train_TIP.sh script, five files are generated.
- TIP_train_predict.csv: saves predicted TIP of the training set.
- TIP_train_plot.jpg: saves the mean square error (MSE) line graph for the training and validation sets at each epoch.
- TIP_train_scatter.jpg: saves a scatter plot of the actual activity and predicted TIP for each sequence in the training set, both taken as logarithmic.
- TIP_val_predict.csv: saves predicted TIP of the validation set.
- TIP_val_scatter.jpg: saves a scatter plot of the actual activity and predicted TIP for each sequence in the validation set, both taken as logarithmic.

As for setup_TIP.sh, it generates the prediction result file and scatterplot named like ***cross_validation_pred.csv*** and ***scatter_cross_validation.jpg*** for the dataset under the optimized cross-validation method. This script uses the following folders during the TIP model training process.

### 2.3 pred_val folder
During the training of each model, 10% of the dataset is randomly divided as a validation set for accuracy evaluation. The prediction result file of the test set will be saved in this folder named like ***0_val_pred.csv***. The file saves, in addition to the input information, the actual activity and the predicted TIA of each UTR sequence, as well as the result of their logarithmic taking. 

### 2.4 scatter_val folder
The scatter_test folder includes scatter plots named like ***0_val_scatter.jpg*** corresponding to the predicted TIP values and the actual activity of the validation set partitioned under each model.

### 2.5 log folder
The log folder recording the runtime status of each subprocess aids users in real-time monitoring.

### 2.6 PID folder
The PID folder records the process IDs of subprocesses, which helps users terminate program execution in real-time.

### 2.7 Arguments
**-i, --input_file** 
- The **xlsx file** including the UTR samples for training. The default path used is ***[./train/input/sample_RBV_dataset.xlsx](./train/input)***.

**-sf, --scan_file** 
- The **csv file** including the TIA matric scanned with a 25nt sliding window based on RBV model group, and the input sequence is aligned with the UTR sequence in the input_file. The default path used is **NULL**.

**-tmn, --tip_model_num**
- The TIP models involved in prediction. The default value is 2000.

**-tmf, --tip_model_fold**
- The folder to save TIP model. The default path used is ***[/saved_models/TIP](/saved_models/TIP)***.

**-to, --tip_output_fold**
- The folder to save output data. The default path used is ***[./train/TIP_output](./train/TIP_output)***.

**-bs, --batch_size**
- The batch size used to train the model. The default value is 128.

**-lr, --learning_rate**
- The learning rate of the Adam optimizer used to optimize the model parameters. The default value is 1e-3. If 4 is provided, the learning rate will be 1e-4.

**-p, --process_sum**
- Maximum number of processes in multiprocess mode, which can be set manually by the user. The default value is based on the number of cores in the CPU.

> Please note that since the RBV model will be used for scanning while training TIP models, you need to pay attention to checking and modifying the parameters related to RBV prediction in the script, such as the number of RBV models, the intermediate result saving path, etc.

