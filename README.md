# TIP MODEL
The repo contains:

1. RBV model: a method for prediction of 25nt 5'UTR TIA (Translation Initiation Activity) based on CNN (Convolutional Neural Network) structure.
2. TIP model: a method for prediction of ***longer*** 5'UTR TIP (Translation Initiation Potential) based on TIA values predicted with RBV model through a 25nt sliding window and feature fusion with MLP (Multilayer Perceptron) structure.

In this package, we provides resources including: source codes of training RBV model and TIP model, our trained models and usage examples. We also offer a ***[web server](https://www.baidu.com)*** as an online tool.

## Table of Contents
> warning: Please note that RBV model and TIP model are completely free for academic usage. However it is licenced for commercial usage. Please first refer to the License section for more info.

- Installation
- Training
- Usage Examples
- Models Availability
- Dataset Availability
- License
- Citation

## 1. Installation

### 1.1 Create and activate a new virtual environment
RBV model and TIP model are easy to use with tensorflow package. We recommend you to build a python virtual environment with Anaconda.
```
$ conda create -n TIP python=3.10.4 -y
$ conda activate TIP
```

### 1.2 Install the package and other requirements
```
$ git clone https://github.com/
$ cd ./TIP-main
$ python -m pip install -r requirements.txt
```

## 2. Training (Skip this section if you predict on trained models)

### 2.1 Data Processing
Please see the template data at ***[train/input/sample_RBV_dataset.xlsx](./train/input)*** and ***[train/input/sample_TIP_dataset.xlsx](./train/input)***. If you are trying to train model with your own data, please process you data into the same format as it. For details, please refer to the ***[README.md](./train/README.md)*** file in the folder ***[train](./train)***.

### 2.2 RBV Model Training
We provide a simple non-parallelized RBV model training program, which is more suitable for a small number of model training. Please call the ***[train_RBV.py](./src/train_RBV.py)*** script from the folder ***[src](./src)***, and we provide optimized parameters. Support for adjusting parameters.
```
$ python ./src/train_RBV.py -rmn 2
```
In the article, we adopted an ensemble model prediction method based on random forest decision-making, which requires training multiple RBV models (We used 2000 models as a model group). We provide programs that run in parallel with multiple processes, please call the ***[setup_RBV.sh](./setup_RBV.sh)*** scripy, also support for adjusting parameters. For details, please refer to the ***[README.md](./train/README.md)*** file in the folder ***[train](./train)***.
```
$ bash setup_RBV.sh
```
With the above command, the following files will be generated: the prediction file ***e.g. 0_pred_test.csv*** with each model ***e.g. 0_best_model.h5*** , prediction file ***test_cross_validation_pred.csv*** with cross validation method and scatter plots ***scatter_cross_validation.jpg***.


### 2.3 TIP Model Training
A sliding and scanning operation of the input sequences is performed before the TIP model training. Specifically, the input to the TIP model is the TIA matrix, which is obtained from the origin sequence of 60nt upstream and 30nt downstream of the start codon AUG, through scanning with a 25nt sliding window and predicting based on the RBV model group. Therefore, the program will default to scan the input sequences and generate the corresponding files before TIP model training. In order to avoid the waste of time caused by repeated scanning, you can directly use the scanned csv file as the input parameter. See Chapter 3 of the scanning and predicting method, and the ***[README.md](./train/README.md)*** file in the folder ***[train](./train)*** for more details. You can run ***[train_TIP.sh](./train_TIP.sh)*** script for simple training of a single model.
```
$ python ./src/train_TIP.py
```
Also, we provide programs that run in parallel with multiple processes, please call the ***[setup_TIP.sh](./setup_TIP.sh)*** scripy, also support for adjusting parameters. Note that you can preset the scan file in the script. Please refer to the ***README.md*** file in the folder ***[train](./train)*** for details.
```
$ bash setup_TIP.sh
```
With the above command, the following files will be generated: the prediction file ***e.g. 0_pred_val.csv*** with each model, prediction file ***test_cross_validation_pred.csv*** with cross validation method and scatter plots ***scatter_cross_validation.jpg***.


## 3. Usage Examples
### 3.1 RBV Model Prediction
TIA prediction based on the RBV model consists of two main aspects:
- Prediction of 25nt short UTR sequence regulatory capacity
- Prediction of long sequences based on 25nt sliding window scanning.

In order to be compatible with the two requirements, we both use the sliding scan method to process the input sequences. We provide a template file ***[sample_RBV_predict.xlsx](./examples/input)*** and a script file ***[predict_RBV.sh](./predict_RBV.sh)*** in the code base. For details, please refer to the ***[README.md](./examples/README.md)*** file in the folder ***[examples](./examples)***.
```
$ bash predict_RBV.sh 
```
You can observe the progress of the program in real time from the command line, and the following files will be generated: the prediction file ***e.g. pred_RBV_0.csv*** of each subprocess and the final prediction file ***e.g. pred_rbv_sample_RBV_predict.csv***.


### 3.2 TIP Model Prediction
Similar to TIP model training, a sliding scan of the input sequences with RBV model group is required before prediction. Therefore, we also provide the path of the scanned file as a parameter in the script file, so as to avoid wasting time by repeated scanning. We provide a template file ***[sample_TIP_testset.xlsx](./examples/input)*** in the code base. For details, please refer to the ***[README.md](./examples/README.md)*** file in the folder ***[examples](./examples)***.
```
$ bash predict_TIP.sh
```
You can observe the progress of the program in real time from the command line, and the following files will be generated: the prediction file ***e.g. pred_TIP_0.csv*** of each subprocess, a scatterplot of actual activity and predicted TIP named ***draw_scatter.jpg***, and the final prediction file ***e.g. pred_rbv_sample_RBV_predict.csv***.

## Models Availability
Access our trained model.
- download RBV model: https://github.com/
- download TIP model: https://github.com/

## Dataset Availability
If you need to access to the original 5'UTR library, please email me at keyezhou@163.com.

## License


## Citation
If you have any question regarding our paper or codes, please feel free to start an issue or email Keyi Zhou (keyezhou@163.com).

If you have used our RBV model or TIP model in your research, please kindly cite the following publications:

```
@article {TIP,
    author = {Keyi Zhou},
    title = "{TIP MODEL}",
    journal = {},
    volume = {},
    number = {},
    pages = {},
    year = {},
    month = {},
    doi = {},
    url = {}
}
```