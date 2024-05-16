# TIP MODEL
The repo contains:

1. 25nt-value model: a method for prediction of 25nt 5'UTR TIP (Translation Initiation Potential) based on CNN (Convolutional Neural Network).
2. TIP model: a method for prediction of **longer** 5'UTR TIP based on sliding and linear combinations with 25nt-value model.

In this package, we provides resources including: source codes of training the 25nt-value model and TIP model, trained models and usage examples. We also offer a [web server](https://www.baidu.com) as an online tool.

## Table of Contents
> warning: Please note that the 25nt-value and TIP model are completely free for academic usage. However it is licenced for commercial usage. Please first refer to the License section for more info.

- Installation
- Quick Start
- Usage Examples
    - 25nt-value Prediction
    - TIP Prediction
- Models Availability
- Dataset Availability
- License
- Citation

## Installation
25nt-value model is easy to use with tensorflow package. We recommend you to build a python virtual environment with Anaconda.

### Initialize with requirements installed
For easy requirement handling, you can use TIP.yml files to initialize conda environment with requirements installed.

```
$ conda env create --name TIP -f TIP.yml
$ conda activate TIP
```

### Create and activate a new virtual environment
You can also install virtual environment by yourself.

```
$ conda create -n TIP python=3.10.4 -y
$ conda activate TIP
$ git clone https://github.com/
$ cd ./TIP-main
$ python -m pip install -r requirements.txt
```


## Quick Start
Please call the run_25nt_value.py script from the ./src directory, and we provide optimized parameters. Also support for adjusting parameters.
```
$ cd ./src
$ python run_RBV.py
```

<!-- **Arguments** -->
#### Arguments
**-mn, --model_num**
- The models involved in prediction, the default value is 2000

**-d, --dataset** 
- The CSV file including the UTR samples for training. The default path used is '../data/train/utrdb.xlsx'.

**-bs, --batch_size**
- The batch size used to train the model, the default value is 128

**-lr, --learning_rate**
- The learning rate of the Adam optimizer used to optimize the model parameters. The default value is 1e-3. If 4 is provided, the learning rate will be 1e-4.


## Usage Examples
The input CSV file should be present in the ../data/inputs directory. We have provided 2 input sample files already in the codebase - tip25.csv for 25nt-value model and tip90.csv for TIP model. Moereover, the intermediate files will be saved in the ../data/record/out folder. In order to speed up the running of the program, we use a multi-threaded parallel approach, and the PID information will be saved in the ../data/record/PID folder.

### 25nt-value Model Prediction
```
# The argument '-i' or '--input' represent the input file name
$ python predict_25nt_value.py -i tip25.csv
```
The input CSV file contains two columns: gene name (or identifier) and UTR sequence. The UTR length is allowed to vary, but is at least 25nt. When the length remains 25nt, the model will directly predict the 25nt-value TIP. Otherwise, the model will scan the sequence with a sliding window of 25nt size and output the sliding results. The result will be present in the ../data/output folder.
> Please notice that gene name (or identifier) cannot be underlined or duplicated.

### TIP Model Prediction
```
# The argument '-i' or '--input' represent the input file name
$ python predict_90nt_TIP.py -i tip90.csv
```
The input CSV file contains two columns: gene name (or identifier) and UTR sequence. The UTR length is allowed to vary, but is at least 90nt. When the length exceeds 90nt, we wil still scan with the method of sliding window. The result will be present in the ../data/output folder.
> Please notice that gene name (or identifier) cannot be underlined or duplicated.

## Models Availability
<!-- <table>
    <tr>
        <td>Model</td>
        <td>Github</td>
    </tr>
    <tr>
        <td>25nt-value Model</td>
        <td>[download](https://github.com/)</td>
    </tr>
</table> -->
Access our trained model.
- download 25nt-value TIP model: https://github.com/

## Dataset Availability


## License


## Citation
If you have any question regarding our paper or codes, please feel free to start an issue or email Keyi Zhou (keyezhou@163.com).

If you have used our 25nt-value or TIP model in your research, please kindly cite the following publications:

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
``` = {}
}
```
