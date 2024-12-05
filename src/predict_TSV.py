from keras.models import load_model
import argparse
from multiprocessing import Process, cpu_count
import sys
system = sys.platform
import psutil
from tqdm import tqdm

from lib.divideData import divideTSVPredictData
from lib.operation import *
from lib.log import *


ATCG_oneHot = {
    'A': [1, 0, 0, 0],
    'T': [0, 1, 0, 0],
    'U': [0, 1, 0, 0],
    'C': [0, 0, 1, 0],
    'G': [0, 0, 0, 1],
}
ntNum = 25


def transX(x):
    return np.array([[ATCG_oneHot[__] for __ in _] for _ in x])


def predictTask(args, process_num, slide_gene_name, slide_sequence, 
                PID_fold, log_fold, TSV_fold, monitor_fold, monitor_len=100):
    # get parameters
    tsv_model_fold = args.tsv_model_fold
    tsv_model_num = args.tsv_model_num

    # Get the PID, bind it to the specified core, log it for process killing
    pid = os.getpid()
    if system == 'win32':
        p = psutil.Process()
        p.cpu_affinity([process_num])
    elif system == 'linux':
        _ = os.system('taskset -pc {} {} '.format(process_num, pid))
    PID_file = '{}/PID_{}.txt'.format(PID_fold, process_num)
    # record PID
    txtSave(PID_file, [str(pid)])

    # create log file
    log_file = '{}/predict_tsv_log_{}.log'.format(log_fold, process_num)
    checkLog(log_file)
    logger = createLogging(log_file, 'Predict TSV data')
    logger.info('-----------------------Process {}(PID-{}) Start--------------------------'.format(process_num, pid))

    monitor_file = '{}/{}_0.txt'.format(monitor_fold, process_num)
    f = open(monitor_file, 'w')
    f.close()

    # ready to predict
    pred_len = len(slide_sequence)
    slide_data = transX(slide_sequence).reshape(pred_len, ntNum, 4, 1)
    predict = []
    for model_idx in range(tsv_model_num):
        # record
        if model_idx % monitor_len == 0:
            monitor_file = '{}/{}_{}.txt'.format(monitor_fold, process_num, model_idx // monitor_len)
            os.remove(monitor_file)
            monitor_file = '{}/{}_{}.txt'.format(monitor_fold, process_num, model_idx // monitor_len + 1)
            f = open(monitor_file, 'w')
            f.close()
            logger.info('Now using model {} to predict'.format(model_idx))
        
        # load model
        model_name = '{}/{}_best_model.h5'.format(tsv_model_fold, model_idx)
        if not os.path.exists(model_name):
            print('Warning: The model named \'{}\' does not exist!'.format(model_name))
            continue
        model = load_model(model_name)
        pred = model.predict(slide_data, verbose=0)  # predict
        pred = pred.reshape(1, pred_len)[0]
        predict.append(pred)
    predict = np.percentile(np.array(predict), 50, axis=0)

    # save data
    TSV_file = '{}/pred_TSV_{}.csv'.format(TSV_fold, process_num)
    save_data = []
    for idx in range(pred_len):
        save_data.append([slide_gene_name[idx], predict[idx]])
    csvSave(TSV_file, save_data)
    logger.info('-----------------------Finish--------------------------')


def getTSV(tsv_output_fold, process_sum, origin_gene, origin_seq, origin_activity):
    # organize the results
    TSV_fold = '{}/pred_tsv'.format(tsv_output_fold)
    predict_tsv = []
    for p_num in range(process_sum):
        TSV_file = '{}/pred_TSV_{}.csv'.format(TSV_fold, p_num)
        predict = csvRead(TSV_file)
        predict_tsv.extend(list(predict))

    # transform predicted slice to original form
    save_data, pred = [['Gene Name', 'UTR Sequence', 'Actual Activity', 'TSV Prediction']], []
    i_origin = 0
    origin = origin_gene[i_origin]
    sequence = origin_seq[i_origin]
    activity = origin_activity[i_origin]
    last_gene = ''
    for i in range(len(predict_tsv)):
        gene = str.split(predict_tsv[i][0], '_')[0]
        tsv = np.exp(float(predict_tsv[i][1]))
        if gene == last_gene:
            pred.append(tsv)
        else:
            if last_gene != '':
                save_data.append(pred)
                i_origin += 1
                origin = origin_gene[i_origin]
                sequence = origin_seq[i_origin]
                activity = origin_activity[i_origin]
            last_gene = gene
            pred = [gene, sequence, activity, tsv]
            while origin != last_gene:
                save_data.append([origin, sequence, ''])
                i_origin += 1
                origin = origin_gene[i_origin]    
                sequence = origin_seq[i_origin]
    save_data.append(pred)
    return save_data


def predictTSV(args):
    # get parameters
    tsv_model_num = args.tsv_model_num  # model nums
    source_file = args.input_file  # input file
    process_sum = args.process_sum  # multiprocess
    tsv_output_fold = args.tsv_output_fold  # output fold

    # get divided data
    origin_gene, origin_seq, divided_gene, divided_seq, Activity = divideTSVPredictData(args)

    # create fold for recording PID
    PID_fold = '{}/PID'.format(tsv_output_fold)
    if os.path.exists(PID_fold):
        removeFolds(PID_fold)
    os.mkdir(PID_fold)

    # create fold for recording log
    log_fold = '{}/log'.format(tsv_output_fold)
    if os.path.exists(log_fold):
        removeFolds(log_fold)
    os.mkdir(log_fold)
    
    # create fold for predictions file
    TSV_fold = '{}/pred_tsv'.format(tsv_output_fold)
    if os.path.exists(TSV_fold):
        removeFolds(TSV_fold)
    os.mkdir(TSV_fold)

    # create fold for monitor
    monitor_fold = '{}/monitor'.format(tsv_output_fold)
    if os.path.exists(monitor_fold):
        removeFolds(monitor_fold)
    os.mkdir(monitor_fold)
    
    # divide tasks
    process_list = []  # process pool
    for p_num in range(process_sum):
        # get assigned sequence tasks for each process
        slide_gene_name = divided_gene[p_num]
        slide_sequence = divided_seq[p_num]
        slide_len = len(slide_gene_name)
        # if there is no data, skip
        if slide_len == 0:
            process_sum -= 1
            continue

        # submit tasks
        p = Process(target=predictTask, args=(args, p_num, slide_gene_name, slide_sequence, 
                                              PID_fold, log_fold, TSV_fold, monitor_fold, ))
        p.start()
        process_list.append(p)

    for p in process_list:
        p.join()
    
    # organize the results
    save_data = getTSV(tsv_output_fold, process_sum, origin_gene, origin_seq, Activity)
    return save_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='You could add these parameter.')
    parser.add_argument('-i', '--input_file', type=str, default='./examples/input/sample_TSV_predict.xlsx', help='input file')
    parser.add_argument('-rmn', '--tsv_model_num', type=int, default=2000, help='TSV model num')
    parser.add_argument('-rmf', '--tsv_model_fold', type=str, default='./saved_models', help='TSV model fold')
    parser.add_argument('-ro', '--tsv_output_fold', type=str, default='./examples/output', help='TSV output fold')
    parser.add_argument('-rsf', '--tsv_save_file', type=str, default='', help='saved TSV file name')
    parser.add_argument('-p', '--process_sum', type=int, default=cpu_count() // 2 - 1, help='process sum')
    parser.add_argument('-ss', '--server_sum', type=int, default=1, help='server sum')
    parser.add_argument('-sn', '--server_num', type=int, default=0, help='server num')
    args = parser.parse_args()

    if args.process_sum >= cpu_count():
        args.process_sum = cpu_count() // 2 - 1
    
    if not os.path.exists(args.input_file):
        print('ERROR: Please ensure the input file path exists!')
        exit()
    
    save_data = predictTSV(args)
    
    save_tsv_file = args.tsv_save_file
    if args.tsv_save_file == '':
        # remove old saved predicted file
        origin_file_name = str.split(args.input_file, '/')[-1]
        origin_file_name = str.split(origin_file_name, '.')[0]
        save_tsv_file = '{}/pred_tsv_{}.csv'.format(args.tsv_output_fold, origin_file_name)
    if os.path.exists(save_tsv_file):
        os.remove(save_tsv_file)

    # save
    csvSave(save_tsv_file, save_data)
