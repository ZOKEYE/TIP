from keras.models import load_model
import argparse
from multiprocessing import Process, cpu_count
import sys
system = sys.platform
import psutil
from tqdm import tqdm
import time

from lib.divideData import divideTIPPredictData
from lib.operation import *
from lib.log import *
from cross_validation_draw import plotDraw


def predictTask(args, process_num, pred_gene_name, pred_sequence, pred_rbv_data, 
                PID_fold, log_fold, TIP_fold, monitor_fold, model_len=100):
    # get parameters
    model_fold = args.tip_model_fold
    model_num = args.tip_model_num

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
    log_file = '{}/predict_tip_log_{}.log'.format(log_fold, process_num)
    checkLog(log_file)
    logger = createLogging(log_file, 'Predict TIP data')
    logger.info('-----------------------Process {}(PID-{}) Start--------------------------'.format(process_num, pid))

    monitor_file = '{}/{}_0.txt'.format(monitor_fold, process_num)
    f = open(monitor_file, 'w')
    f.close()

    # ready to predict
    pred_len = len(pred_sequence)
    predict = []
    for model_idx in range(model_num):
        # record
        if model_idx % model_len == 0:
            monitor_file = '{}/{}_{}.txt'.format(monitor_fold, process_num, model_idx // model_len)
            os.remove(monitor_file)
            monitor_file = '{}/{}_{}.txt'.format(monitor_fold, process_num, model_idx // model_len + 1)
            f = open(monitor_file, 'w')
            f.close()
            logger.info('Now using model {} to predict'.format(model_idx))
        
        # load model
        model_name = '{}/TIP_model_{}.h5'.format(model_fold, model_idx)
        if not os.path.exists(model_name):
            print('Warning: The model named \'{}\' does not exist!'.format(model_name))
            continue
        model = load_model(model_name)
        pred = model.predict(pred_rbv_data, verbose=0)  # predict
        pred = pred.reshape(1, pred_len)[0]
        predict.append(pred)
    predict = np.percentile(np.array(predict), 50, axis=0)

    # save data
    TIP_file = '{}/pred_TIP_{}.csv'.format(TIP_fold, process_num)
    save_data = []
    for idx in range(pred_len):
        save_data.append([pred_gene_name[idx], predict[idx]])
    csvSave(TIP_file, save_data)
    logger.info('-----------------------Finish--------------------------')


def getTIP(TIP_fold, process_sum, origin_gene, origin_seq, origin_activity):
    predict_tip = dict()
    for p_num in range(process_sum):
        TIP_file = '{}/pred_TIP_{}.csv'.format(TIP_fold, p_num)
        predict = list(csvRead(TIP_file))
        for p_data in predict:
            predict_tip[p_data[0]] = p_data[1]

    # transform predicted slice to original form
    save_data = [['Gene Name', 'UTR Sequence', 'Actual Activity', 'Actual Activity-ln', 'Pred TIP', 'Pred TIP-ln']]
    for idx in range(len(origin_gene)):
        gene = origin_gene[idx]
        seq = origin_seq[idx]
        activity = origin_activity[idx]
        pred_tip = predict_tip[gene]
        save_data.append([gene, seq, float(activity), np.log(float(activity)), 
                          np.exp(float(pred_tip)), float(pred_tip)])
    save_data = np.array(save_data)
    return save_data


def predictTIP(args):
    # get parameters
    process_sum = args.process_sum  # multiprocess
    tip_output_fold = args.tip_output_fold  # output fold

    # get divided data
    origin_gene, origin_seq, divided_gene, divided_seq, divided_rbv, Activity = divideTIPPredictData(args)

    # create fold for recording PID
    PID_fold = '{}/PID'.format(tip_output_fold)
    if os.path.exists(PID_fold):
        removeFolds(PID_fold)
    os.mkdir(PID_fold)

    # create fold for recording log
    log_fold = '{}/log'.format(tip_output_fold)
    if os.path.exists(log_fold):
        removeFolds(log_fold)
    os.mkdir(log_fold)
    
    # create fold for predictions file
    pred_tip_fold = '{}/pred_tip'.format(tip_output_fold)
    if os.path.exists(pred_tip_fold):
        removeFolds(pred_tip_fold)
    os.mkdir(pred_tip_fold)

    # create fold for monitor
    monitor_fold = '{}/monitor'.format(tip_output_fold)
    if os.path.exists(monitor_fold):
        removeFolds(monitor_fold)
    os.mkdir(monitor_fold)
    
    # divide tasks
    process_list = []  # process pool
    for p_num in range(process_sum):
        # get assigned sequence tasks for each process
        pred_gene_name = divided_gene[p_num]
        pred_sequence = divided_seq[p_num]
        pred_rbv_data = divided_rbv[p_num]
        pred_len = len(pred_gene_name)
        # if there is no data, skip
        if pred_len == 0:
            process_sum -= 1
            continue

        # submit tasks
        p = Process(target=predictTask, args=(args, p_num, pred_gene_name, pred_sequence, pred_rbv_data,  
                                              PID_fold, log_fold, pred_tip_fold, monitor_fold, ))
        p.start()
        process_list.append(p)

    for p in process_list:
        p.join()
    
    # organize the results
    save_data = getTIP(pred_tip_fold, process_sum, origin_gene, origin_seq, Activity)
    return save_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='You could add these parameter.')
    parser.add_argument('-i', '--input_file', type=str, default='./train/examples/sample_TIP_testset.xlsx', help='dataset file name')
    parser.add_argument('-sf', '--scan_file', type=str, default='./train/examples/sample_TIP_test_scanned.csv', help='scanned dataset file name')
    parser.add_argument('-tmn', '--tip_model_num', type=int, default=1, help='tip model num')
    parser.add_argument('-tmf', '--tip_model_fold', type=str, default='./saved_models/TIP', help='TIP model fold')
    parser.add_argument('-to', '--tip_output_fold', type=str, default='./examples/TIP_output', help='TIP output fold')
    parser.add_argument('-p', '--process_sum', type=int, default=cpu_count() // 2 - 1, help='process sum')
    parser.add_argument('-ss', '--server_sum', type=int, default=1, help='server sum')
    parser.add_argument('-sn', '--server_num', type=int, default=0, help='server num')
    parser.add_argument('-d', '--draw_flag', type=int, default=1, help='draw pic flag, 0->not draw, 1->draw')
    args = parser.parse_args()

    if args.process_sum >= cpu_count():
        args.process_sum = cpu_count() // 2 - 1

    scan_file = args.scan_file

    if not os.path.exists(scan_file):
        # wait scanned file
        print('Please waiting for generating scanned file.', end='')
    while not os.path.exists(scan_file):
        time.sleep(5)
        print('.', end='')
    print('\n')
    
    save_data = predictTIP(args)
    
    # remove old saved predicted file
    origin_file_name = str.split(args.input_file, '/')[-1]
    origin_file_name = str.split(origin_file_name, '.')[0]
    save_tip_file = '{}/pred_tip_{}.csv'.format(args.tip_output_fold, origin_file_name)
    if os.path.exists(save_tip_file):
        os.remove(save_tip_file)

    # save
    csvSave(save_tip_file, save_data)

    # draw scatter pictures if set params
    if args.draw_flag:
        act_data = np.array(save_data[1:, 3]).astype(float)
        pred_data = np.array(save_data[1:, 5]).astype(float)
        save_scatter = '{}/draw_scatter.jpg'.format(args.tip_output_fold)
        plotDraw(act_data, pred_data, save_scatter)
