import logging
import os


def createLogging(log_file, log_name):
    # create log
    logger = logging.getLogger(log_name)
    logging.basicConfig(filename=log_file,
                        format="'%(asctime)s - %(name)s - %(levelname)s - %(message)s - %(funcName)s',",
                        level=logging.INFO)
    return logger


def checkLog(log_file, limit_len=1000):
    # Delete half of the log file if it is too long
    if not os.path.exists(log_file):
        return
    log_data = []
    f = open(log_file, 'r')
    log = f.readline()
    while log != '' and log != '\n':
        log_data.append(log)
        log = f.readline()
    f.close()
    if len(log_data) <= limit_len:
        return
    os.remove(log_file)
    f = open(log_file, 'w', newline='')
    for i in range(int(limit_len / 2), len(log_data)):
        f.write(log_data[i])
    f.close()
    return
