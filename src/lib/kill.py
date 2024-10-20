import os
import sys


pid_fold = sys.argv[1]
system = sys.platform
process_sum = len(os.listdir(pid_fold))
for i in range(process_sum):
    PID_file = '{}/PID_{}.txt'.format(pid_fold, i)
    f = open(PID_file, 'r')
    pid = f.readline()
    while pid != '':
        if system == 'win32':
            os.system('"taskkill /F /PID {}"'.format(pid))
        elif system == 'linux':
            _ = os.system('kill ' + pid)
        pid = f.readline()
    f.close()
