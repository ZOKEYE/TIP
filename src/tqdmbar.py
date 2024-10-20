from tqdm import *
import sys
import os
import time

from lib.operation import removeFolds


def monitor_files_exists(total_num, monitor, description):
    wait_count = 0
    with tqdm(total=total_num) as pbar:
        pbar.set_description('{}:'.format(description))
        file_num = len(os.listdir(monitor))
        last_num = 0
        while file_num < total_num:
            file_num = len(os.listdir(monitor))
            add_num = file_num - last_num
            if add_num > 0:
                pbar.update(add_num)
                last_num = file_num
            else:
                time.sleep(5)
                wait_count += 1
                if wait_count == 24:
                    print('Warning: There may have been some errors! Please check if there are any error messages in main.out!')


def monitor_files_masks(total_num, monitor, description):
    wait_count = 0
    with tqdm(total=total_num) as pbar:
        pbar.set_description('{}:'.format(description))
        def count_masks(file_lists):
            count = 0
            for file in file_lists:
                filename = str.split(file, '.')[0]
                count += int(str.split(filename, '_')[1])
            return count
        
        masks = count_masks(os.listdir(monitor))
        last_masks = 0
        while masks < total_num:
            masks = count_masks(os.listdir(monitor))
            add_masks = masks - last_masks
            if add_masks > 0:
                pbar.update(add_masks)
                last_masks = masks
            else:
                time.sleep(5)
                wait_count += 1
                if wait_count == 10:
                    print('Warning: There may have been some errors! Please check if there are any error messages in main.out!')


if __name__ == "__main__":
    total_num = int(sys.argv[1])
    monitor = sys.argv[2]
    description = sys.argv[3]
    time.sleep(5)
    
    while not os.path.exists(monitor):
        time.sleep(5)
    if description == 'Training':
        monitor_files_exists(total_num, monitor, description)
    elif description == 'Predicting':
        monitor_files_masks(total_num, monitor, description)
    