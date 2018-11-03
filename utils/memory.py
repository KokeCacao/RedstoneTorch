import os
import sys
import threading
import time

import torch
import psutil
import subprocess

import config

exitFlag = 0


class memory_thread(threading.Thread):
    def __init__(self, threadID, writer):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.writer = writer
        self.count = 0

    def run(self):
        while(True):
            if config.TRAIN_GPU_ARG:
                torch.cuda.empty_cache()  # release gpu memory
                for key, value in get_gpu_memory_map().items():
                    self.writer.add_scalars('memory/GPU', {"GPU-" + str(key): value}, self.count)
            self.writer.add_scalars('memory/CPU', {"CPU Usage": psutil.cpu_percent()}, self.count)
            # if psutil.virtual_memory() != None: self.writer.add_scalars('memory/Physical', {"Physical_Mem Usage": psutil.virtual_memory()}, self.count)
            self.count = self.count + 1
            time.sleep(1)

def get_gpu_memory_map():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = dict()
    try:
        result = subprocess.check_output(
            [
                'nvidia-smi', '--query-gpu=memory.used',
                '--format=csv,nounits,noheader'
            ]) #nvidia-smi --query-gpu=memory.used --format=csv,nounits,noheader
        # Convert lines into a dictionary
        gpu_memory = [int(x) for x in result.strip().split('\n')]
        result = dict(zip(range(len(gpu_memory)), gpu_memory))
    except KeyboardInterrupt as e:
        print(e)
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
    return result