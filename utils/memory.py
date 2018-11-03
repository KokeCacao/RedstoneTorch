import os
import sys
import threading

import torch
import psutil
import subprocess
import time

import config

exitFlag = 0


class memory_thread(threading.Thread):
    def __init__(self, threadID, writer):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.writer = writer

    def run(self):
        while(True):
            write_memory(self.writer, "thread")
            time.sleep(1)

def write_memory(writer, arg):
    t = int(time.time())
    if config.TRAIN_GPU_ARG:
        torch.cuda.empty_cache()  # release gpu memory
        sum = 0
        for key, value in get_gpu_memory_map().items():
            writer.add_scalars('stats/GPU-Memory', {"GPU-{}-{}".format(str(key), arg): value}, t)
            sum = sum + value
        writer.add_scalars('stats/GPU-Memory', {"GPU-Sum-{}".format(arg): sum}, t)
    writer.add_scalars('stats/CPU-Usage', {"CPU-Sum-{}".format(arg): psutil.cpu_percent()}, t)

    mem_bytes = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')  # e.g. 4015976448
    mem_gib = mem_bytes / (1024. ** 3)
    writer.add_scalars('stats/CPU-Memory', {"CPU-Sum-{}".format(arg): mem_gib}, t)
    # if psutil.virtual_memory() != None: self.writer.add_scalars('memory/Physical', {"Physical_Mem Usage": psutil.virtual_memory()}, self.count)

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