# -*- coding: utf-8 -*-

import os
import shutil

CURRENT_DIR = os.path.dirname(__file__)

def move_cpu_pyd():
    """
    move cpu dll to pyd
    """
    os.remove(f"{CURRENT_DIR}/../python/needle/backend_ndarray/ndarray_backend_cpu.pyd")
    shutil.move(f"{CURRENT_DIR}/../python/needle/backend_ndarray/Release/ndarray_backend_cpu.dll", 
                f"{CURRENT_DIR}/../python/needle/backend_ndarray/ndarray_backend_cpu.pyd")


def move_cuda_pyd():
    """
    move cuda dll to pyd
    """
    os.remove(f"{CURRENT_DIR}/../python/needle/backend_ndarray/ndarray_backend_cuda.pyd")
    shutil.move(f"{CURRENT_DIR}/../python/needle/backend_ndarray/Release/ndarray_backend_cuda.dll", 
                f"{CURRENT_DIR}/../python/needle/backend_ndarray/ndarray_backend_cuda.pyd")
    

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print(f"usage(0:cpu, 1:cuda): python {__file__} 0|1")
    else:
        if int(sys.argv[1]) == 0:
            move_cpu_pyd()
        elif int(sys.argv[1]) == 1:
            move_cuda_pyd()
        else:
            raise NotImplementedError(f"param not supported: {sys.argv[1]}")
