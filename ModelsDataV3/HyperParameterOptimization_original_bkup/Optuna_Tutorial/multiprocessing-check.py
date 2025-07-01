#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 15:51:41 2024

@author: attari.v
"""

import multiprocessing
import os

def worker():
    print(f"Process ID: {os.getpid()} is running.")

if __name__ == '__main__':
    # Create a pool of workers (e.g., 4 workers)
    with multiprocessing.Pool(processes=4) as pool:
        pool.map(worker, range(4))  # Running the worker on 4 processes