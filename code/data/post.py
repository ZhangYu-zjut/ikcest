# encoding: utf-8
import numpy as np
import pandas as pd
import math
import argparse
import os

def judge(file1,file2):
    
    #
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_file', type=str,help='path of the input file')
    parser.add_argument('--cmp_file', type=str,help='path of the compare file')
    args = parser.parse_args()
    
    #root_path = "./submission_files/"
    file1 = args.in_file#"./predict/submission_res.csv"
    file2 = args.cmp_file#"./predict/submission_X271_rmsle_1.2315.csv"
    cal_factor(file1,file2)
    rmsle_cmp(file1,file2)
    


