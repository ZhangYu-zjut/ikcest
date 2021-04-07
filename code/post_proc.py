# encoding: utf-8
import numpy as np
import pandas as pd
import math
import argparse

def cal_factor(file1,file2):
    
    # model
    df1 = pd.read_csv(file1, header=None, names=['city', 'zone', 'date', 'ifc'])
    ifc1 = df1.loc[:, 'ifc'].values

    # act
    df2 = pd.read_csv(file2, header=None, names=['city', 'zone', 'date', 'ifc'])
    ifc2 = df2.loc[:, 'ifc'].values

    factor = []
    for i in range(ifc2.shape[0]):
        if(ifc1[i]!=0 and ifc2[i]!=0):
            temp = ifc2[i]/ifc1[i]
            factor.append(temp)
        if (ifc1[i] == 0 and ifc2[i] == 0):
            temp = 1
            factor.append(temp)
        if (ifc1[i] == 0 and ifc2[i] != 0):
            temp = 666
            factor.append(temp)

        if (ifc1[i] != 0 and ifc2[i] == 0):
            temp = 0
            factor.append(temp)
    print("factor is",factor)
    return factor
    pass
    
def rmsle_cmp(file1,file2):
    df1 = pd.read_csv(file1,header=None,names=['city','zone','date','ifc'])
    ifc1 = df1.loc[:,'ifc'].values
    df2 = pd.read_csv(file2,header=None,names=['city','zone','date','ifc'])
    ifc2 = df2.loc[:,'ifc'].values
    rmsle = rmsle_cal(ifc1,ifc2)
    print("rmsle final is",rmsle)
    pass
def rmsle_cal(ifc1,ifc2):
    return math.sqrt( np.mean((np.log1p(ifc1)-np.log1p(ifc2))**2) )
    pass
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
    


