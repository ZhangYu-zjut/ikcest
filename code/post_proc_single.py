'''
convert the predict result from 30*903 matrix to the standard submission format
'''
import os
import numpy as np
import pandas as pd
import warnings
import math
import argparse

warnings.filterwarnings("ignore")


def get_date_pred():
    pred_start = '2120-06-30'
    days_pred = 30
    # predict date range
    pred = pd.date_range(start=pred_start, periods=days_pred, freq="D")
    pred = pd.DataFrame(pred).reset_index()
    pred.columns = ['dateid', 'date']
    pred['date'] = pred['date'].apply(lambda x: int(x.strftime('%Y%m%d')))
    return pred    
    
def to_submission(pred_path, pred_file, submission_file='submission.csv'):
    # must include zone_all.csv file in the same folder
    input_data_file = pred_file
    save_data_file = submission_file
    pred = get_date_pred()
    zone_all = pd.read_csv('zone_all.csv')
    ifc = pd.read_csv(input_data_file, header=None)
    ifc = pd.DataFrame(ifc.unstack()).reset_index()
    ifc.columns = ['zoneid1', 'dateid', 'ifc']
    ifc = ifc.merge(pred).merge(zone_all)
    ifc = ifc[['city', 'zoneid', 'date', 'ifc']]
    ifc.ifc = ifc.ifc.astype(int)
    ifc.to_csv(save_data_file, index=False, header=False)
	


def main1(args):
    pred_path = './predict/temp/'
    pred_file = args.file
    submission_file = args.save
    # convert the matrix to standard submission format
    if(args.file!=None):
        to_submission(pred_path, pred_file, submission_file)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fid', type=str,help='id of the file')
    parser.add_argument('--file', type=str,default=None,help='location of the model file')
    parser.add_argument('--save', type=str,help='location of the save file')
    args = parser.parse_args()
    main1(args)
    
