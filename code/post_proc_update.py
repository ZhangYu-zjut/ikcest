import os
import numpy as np
import pandas as pd
import time
import re
import warnings
import argparse
import torch

warnings.filterwarnings("ignore")
    
def get_files_pool(combine_path):
    files_pool={}
    for file in os.listdir(combine_path):
            m= re.match('submission_(\D\d{3})\.csv$', file)
            if m:
                files_pool[m.group(1)] = [file]

    return files_pool
def city_rmsle(g):
    x=g.rmsle_base_cover.values
    x=np.power(x,2)
    return np.sqrt(x.mean())
def whole_rmsle(x):
    x=np.power(x,2)
    return np.sqrt(x.mean())
    
def post():
    path1 = "./predict/format"
    path2 = "./predict/temp"
    for file in os.listdir(path1):
        os.remove(os.path.join(path1,file))
    for file in os.listdir(path2):
        os.remove(os.path.join(path2,file))
    os.remove(os.path.join("./predict/","submission_C105_model.csv"))
        
def combine(combine_path,ID_base,ID_cover,cover_range,save_id,save_path):
    # cover predict value with better value in different citys and date ranges
    name_part=str(cover_range).replace("'",'')
    
    for index,sub_range in enumerate(cover_range):
        citys,date_start,date_end=sub_range
        if citys=='all':
            citys='ABCDEFGHIJK'
        date_start=int(time.strftime('2120%m%d',time.strptime(date_start,'%m-%d')))
        date_end=int(time.strftime('2120%m%d',time.strptime(date_end,'%m-%d')))
        if date_end<date_start:
            date_start,date_end=date_end,date_start
        cover_range[index]=[citys,date_start,date_end]
    
    
    files_pool=get_files_pool(combine_path)
    if ID_base not in files_pool.keys():
        print("error: file not find!",ID_base)
        return -1
    if ID_cover not in files_pool.keys():
        print("error: file not found!",ID_cover)
        return -1
    df_base=pd.read_csv(os.path.join(combine_path,files_pool[ID_base][0]),
                        header=None,names=['city', 'zoneid', 'date', 'ifc_base'])
    df_cover=pd.read_csv(os.path.join(combine_path,files_pool[ID_cover][0]),
                        header=None,names=['city', 'zoneid', 'date', 'ifc_cover'])

    df=df_base.merge(df_cover)
    
    # key step: cover infection predict value with better value
    df['ifc']=df.ifc_base
    for sub_range in cover_range:
        citys,date_start,date_end=sub_range
        df.loc[(df.city.isin(list(citys)))&(df.date>=date_start)&(df.date<=date_end),'ifc']=df.ifc_cover
    
    
    # save file
    save_name ="submission_%s.csv"%(save_id)
    df_save=df[['city','zoneid','date','ifc']]
    df_save.ifc=df_save.ifc.astype(int)
    if(save_id == "C105"):
        save_name = "submission_%s_model.csv"%save_id
        save_path = "./predict/"
        save_file = save_path + save_name
        df_save.to_csv(save_file,header=None,index=False)
    
    else:
        save_file = save_path + save_name
        df_save.to_csv(save_file,header=None,index=False)
    return df
    

    
def update_1(args):
    combine_path = "./predict/format/"
    save_path = combine_path
    save_id = "C101"
    df=combine(combine_path,'Y081','Y055',[['A','6-30','7-08'],['DEGH','7-18','7-24']],save_id,save_path)
    save_id = "C102"
    df=combine(combine_path,'C101','X161',[['K','6-30','7-15']],save_id,save_path)
    save_id = "C103"
    df=combine(combine_path,'C102','T193',[['CIJ','6-30','7-29'],['K','7-15','7-29']],save_id,save_path)
    save_id = "C104"
    df=combine(combine_path,'C103','M211',[['BH','6-30','7-04'],['F','6-30','7-09']],save_id,save_path)
    save_id = "C105"
    save_path = args.out_file
    df=combine(combine_path,'C104','Z135',[['DEG','6-30','7-09'],['BH','7-05','7-09']],save_id,save_path)
    pass

def get_best_curve(src_path,base_file,pt_file2):
    # load predict infection base value
    df_base = pd.read_csv(os.path.join(src_path, base_file),
                          header=None,
                          names=['city', 'zoneid', 'date', 'ifc_cover'])
    df_base.rename(columns={'ifc_cover': 'ifc_base'}, inplace=True)
    df_base.ifc_base.replace(0, 1, inplace=True)

    # some additional files to help convert attention to standard format
    zone_all = df_base[['city', 'zoneid']].sort_values(['city', 'zoneid']).drop_duplicates()
    zone_all = zone_all.reset_index().drop(columns='index').reset_index().rename(columns={'index': 'zoneid1'})
    pred=df_base[['date']].sort_values('date').drop_duplicates()
    pred = pred.reset_index().drop(columns='index').reset_index().rename(columns={'index': 'dateid'})

    # load attention matrix, convert to the same format as submission
    pt_path = "./mysave/"
    pt = torch.load(os.path.join(pt_path, pt_file2), map_location=torch.device('cpu'))
    attention = np.array(pt['attention'])
    df_atte = pd.DataFrame(pd.DataFrame(attention).unstack()).reset_index()
    df_atte.columns = ['zoneid1', 'dateid', 'atte']
    df_atte = df_atte.merge(pred).merge(zone_all)
    df_atte = df_atte[['city', 'zoneid', 'date', 'atte']]

    # predict infection base value times attention
    df = df_base.merge(df_atte)
    df['ifc_pred'] = df.atte * df.ifc_base
    df.ifc_pred = df.ifc_pred.apply(lambda x: round(x))
    df_pred=df[['city', 'zoneid', 'date', 'ifc_pred']]
    df_pred.to_csv(os.path.join(src_path, 'submission_final.csv'), index=False,header=None)
    print("final submission file is at %s"%(src_path+"submission_final.csv"))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_file', type=str,help='path of the output file')
    args = parser.parse_args()
    update_1(args)
    src_path='./predict/'
    base_file='submission_C105_model.csv'
    pt_file2='T161_relu_cnnrnn_res.hhs.w-9.h-1.ratio.0.01.hw-4-dp0.pt'
    get_best_curve(src_path, base_file,pt_file2)
    post()
    
    
    
    
