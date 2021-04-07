# -*- coding: UTF-8 -*-

import pandas as pd
import numpy as np
from sympy import *
import os
import matplotlib.pyplot as plt
from math import log
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
class Curve(object):
    def __init__(self,data_file,peak_day=5,peak_ratio=1.2,
                 turning_day=20,turning_ratio=3.0,decay_times=50):
        self.data_path = data_file
        self.peak_day = peak_day
        self.peak_ratio = peak_ratio
        self.turning_day = turning_day
        self.turning_ratio = turning_ratio
        #self.buttom_num = buttom_num
        self.decay_times = decay_times
        pass

    def read_data(self,file):
        df = pd.read_csv(file,header=None,names=['city', 'zoneid', 'date', 'ifc'])
        df = df['ifc']
        return df
        pass
    # f(x) = a*x*x + b*x + c = y
    def poly_fun(self,x,y):
        a,b,c= symbols('a b c')
        res = solve([a*x[0]*x[0] + b*x[0] + c-y[0],
               a*x[1]*x[1] + b*x[1] + c-y[1],
               a*x[2]*x[2] + b*x[2] + c-y[2]],[a,b,c])

        return (res[a],res[b],res[c])
        pass

    # fx = ax + y +c = 0

    def Linear(self,x,y):
        a, b = symbols('a b')
        res = solve([a * x[0] + b - y[0],
                     a * x[1] + b - y[1],
                     ], [a, b])
        return (res[a], res[b])
        pass
    # f(x) = a*log_0.5(x+b)+c = y
    def log_fun(self,x,y):
        a, b, c = symbols('a b c')
        # a * log(x[1] + b, 0.5) + c - y[1],
        res = solve([a * (x[0] + b) + c - y[0],
                     a * (x[1] + b) + c - y[1],
                     a * (x[2] + b) + c - y[2]], [a, b, c])
        return (res[a], res[b], res[c])
        pass

    # f(x) = c/(a*x+b) = y
    def inverse_fun(self,x,y):
        d,e,f = symbols('d e f')
        inv_res = solve([f/(d * x[0] + e) - y[0],
                     f/(d * x[1] + e) - y[1],
                     f/(d * x[2] + e) - y[2]], [d,e,f])

        return inv_res[d], inv_res[e], inv_res[f]
        pass
    #f(x) = a * x * x + b * x + c
    def cal_future_data(self,a,b,c,day_internal,fun_type):
        start = day_internal[0]
        end = day_internal[1]
        res_list = []
        for day in range(start,end):
            if(fun_type ==  "poly"):
                infect_data = round(a * day * day + b * day + c)
                if (infect_data < 0):
                    infect_data = 0
                res_list.append(infect_data)
            if (fun_type == "log"):
                infect_data = round(a * log(day + b, 0.5) + c)
                #a * log(day  + b, 0.5) + c
                if (infect_data < 0):
                    infect_data = 0
                res_list.append(infect_data)
            if (fun_type == "inverse"):
                infect_data = round(c / (a * day + b))
                if(infect_data<0):
                    infect_data = 0
                res_list.append(infect_data)
            if (fun_type == "linear"):
                infect_data =round(a * day + b)
                #a * log(day  + b, 0.5) + c
                if (infect_data < 0):
                    infect_data = 0
                res_list.append(infect_data)
        return res_list
        pass

    def save(self,fitdataCity,format_file,save_file,type):
        if(type!='linear'):
            notfitCity = pd.read_csv('compare/pinpinle/submission_X094.csv',
                          header=None,
                          names=['city', 'zoneid', 'date', 'ifc'])
            notfitCity.loc[notfitCity.city != 'A', 'ifc'] = fitdataCity.values

        else:
            df = pd.read_csv(format_file,
                                 header=None,
                                 names=['city', 'zoneid', 'date', 'ifc'])

            df.loc[(df.city =='F')| (df.city =='B') |(df.city =='D')| (df.city =='E')| (df.city =='G')|(df.city =='H') , 'ifc'] = fitdataCity.values
        df.to_csv(save_file,index=None,header=None)
        pass

    def linear_fit(self,city):
        predict_list = []
        file = self.data_path + "/city_" + city + "/infection.csv"
        df = self.read_data(file)
        group = int(len(df) / 60)

        for j in range(group):
            index = 59 + j * 60
            # 0,5,10
            start_data = df[index]

            change_day = self.peak_day
 
            buttom_day = self.turning_day
            buttom_data = float(1.0*start_data / self.decay_times)
            x1 = [change_day, buttom_day]
            y1 = [start_data, buttom_data]
            a, b = self.Linear(x1, y1)
            for t in range(change_day):
                predict_list.append(start_data)
                pass
            predict_1 = self.cal_future_data(a=a, b=b, c=1, day_internal=x1,fun_type= "linear")
            predict_list += predict_1
            for t in range(buttom_day,30):
                predict_list.append(buttom_data)
                pass

        return predict_list
        pass
    def display_result(self):

        pass

if __name__ == '__main__':

    """To do 下面四个参数需要根据实际情况进行设置和修改"""
    
    data_file = "train_data"
    
    save_file = "./predict/format/submission_Z135.csv"

    format_file = './data/submission_X102.csv'
    
    # BDEFGH
    city_list = "BDEFGH"
    """To do 需要修改的参数范围"""

    city_predict = []
    for city in city_list:
        # peak order:B->D->C->E
        if (city == "B"):
            curve = Curve(data_file=data_file, peak_day=5, peak_ratio=1.2,
                          turning_day=11, turning_ratio=10.0, decay_times=280)

            data = curve.linear_fit(city=city)
            city_predict += data

        if (city == "D"):
            curve = Curve(data_file=data_file, peak_day=5, peak_ratio=1.2,
                          turning_day=11, turning_ratio=10.0, decay_times=280)

            data = curve.linear_fit(city=city)
            city_predict += data
        if (city == "E"):
            curve = Curve(data_file=data_file, peak_day=5, peak_ratio=1.2,
                          turning_day=11, turning_ratio=10.0, decay_times=280)

            data = curve.linear_fit(city=city)
            city_predict += data

        if (city == "F"):
            # 有用的参数：peak_day=3.turning_day=10, decay_times=50
            curve = Curve(data_file=data_file, peak_day=5, peak_ratio=1.2,
                          turning_day=11, turning_ratio=10.0, decay_times=280)

            data = curve.linear_fit(city=city)
            city_predict += data
        if (city == "G"):
            curve = Curve(data_file=data_file, peak_day=5, peak_ratio=1.2,
                          turning_day=11, turning_ratio=10.0, decay_times=280)

            data = curve.linear_fit(city=city)
            city_predict += data
        if (city == "H"):
            curve = Curve(data_file=data_file, peak_day=5, peak_ratio=1.2,
                          turning_day=11, turning_ratio=10.0, decay_times=280)

            data = curve.linear_fit(city=city)
            city_predict += data
    predict_data = pd.DataFrame(city_predict)
    curve.save(predict_data,format_file,save_file,type='linear')

