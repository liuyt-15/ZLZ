# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 02:13:45 2017

@author: dell
"""
import re
import os
import pickle

def load_data(data_path):
    """从文件中读取数据
    Args：文件路径
    Returns：
        data_dict_list: [{'id':'123456',...}, {}, ...]
        min_time: 最小时间戳
        max_time: 最大时间戳
    """
    data_dict_list = list()
    min_time, max_time = float("inf"), 0
    true = 'true'   #使eval()正常运行
    null = 'null'
    with open(data_path, 'rb') as data_file:
        for line in data_file:
            line_dict = eval(line.strip())
            if 'deleted' in line_dict: #过滤已删除的数据
                pass
            else:
                data_dict_list.append(line_dict)
                time = int(line_dict['time'])
                if time < min_time:
                    min_time = time
                if time > max_time:
                    max_time = time
    return data_dict_list,min_time, max_time

def get_one_to_text(start_time, end_time, data_dict_list, input_key=None):
    """得到属性（id或auther）与text的对应关系
    Args：
        start_time：开始的时间戳
        end_time： 结束的时间戳
        data_dict_list: 包含所有属性的数据列表
        input_key: 'id', 'author'等，默认为'id'
    Returns:
         one_to_text: [['102', 'Why target people specifically?'], ....]       
    """
    one_to_text = list()
    if input_key == None:
        input_key = 'id'
    for line_dict in data_dict_list:
        time = int(line_dict['time'])
        if time < start_time or time > end_time:
            pass
        else:
            one_to_text.append([line_dict[input_key], line_dict['text']])
    return one_to_text
                
    
if __name__=='__main__':
    data_path = os.path.join('data', 'comments_000000000000')
    data_dict_list, min_time, max_time = load_data(data_path)
    time_span = max_time - min_time
    one_to_text = get_one_to_text(min_time, min_time + time_span / 5, data_dict_list, input_key=None)    
    pickle.dump(one_to_text, open('id_to_text.pklb', 'wb'), True)
#    id_to_text = pickle.load(open(os.path.join('', 'id_to_text.pklb'), 'rb'))