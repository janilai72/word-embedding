#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   utils.py
@Time    :   2018/11/20 17:44:11
@Author  :   Jian.Lai
@Version :   1.0
@Contact :   jani.lai@aliyun.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Desc    :   None
'''
import pickle


def get_stopwords(stopword_path):
    with open(stopword_path, 'r') as f:
        stopword = [line.strip() for line in f]
    return set(stopword)


def generate_ngram(input_list, n):
    result = []
    for i in range(1, n+1):
        result.extend(zip(*[input_list[j:] for j in range(i)]))
    return result


def load_dictionary(filename):
    """
    加载外部词频记录
    :param filename:
    :return:
    """
    word_freq = {}
    print('------> 加载外部词集')
    with open(filename, 'r') as f:
        for line in f:
            try:
                line_list = line.strip().split(' ')
                # 规定最少词频
                if int(line_list[1]) > 2:
                    word_freq[line_list[0]] = line_list[1]
            except IndexError as e:
                print(line)
                continue
    return word_freq


def save_model(model, filename):
    with open(filename, 'wb') as fw:
        pickle.dump(model, fw)


def load_model(filename):
    with open(filename, 'rb') as fr:
        model = pickle.load(fr)
    return model
