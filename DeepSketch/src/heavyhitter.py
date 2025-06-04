# -*- coding: utf-8 -*-
"""
# @Author : J.Liu
# @Time : 2024/4/25 下午6:23
# @File : test.py
"""
import torch
import data_loader
from seq2seq_atten import Seq2Seq
import numpy
import pandas as pd


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULT_TEST = []

def testing(model, testing_data, packet_num, train_X_min, train_X_max, train_y_min, train_y_max):
    real_hitter = {}
    detect_hitter = {}
    cnt = 0
    tp = 0
    
    with torch.no_grad():
        for batch, (data, target) in enumerate(testing_data):
            data, target = data.to(device), target.to(device)
            model.eval()

            mask = target >= 0.0005 * packet_num
            if mask.any():
                selected_data = data[mask.squeeze()]
                selected_target = target[mask.squeeze()]
                for d, t in zip(selected_data, selected_target):
                    cnt += 1
                    flowID = '-'.join(map(str, d.cpu().numpy().flatten().tolist()))
                    real_hitter[flowID] = t.item()

            nor_data = (data - train_X_min) / (train_X_max - train_X_min)
            pred = model(nor_data)
            inv_pred = torch.round(pred * (train_y_max - train_y_min) + train_y_min)

            pred_mask = inv_pred >= 0.0005 * packet_num
            if pred_mask.any():
                selected_data = data[pred_mask.squeeze()]
                selected_pred = inv_pred[pred_mask.squeeze()]
                for d, p in zip(selected_data, selected_pred):
                    flowID = '-'.join(map(str, d.cpu().numpy().flatten().tolist()))
                    detect_hitter[flowID] = p.item()
                    
    # 计算precision和recall
    for key in detect_hitter.keys():
        if key in real_hitter:
            tp += 1
    precision = tp / len(detect_hitter) if len(detect_hitter) > 0 else 0
    recall = tp / cnt if cnt > 0 else 0
    with open('precision_recall.csv', 'a') as f:
        f.write(f"{precision},{recall}\n")
    print('{:6f}'.format(precision))
    print('{:6f}'.format(recall))
    print('-------------------------------')



if __name__ == '__main__':
    torch.manual_seed(1)
    # for i in range(1, 61):
    #     data_path = './test_data/dateset' + str(i) + '.csv'
    #     _, train_X_min, train_X_max, train_y_min, train_y_max = data_loader.training_data("./test_data/dateset1.csv", batch_size=8192)
    #     packet_num, testing_data = data_loader.testing_data(path=data_path, batch_size=30000)
    #     model = Seq2Seq().to(device)
    #     model = torch.load('seq2seq.pth')
    #     testing(model=model, testing_data=testing_data, packet_num=packet_num, train_X_min=train_X_min, train_X_max=train_X_max, train_y_min=train_y_min, train_y_max=train_y_max)

    data_path = 'datesetweb.csv'
    _, train_X_min, train_X_max, train_y_min, train_y_max = data_loader.training_data("./test_data/dateset1.csv", batch_size=8192)
    packet_num, testing_data = data_loader.testing_data(path=data_path, batch_size=30000)
    model = Seq2Seq().to(device)
    model = torch.load('seq2seq.pth')
    testing(model=model, testing_data=testing_data, packet_num=packet_num, train_X_min=train_X_min, train_X_max=train_X_max, train_y_min=train_y_min, train_y_max=train_y_max)
