# -*- coding: utf-8 -*-
"""
# @Author : J.Liu
# @Time : 2024/4/25 下午6:23
# @File : test.py
"""
import torch
import data_loader
from seq2seq_atten import Seq2Seq


RESULT_TEST = []
# log_test = open('../log/test_log.txt', 'a')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def testing(model, testing_data, train_X_min, train_X_max, train_y_min, train_y_max):
    total_loss_test = 0
    ARE = 0
    AAE = 0
    flownum = 0
    results = []  # 新增：用于存储预测值和真实值
    with torch.no_grad():
        for batch, (X, y) in enumerate(testing_data):
            model.eval()
            X, y = X.to(device), y.to(device)
            nor_X = (X - train_X_min) / (train_X_max - train_X_min)
            pred = model(nor_X)
            y = y.view(-1,1)
            inv_pred = torch.round(pred * (train_y_max - train_y_min) + train_y_min)
            inv_pred = torch.where(inv_pred <= 0, torch.tensor(1.0, device=device), inv_pred)
            flownum += len(X)
            ARE += torch.sum(abs((inv_pred - y)) / y, dim=0).item()
            AAE += torch.sum(abs(inv_pred - y), dim=0).item()

            # 新增：收集预测值和真实值
            for p, t in zip(pred, y):
                results.append([p.item(), t.item()])
        ARE = ARE / flownum
        AAE = AAE / flownum
    print('{:6f}'.format(ARE))
    print('{:6f}'.format(AAE))


if __name__ == '__main__':
    torch.manual_seed(1)
    _, train_X_min, train_X_max, train_y_min, train_y_max = data_loader.training_data("./test_data/dateset1.csv", batch_size=8192)
    for i in range(5, 11):
        data_path = 'datesetweb.csv'
        _, testing_data = data_loader.testing_data(path=data_path, batch_size=300000)
        model = Seq2Seq().to(device)
        model = torch.load('seq2seq.pth')
        testing(model=model, testing_data=testing_data, train_X_min=train_X_min, train_X_max=train_X_max, train_y_min=train_y_min, train_y_max=train_y_max)
   