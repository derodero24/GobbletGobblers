from evaluate_network import evaluate_network
from train_network import train_network

from self_play import self_play

for i in range(10):
    print('Train', i, '====================')
    self_play()  # セルフプレイ部
    train_network()  # パラメータ更新部
    evaluate_network()  # 新パラメータ評価部
