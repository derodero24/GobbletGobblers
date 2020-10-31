import os
import pickle
from glob import glob

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch import nn, optim

from dual_network import DN_INPUT_SHAPE, DualModel, device

RN_EPOCHS = 100  # 学習回数


def load_data():
    """学習データの読み込み"""
    history_path = sorted(glob('./data/*.history'))[-1]
    with open(history_path, mode='rb') as f:
        return pickle.load(f)


def train_network():
    """デュアルネットワークの学習"""
    # 学習データの読み込み
    history = load_data()
    x, policies, values = zip(*history)

    # データ変換
    x = np.array(x).reshape((-1, *DN_INPUT_SHAPE))
    x = torch.from_numpy(x).to(device)
    policies = torch.tensor(policies).to(device)
    values = torch.tensor(values).reshape((-1, 1)).to(device)
    print(x.shape, policies.shape, values.shape)
    # print(policies)
    print(values)

    # モデル読み込み
    model = DualModel()
    if os.path.exists('./model/best.h5'):
        model.load_state_dict(torch.load('./model/best.h5'))
    model.to(device)
    # 複数GPU使用宣言
    if str(device) == 'cuda':
        model = nn.DataParallel(model)
        cudnn.benchmark = True

    optimizer = optim.Adam(model.parameters(),
                           lr=0.001, amsgrad=True)
    criterion = nn.CrossEntropyLoss()

    p_pred, v_pred = model(x)
    print(p_pred.shape, v_pred.shape)
    # p_loss = F.nll_loss(p_pred, policies)
    # v_loss = F.nll_loss(v_pred, values)
    # p_loss = criterion(p_pred, policies)
    v_loss = criterion(v_pred, values)
    print(v_loss.shape)
    # p_loss.backward()
    # optimizer.step()

    # 出力
    # print(f'\rTrain {epoch + 1}/{RN_EPOCHS}', end='')

    # # 学習の実行
    # model.fit(x, [y_policies, y_values],
    #           batch_size=128, epochs=RN_EPOCHS,
    #           verbose=0, callbacks=[lr_decay, print_callback])
    # print('')

    # # 最新プレイヤーのモデルの保存
    # torch.save(model.state_dict(), './model/latest.h5')


# 動作確認
if __name__ == '__main__':
    train_network()
