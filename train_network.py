import pickle
from glob import glob

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset

from dual_network import DN_INPUT_SHAPE, device, load_model, save_model

RN_EPOCHS = 100
BATCH_SIZE = 128


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

    # モデル読み込み
    model = load_model('./model/best.h5')
    # 複数GPU使用宣言
    if str(device) == 'cuda':
        model = nn.DataParallel(model)
        cudnn.benchmark = True
    model.train()

    optimizer = optim.Adam(model.parameters(),
                           lr=0.001, amsgrad=True)

    loader = DataLoader(TensorDataset(x, policies, values),
                        batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(RN_EPOCHS):
        for x_batch, p_batch, v_batch in loader:

            optimizer.zero_grad()
            # モデル推論
            p_pred, v_pred = model(x_batch)
            # loss計算
            p_loss = -p_pred.log().mul(p_batch).mean()  # cross_entropy_lossみたいな
            v_loss = F.mse_loss(v_batch, values)
            # 更新
            (p_loss + v_loss).backward()
            optimizer.step()

            # 出力
            print(f'\r{epoch + 1}/{RN_EPOCHS} '
                  f'p_loss: {p_loss:.03}, v_loss: {v_loss:.03}', end='')

    # 最新プレイヤーのモデルの保存
    save_model(model, './model/latest.h5')


# 動作確認
if __name__ == '__main__':
    train_network()
