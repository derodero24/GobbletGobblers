import os
import pickle
from datetime import datetime

import numpy as np
import torch

from dual_network import DN_OUTPUT_SIZE, DualModel, device, load_model
from game import State
from pv_mcts import pv_mcts_scores

SP_GAME_COUNT = 500  # セルフプレイを行うゲーム数(本家は25000)
SP_TEMPERATURE = 1.0  # ボルツマン分布の温度パラメータ


def first_player_value(ended_state) -> float:
    """先手プレイヤーの価値"""
    # 1:先手勝利, -1:先手敗北, 0:引き分け
    if ended_state.is_win():
        return 1.0 if ended_state.is_first_player() else -1.0
    elif ended_state.is_lose():
        return -1.0 if ended_state.is_first_player() else 1.0
    return 0.0


def write_data(history):
    """学習データの保存"""
    now = datetime.now()
    os.makedirs('data/', exist_ok=True)
    path = 'data/{:04}{:02}{:02}{:02}{:02}{:02}.history'.format(
        now.year, now.month, now.day, now.hour, now.minute, now.second)
    with open(path, mode='wb') as f:
        pickle.dump(history, f)


def play(model):
    """1ゲームの実行"""
    state = State()  # 状態の生成
    history = []  # 学習データ

    while True:
        if state.is_done():
            break

        # 合法手の確率分布の取得
        scores = pv_mcts_scores(model, state, SP_TEMPERATURE)

        # 学習データに状態と方策を追加
        policies = [0] * DN_OUTPUT_SIZE
        for action, policy in zip(state.legal_actions(), scores):
            policies[action] = policy
        history.append([state.pieces_array(), policies, None])

        # 行動の取得
        action = np.random.choice(state.legal_actions(), p=scores)

        # 次の状態の取得
        state = state.next(action)

    # 学習データに価値を追加
    value = first_player_value(state)
    for i in range(len(history)):
        history[i][2] = value
        value = -value
    return history


def self_play():
    """自己対戦"""
    history = []  # 学習データ

    # ベストプレイヤーのモデルの読み込み
    model = load_model('./model/best.h5')
    model.eval()

    # 複数回のゲームの実行
    for i in range(SP_GAME_COUNT):
        h = play(model)  # 1ゲームの実行
        history.extend(h)
        print(f'\rSelfPlay {i + 1}/{SP_GAME_COUNT}', end='')
    print()

    write_data(history)  # 学習データの保存


if __name__ == '__main__':
    self_play()
