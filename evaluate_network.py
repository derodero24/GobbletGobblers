# ====================
# 新パラメータ評価部
# ====================

from shutil import copy

from dual_network import load_model
# パッケージのインポート
from game import State
from pv_mcts import pv_mcts_action

# パラメータの準備
EN_GAME_COUNT = 10  # 1評価あたりのゲーム数（本家は400）
EN_TEMPERATURE = 1.0  # ボルツマン分布の温度


def first_player_point(ended_state) -> float:
    """先手プレイヤーの価値"""
    # 1:先手勝利, 0.5:引き分け, 0:先手敗北
    if ended_state.is_win():
        return 1 if ended_state.is_first_player() else 0
    elif ended_state.is_lose():
        return 0 if ended_state.is_first_player() else 1
    return 0.5


def play(next_actions) -> float:
    """1ゲームの実行"""
    state = State()  # 状態の生成

    # ゲーム終了までループ
    while True:
        if state.is_done():
            break

        # 行動の取得
        next_action = next_actions[0] if state.is_first_player() else next_actions[1]
        action = next_action(state)

        # 次の状態の取得
        state = state.next(action)

    # 先手プレイヤーのポイントを返す
    return first_player_point(state)


def update_best_player() -> None:
    """ベストプレイヤーの交代"""
    copy('./model/latest.h5', './model/best.h5')
    print('Change BestPlayer')


def evaluate_network() -> bool:
    """ネットワークの評価"""
    # モデル読み込み
    model0 = load_model('./model/latest.h5')
    model1 = load_model('./model/best.h5')

    # PV MCTSで行動選択を行う関数の生成
    next_action0 = pv_mcts_action(model0, EN_TEMPERATURE)
    next_action1 = pv_mcts_action(model1, EN_TEMPERATURE)
    next_actions = (next_action0, next_action1)

    # 複数回の対戦を繰り返す
    total_point = 0
    for i in range(EN_GAME_COUNT):
        if i % 2 == 0:
            total_point += play(next_actions)
        else:
            total_point += 1 - play(next_actions[::-1])
        print(f'\rEvaluate {i + 1}/{EN_GAME_COUNT}', end='')
    print('')

    # 平均ポイントの計算
    average_point = total_point / EN_GAME_COUNT
    print('AveragePoint', average_point)

    # ベストプレイヤーの交代
    if average_point > 0.5:
        update_best_player()
        return True
    else:
        return False


# 動作確認
if __name__ == '__main__':
    evaluate_network()
