import random


def random_action(state) -> int:
    """ランダムで行動選択"""
    legal_actions = state.legal_actions()
    return legal_actions[random.randint(0, len(legal_actions) - 1)]


def alpha_beta(state, alpha, beta) -> int:
    """アルファベータ法で状態価値計算"""
    if state.is_win():
        # print('win')
        return 1
    if state.is_lose():
        # print('lose')
        return -1
    if state.is_draw():
        # print('draw')
        return 0

    # 合法手の状態価値の計算
    for action in state.legal_actions():
        score = -alpha_beta(state.next(action), -beta, -alpha)
        if score > alpha:
            alpha = score
        # 現ノードのベストスコアが親ノードを超えたら探索終了
        if alpha >= beta:
            return alpha

    # 合法手の状態価値の最大値を返す
    return alpha


def alpha_beta_action(state):
    """アルファベータ法で行動選択"""
    # 合法手の状態価値の計算
    best_action = 0
    alpha = -float('inf')
    str = ['', '']
    for action in state.legal_actions():
        score = -alpha_beta(state.next(action), -float('inf'), -alpha)
        if score > alpha:
            best_action = action
            alpha = score

        str[0] = '{}{:2d},'.format(str[0], action)
        str[1] = '{}{:2d},'.format(str[1], score)
    print('action:', str[0], '\nscore: ', str[1], '\n')

    # 合法手の状態価値の最大値を持つ行動を返す
    return best_action
