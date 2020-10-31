from copy import deepcopy
from typing import List, Tuple

import numpy as np

from search import random_action


class State:
    """ゲーム状態"""

    def __init__(self, pieces=None, enemy_pieces=None,
                 past_states=None, depth=0):
        # カップの初期配置
        if (pieces is None) or (enemy_pieces is None):
            # 縦 * 横 + 持ち駒(3種類)
            self.pieces = [[0, 0, 0] for _ in range(9)]
            self.pieces += [[2, 2, 2] for _ in range(3)]
            self.enemy_pieces = [[0, 0, 0] for _ in range(9)]
            self.enemy_pieces += [[2, 2, 2] for _ in range(3)]
            self.past_states = []
        else:
            self.pieces = pieces  # 自分のカップ配置
            self.enemy_pieces = enemy_pieces  # 相手のカップ配置
            self.past_states = past_states  # 過去の状態

        self.depth = depth
        # 現在の状態を登録
        self.past_states.append(str((self.pieces, self.enemy_pieces)))

    def top_cup(self, pos) -> int:
        """一番上に見えているカップ"""
        for i in (2, 1, 0):
            if self.pieces[pos][i]:
                return i  # 自分の駒(0~2)
            elif self.enemy_pieces[pos][i]:
                return i + 3  # 相手のカップ(3~5)
        return -1  # カップなし

    def is_comp(self, x, y, dx, dy, focus_cups) -> bool:
        """3並びかどうか
        x,y (int): 始点座標
        dx,dy (int): 方向
        focus_cups (Tuple[int]): 注目するカップ番号
        """
        for _ in range(3):
            if not (0 <= y < 3) or not (0 <= x < 3) or \
                    (self.top_cup(x + y * 3) not in focus_cups):
                return False
            x, y = x + dx, y + dy
        return True

    def is_win(self) -> bool:
        """勝ち判定"""
        focus_cups = (0, 1, 2)
        # 斜め判定
        if self.is_comp(0, 0, 1, 1, focus_cups) \
                or self.is_comp(0, 2, 1, -1, focus_cups):
            return True
        # 縦横判定
        for i in range(3):
            if self.is_comp(0, i, 1, 0, focus_cups) \
                    or self.is_comp(i, 0, 0, 1, focus_cups):
                return True
        return False

    def is_lose(self) -> bool:
        """負け判定"""
        focus_cups = (3, 4, 5)
        # 斜め判定
        if self.is_comp(0, 0, 1, 1, focus_cups) \
                or self.is_comp(0, 2, 1, -1, focus_cups):
            return True
        # 縦横判定
        for i in range(3):
            if self.is_comp(0, i, 1, 0, focus_cups) \
                    or self.is_comp(i, 0, 0, 1, focus_cups):
                return True
        return False

    def is_draw(self) -> bool:
        """引き分け判定"""
        return self.depth >= 50  # 100手以上

    def is_done(self) -> bool:
        """ゲーム終了判定"""
        return self.is_win() or self.is_lose() or self.is_draw()

    def position_to_action(self, dst_pos, src_pos) -> int:
        """駒の移動先と移動元を行動に変換"""
        return dst_pos * 12 + src_pos

    def action_to_position(self, action) -> Tuple[int, int]:
        """行動を駒の移動先と移動元に変換"""
        return (action // 12, action % 12)

    def is_deployable(self, dst_pos, cup_size) -> bool:
        """配置可能判定"""
        dst_cup = self.top_cup(dst_pos)
        # 移動先になにもない or 移動先のカップの方が小さい
        if (dst_cup == -1) or (dst_cup % 3 < cup_size):
            return True
        return False

    def legal_actions(self) -> List[int]:
        """合法手のリストの取得"""
        actions = []

        # 駒の移動時
        for src_pos in range(9):
            actions += self.legal_actions_pos(src_pos)
        # print(actions)

        # 持ちカップの配置時
        for src_pos in (9, 10, 11):
            if self.pieces[src_pos][0] == 0:  # 持ち駒なし
                continue
            cup_size = src_pos % 3
            for dst_pos in range(9):
                if self.is_deployable(dst_pos, cup_size):
                    act_num = self.position_to_action(dst_pos, src_pos)
                    actions.append(act_num)

        # 過去と同じ状態になる行動は除去
        new_actions = []
        for action in actions:
            state = self.next(action)
            if str((state.pieces, state.enemy_pieces)) not in self.past_states:
                new_actions.append(action)

        # print(new_actions)
        return new_actions

    def legal_actions_pos(self, src_pos) -> List[int]:
        """駒の移動時の合法手のリストの取得"""
        actions = []
        src_cup = self.top_cup(src_pos)

        # 自分のカップではない
        if src_cup not in (0, 1, 2):
            return actions

        for dst_pos in range(9):
            # 移動先になにもない or 移動先のカップの方が小さい
            if self.is_deployable(dst_pos, src_cup):
                act_num = self.position_to_action(dst_pos, src_pos)
                actions.append(act_num)

        return actions

    def next(self, action):
        """次の状態の取得"""
        state = State(deepcopy(self.pieces),
                      deepcopy(self.enemy_pieces),
                      deepcopy(self.past_states),
                      self.depth + 1)

        # 行動を(移動先, 移動元)に変換
        dst_pos, src_pos = self.action_to_position(action)
        # print(action, src_pos, '->', dst_pos)

        # カップの移動
        if src_pos < 9:
            src_cup = self.top_cup(src_pos)
            state.pieces[src_pos][src_cup] = 0
            if state.is_lose():  # ここで終了判定
                # print('負け！')
                pass
            else:
                state.pieces[dst_pos][src_cup] = 1

        # 持ちカップの配置
        else:
            cup_size = src_pos % 3
            for i in range(3):
                state.pieces[src_pos][i] -= 1
            state.pieces[dst_pos][cup_size] = 1

        # カップ交換
        state.pieces, state.enemy_pieces = \
            state.enemy_pieces, state.pieces

        return state

    def is_first_player(self) -> bool:
        """先手かどうか"""
        return self.depth % 2 == 0

    def __str__(self) -> str:
        """文字列表示"""
        is_first = self.is_first_player()
        if is_first:
            pieces_0, pieces_1 = self.pieces, self.enemy_pieces
        else:
            pieces_0, pieces_1 = self.enemy_pieces, self.pieces
        cups_0, cups_1 = ('A1', 'A2', 'A3'), ('B1', 'B2', 'B3')

        text = f'--- {"B" if is_first else "A"}-{(self.depth-1)//2} ---\n'

        # 後手の持ちカップ
        cups = ['--'] * 6
        for i in (9, 10, 11):
            for j in range(pieces_1[i][0]):
                cups[(i % 3) * 2 + j] = cups_1[i % 3]
        text += f'[{" ".join(cups)}]\n'

        # ボード
        for pos in range(9):
            cup = self.top_cup(pos)
            if pos % 3 == 0:
                text += '    '
            if cup == -1:  # カップなし
                text += '--'
            else:  # カップあり
                # if not is_first:
                if not is_first:
                    cup = (cup + 3) % 6
                if cup < 3:
                    text += cups_0[cup]
                else:
                    text += cups_1[cup % 3]
            text += '\n' if pos % 3 == 2 else ' '

        # 先手の持ちカップ
        cups = ['--'] * 6
        for i in (9, 10, 11):
            for j in range(pieces_0[i][0]):
                cups[(i % 3) * 2 + j] = cups_0[i % 3]
        text += f'[{" ".join(cups)}]\n'

        return text

    def pieces_array(self) -> np.ndarray:
        """デュアルネットワークの入力の2次元配列の取得"""
        def pieces_array_of(pieces):
            """プレイヤー毎の処理"""
            table_list = []
            # 盤上
            for i in range(3):
                table = [p[i] for p in pieces[:9]]
                table_list.append(table)
            # 手持ち
            for i in range(3):
                table = [pieces[9 + i][0]] * 9
                table_list.append(table)

            return table_list

        array = np.array([
            *pieces_array_of(self.pieces),
            *pieces_array_of(self.enemy_pieces)
        ], dtype='float32')

        return array


if __name__ == '__main__':
    # 状態の生成
    state = State()

    # ゲーム終了までのループ
    while True:
        # ゲーム終了時
        if state.is_done():
            break

        # 行動の取得
        action = random_action(state)
        dst_pos, src_pos = state.action_to_position(action)
        print(src_pos, '->', dst_pos, end='\n\n')

        # 次の状態の取得
        state = state.next(action)
        print(state)

    array = state.pieces_array()
    print(*array, sep='\n')
