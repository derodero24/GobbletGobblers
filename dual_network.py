import os

import torch
import torch.nn.functional as F
from torch import cuda, nn

DN_RESIDUAL_NUM = 10  # 残差ブロックの数（本家は19）
DN_INPUT_SHAPE = (12, 3, 3)  # 入力shape
DN_OUTPUT_SIZE = 108  # 行動数(移動先(9)*移動元(12))

device = torch.device('cuda' if cuda.is_available() else 'cpu')


class Block(nn.Module):
    def __init__(self, channel_in, channel_out):
        super().__init__()
        self.conv1 = nn.Conv2d(channel_in, channel_out,
                               kernel_size=(3, 3),
                               padding=1)
        self.bn1 = nn.BatchNorm2d(channel_out)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(channel_out, channel_out,
                               kernel_size=(3, 3),
                               padding=1)
        self.bn2 = nn.BatchNorm2d(channel_out)
        self.shortcut = nn.Conv2d(channel_in, channel_out,
                                  kernel_size=(3, 3),
                                  padding=1)

        self.relu3 = nn.ReLU()

    def forward(self, x):
        h = self.conv1(x)
        h = self.bn1(h)
        h = self.relu1(h)
        h = self.conv2(h)
        h = self.bn2(h)
        shortcut = self.shortcut(x)
        y = self.relu3(h + shortcut)  # skip connection
        return y


class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:]).view(-1, x.size(1))


class DualModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(DN_INPUT_SHAPE[0], 64,
                               kernel_size=(3, 3),
                               padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.block0 = self._building_block(128, channel_in=64)
        self.block1 = nn.ModuleList([
            self._building_block(128) for _ in range(DN_RESIDUAL_NUM)
        ])
        self.avg_pool = GlobalAvgPool2d()
        # ポリシー出力
        self.out1 = nn.Linear(128, DN_OUTPUT_SIZE)
        # バリュー出力
        self.out2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.block0(x)
        for block in self.block1:
            x = block(x)
        x = self.avg_pool(x)

        policy = self.out1(x)
        policy = torch.softmax(policy, dim=-1)

        value = self.out2(x)
        value = torch.tanh(value)

        return policy, value

    def _building_block(self,
                        channel_out,
                        channel_in=None):
        if channel_in is None:
            channel_in = channel_out
        return Block(channel_in, channel_out)


def load_model(file_path) -> DualModel:
    model = DualModel()
    if os.path.exists(file_path):
        model.load_state_dict(torch.load(file_path))
    return model.to(device)


def save_model(model, file_path) -> None:
    torch.save(model.state_dict(), file_path)


if __name__ == '__main__':
    model = DualModel()
    x = torch.zeros((1, *DN_INPUT_SHAPE))
    policy, value = model(x)
    print(x.shape, policy.shape, value.shape)
    print(float(policy.sum()), float(value))
