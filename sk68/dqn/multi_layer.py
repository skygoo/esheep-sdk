from mxnet import gluon, nd
from mxnet.gluon import nn

class MultiLayer(nn.Block):
    def __init__(self, action_num, **kwargs):
        super(MultiLayer, self).__init__(**kwargs)
        self.l1_1 = nn.Dense(256, activation="relu")
        self.l1_2 = nn.Dense(1)
        self.l2_1 = nn.Dense(256, activation="relu")
        self.l2_2 = nn.Dense(action_num)

    def forward(self, x):
        l1 = self.l1_2(self.l1_1(x)) #state
        l2 = self.l2_2(self.l2_1(x)) #advantage
        return l1 + (l2 - nd.mean(l2, axis=1, keepdims=True))

