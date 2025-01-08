import torch
from torch import nn
from torch.nn import init

class LipNet(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        self.conv1 = nn.Conv3d(in_channels = 1,
                               out_channels = 128,
                               kernel_size = 3,
                               padding = "same")
        self.pool1 = nn.MaxPool3d(kernel_size = (1,2,2))

        self.conv2 = nn.Conv3d(in_channels = 128,
                               out_channels = 256,
                               kernel_size = 3,
                               padding = "same")
        self.pool2 = nn.MaxPool3d(kernel_size = (1,2,2))

        self.conv3 = nn.Conv3d(in_channels = 256,
                               out_channels = 75,
                               kernel_size = 3,
                               padding = "same")
        self.pool3 = nn.MaxPool3d(kernel_size = (1,2,2))

        self.gru1 = nn.GRU(input_size=75*5*7,
                           hidden_size=128,
                           num_layers=1,
                           bidirectional=True)
        self.gru2 = nn.GRU(input_size=256,
                           hidden_size=128,
                           num_layers=1,
                           bidirectional=True)


        self.pred = nn.Linear(in_features=256,
                              out_features=vocab_size)

        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, nonlinearity='sigmoid')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if "weight_ih" in name:
                        init.orthogonal_(param.data)
                    elif "weight_hh" in name:
                        init.orthogonal_(param.data)
                    elif "bias" in name:
                        init.constant_(param.data, 0)

    def forward(self, x):
        # Spatiotemporal CNN
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool3(x)

        # Flatten for RNN
        batch_size, channels, depth, height, width = x.size()
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(batch_size, depth, -1)

        # GRU RNNs
        x, _ = self.gru1(x)
        x = self.dropout(x)
        x, _ = self.gru2(x)
        x = self.dropout(x)

        # Linear Layer
        x = self.pred(x)

        return x