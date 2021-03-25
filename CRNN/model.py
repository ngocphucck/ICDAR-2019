from torch import nn
import torchvision


from .dataset import ReceiptDataset


class ConvolutionLayer(nn.Module):
    def __init__(self, in_channels=1, pretrained=False):
        super(ConvolutionLayer, self).__init__()
        self.pretrained = pretrained

        self.conv1_1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, padding=1)  # (64, 64, 280)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)  # (64, 64, 280)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # (64, 32, 140)

        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)  # (128, 32, 140)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)  # (128, 32, 140)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # (128, 16, 70)

        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)  # (256, 16, 70)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)  # (256, 16, 70)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)  # (256, 16, 70)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))  # (256, 8, 70)

        self.conv4_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)  # (512, 8, 70)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.bn4_1 = nn.BatchNorm2d(num_features=512)  # (512, 8, 70)
        self.conv4_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)  # (512, 8, 70)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.bn4_2 = nn.BatchNorm2d(num_features=512)  # (512, 8, 70)
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))  # (512, 4, 70)
        self.conv4_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)  # (512, 4, 70)
        self.relu4_3 = nn.ReLU(inplace=True)

        self.init_weight()

    def forward(self, x):
        out = self.relu1_1(self.conv1_1(x))
        out = self.relu1_2(self.conv1_2(out))
        out = self.pool1(out)

        out = self.relu2_1(self.conv2_1(out))
        out = self.relu2_2(self.conv2_2(out))
        out = self.pool2(out)

        out = self.relu3_1(self.conv3_1(out))
        out = self.relu3_2(self.conv3_2(out))
        out = self.relu3_3(self.conv3_3(out))
        out = self.pool3(out)

        out = self.relu4_1(self.conv4_1(out))
        out = self.bn4_1(out)
        out = self.relu4_2(self.conv4_2(out))
        out = self.bn4_2(out)
        out = self.relu4_3(self.conv4_3(out))
        out = self.pool4(out)

        return out

    def init_weight(self):
        state_dict = self.state_dict()

        pretrained_state_dict = torchvision.models.vgg16(pretrained=self.pretrained).state_dict()

        state_dict['conv1_2.weight'] = pretrained_state_dict['features.2.weight']
        state_dict['conv1_2.bias'] = pretrained_state_dict['features.2.bias']

        state_dict['conv2_1.weight'] = pretrained_state_dict['features.5.weight']
        state_dict['conv2_1.bias'] = pretrained_state_dict['features.5.bias']

        state_dict['conv2_2.weight'] = pretrained_state_dict['features.7.weight']
        state_dict['conv2_2.bias'] = pretrained_state_dict['features.7.bias']

        state_dict['conv3_1.weight'] = pretrained_state_dict['features.10.weight']
        state_dict['conv3_1.bias'] = pretrained_state_dict['features.10.bias']

        state_dict['conv3_2.weight'] = pretrained_state_dict['features.12.weight']
        state_dict['conv3_2.bias'] = pretrained_state_dict['features.12.bias']

        state_dict['conv3_3.weight'] = pretrained_state_dict['features.14.weight']
        state_dict['conv3_3.bias'] = pretrained_state_dict['features.14.bias']

        state_dict['conv4_1.weight'] = pretrained_state_dict['features.17.weight']
        state_dict['conv4_1.bias'] = pretrained_state_dict['features.17.bias']
        state_dict['conv4_2.weight'] = pretrained_state_dict['features.19.weight']
        state_dict['conv4_2.bias'] = pretrained_state_dict['features.19.bias']
        state_dict['conv4_3.weight'] = pretrained_state_dict['features.21.weight']
        state_dict['conv4_3.bias'] = pretrained_state_dict['features.21.bias']

        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0)


class RNNLayer(nn.Module):
    def __init__(self, n_classes, hidden_dim=256):
        super(RNNLayer, self).__init__()
        self.n_classes = n_classes
        self.hidden_dim = hidden_dim
        self.lstm1 = nn.LSTM(input_size=self.hidden_dim, hidden_size=self.hidden_dim,
                             bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=self.hidden_dim, hidden_size=self.hidden_dim,
                             bidirectional=True, batch_first=True)
        self.linear = nn.Linear(self.hidden_dim * 2, self.n_classes)

        self.init_weight()

    def forward(self, x):
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(x)
        out = self.linear(out)

        return out

    def init_weight(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0)


class CRNN(nn.Module):
    def __init__(self, pretrained=False, hidden_size=256, n_classes=73):
        super(CRNN, self).__init__()
        self.convolution_layer = ConvolutionLayer()
        self.rnn_layer = RNNLayer(n_classes=n_classes)
        self.linear = nn.Linear(in_features=2048, out_features=hidden_size)

        self.init_weight()

    def forward(self, x):
        out = self.convolution_layer(x)  # (N, 512, 4, 70)
        out = out.permute(0, 3, 1, 2)  # (N, 70, 512, 4)
        out = out.view(-1, 70, 2048)  # (N, 70, 2048)
        out = self.linear(out)  # (N, 70, 256)
        out = out.permute(1, 0, 2)  # (70, N, 256)
        out = self.rnn_layer(out)  # (70, N, 73)

        return out

    def init_weight(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0)


if __name__ == '__main__':
    dataset = ReceiptDataset(['data/task2/image/1.jpg'], ['abc'])
    model = CRNN()
    out = model(dataset[0][0].unsqueeze(0))
    print(out.size())
