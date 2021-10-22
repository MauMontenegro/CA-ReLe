import torch.nn as nn
import torch.nn.functional as F


class Simple_CNN(nn.Module):
    def __init__(self, config):
        super(Simple_CNN, self).__init__()

        # Available Actions
        self.output = config.env_actions_n

        # Create Model
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        convh, convw = self.conv2D_size_calc(config.grid_h_size, config.grid_w_size, kernel=8, stride=4, padding=0)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        convh, convw = self.conv2D_size_calc(convh, convw, kernel=4, stride=2, padding=0)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(128)
        convh, convw = self.conv2D_size_calc(convh, convw, kernel=3, stride=1, padding=0)

        # Linear Action Output layer
        self.linear1 = nn.Linear(in_features=convh * convw * 128, out_features=512)
        self.linear2 = nn.Linear(in_features=512, out_features=self.output)

        # Initial Parameters
        self.initial_parameters = self.state_dict()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)

        x = x.view(x.size(0), -1)  # Flatten every batch

        x = self.linear1(x)

        x = F.relu(x)

        x = self.linear2(x)  # No activation on last layer

        return x

    def conv2D_size_calc(self, h_size, w_size, kernel, stride, padding):
        """
        :param padding: Used Padding
        :param stride: Used Stride
        :param kernel: Used Kernel size
        :param h_size: Row Size
        :param w_size: Columns size
        :return: Convolution output Size

        output_size = [(w - k + 2p)/s] + 1
        """
        h_conv = int(((h_size - kernel + 2 * padding) / stride) + 1)
        w_conv = int(((w_size - kernel + 2 * padding) / stride) + 1)

        return h_conv, w_conv

    def reset_parameters(self):
        self.load_state_dict(self.initial_parameters)


# Powers of Dueling Q Network
class Dueling_CNN(nn.Module):
    def __init__(self, config):
        super(Dueling_CNN, self).__init__()

        # Available Actions
        self.output = config.env_actions_n

        # Common CNN Layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        convh, convw = self.conv2D_size_calc(config.grid_h_size, config.grid_w_size, kernel=8, stride=4, padding=0)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        convh, convw = self.conv2D_size_calc(convh, convw, kernel=4, stride=2, padding=0)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(128)
        convh, convw = self.conv2D_size_calc(convh, convw, kernel=3, stride=1, padding=0)

        # Fully Connected Layers
        # Linear Action Output layer
        self.linear1 = nn.Linear(in_features=convh * convw * 128, out_features=1024)
        self.linear2 = nn.Linear(in_features=1024, out_features=512)


        # Advantage of each action Layers
        self.advantage = nn.Linear(512, self.output)
        # State-Value Layers
        self.value = nn.Linear(512, 1)

        # Initial Parameters
        self.initial_parameters = self.state_dict()

    def forward(self, x):
        # Common Layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)

        x = x.view(x.size(0), -1)  # Flatten every batch

        x = self.linear1(x)

        x = F.relu(x)

        x = self.linear2(x)

        # Advantage and Value Layers
        advantage = self.advantage(x)  # Advantage Stream
        value = self.value(x)  # State-Value Stream

        # Maybe return separated values to choose action based only on advantage?
        q_aggregated_values = value + advantage - advantage.mean()

        return q_aggregated_values

    def conv2D_size_calc(self, h_size, w_size, kernel, stride, padding):
        """
        :param padding: Used Padding
        :param stride: Used Stride
        :param kernel: Used Kernel size
        :param h_size: Row Size
        :param w_size: Columns size
        :return: Convolution output Size

        output_size = [(w - k + 2p)/s] + 1
        """
        h_conv = int(((h_size - kernel + 2 * padding) / stride) + 1)
        w_conv = int(((w_size - kernel + 2 * padding) / stride) + 1)

        return h_conv, w_conv

    def reset_parameters(self):
        self.load_state_dict(self.initial_parameters)
