__author__ = "WoongwonLee@j-marple.com"

import librosa
from torch.autograd import Variable
import torch.utils.data as data_utils
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np

sampling_rate = 16000


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(CausalConv1d, self).__init__()
        # attributes:
        self.kernel_size = kernel_size
        self.dilation = dilation
        # modules:
        self.conv1d = torch.nn.Conv1d(in_channels,
                                      out_channels,
                                      kernel_size,
                                      padding=(kernel_size - 1),
                                      dilation=dilation)

    def forward(self, seq):
        conv1d_out = self.conv1d(seq)
        # remove k-1 values from the end:
        if self.kernel_size > 1:
            return conv1d_out[:, :, 0:-(self.kernel_size - 1)]
        else:
            return conv1d_out[:, :, 0:-(self.kernel_size - 1)]


class ResidualBlock(nn.Module):
    def __init__(self, num_filters, kernel_size, dilation):
        super(ResidualBlock, self).__init__()
        # attributes:
        self.kernel_size = kernel_size
        self.dilation = dilation
        # modules:
        self.causal = CausalConv1d(num_filters, num_filters,
                                   kernel_size, dilation=dilation)
        self.conv1d = nn.Conv1d(num_filters, num_filters, 1)
        self.activation_f = nn.Tanh()
        self.activation_g = nn.Sigmoid()

    def forward(self, input):
        origin_input = input
        conv = self.causal.forward(input)
        tanh = self.activation_f(conv)
        sigmoid = self.activation_g(conv)
        out = tanh * sigmoid
        res = self.conv1d(out)
        skip = self.conv1d(out)
        res += origin_input
        return res, skip


class Wavenet(nn.Module):
    def __init__(self):
        super(Wavenet, self).__init__()
        self.dilation = 1
        self.causal = CausalConv1d(in_channels=1,
                                   out_channels=256,
                                   kernel_size=2,
                                   dilation=2)

        self.residual = ResidualBlock(num_filters=256,
                                      kernel_size=2,
                                      dilation=self.dilation)
        self.relu = nn.ReLU()
        self.conv1d = nn.Conv1d(256, 256, 1, padding=1)
        self.softmax = nn.Softmax()

    def forward(self, input):
        skip_connections = []
        output = self.causal.forward(input)

        for i in range(10):
            print(i)
            output, skip = self.residual.forward(output)
            self.dilation *= 2
            skip_connections.append(skip)

        skip_connections = torch.stack(skip_connections)
        output = torch.sum(skip_connections, dim=0)

        output = self.relu(output)
        output = self.conv1d(output)
        output = output[:, :, :-1]

        output = self.relu(output)
        output = self.conv1d(output)
        output = output[:, :, :-1]

        output = output.view(output)
        output = self.softmax(output)
        return output


def mu_law(data, mu):
    data = data.astype('float64', casting='safe')
    max = np.max(data)
    min = np.min(data)
    data = (data - min) / (max - min/2.) - 1.
    data = np.sign(data) * (np.log(1 + mu * np.abs(data))
                            / np.log(1 + mu))
    return data


def digitize_audio(audio):
    audio = (audio + 1.)/2.
    max_value = np.iinfo('uint8').max
    audio *= max_value
    audio = audio.astype('uint8')
    return audio


def import_audio(audio_path):
    audio, sr = librosa.load(audio_path, sr=sampling_rate)
    audio = mu_law(audio, mu=255)
    inputs = digitize_audio(audio)
    return inputs


if __name__ == "__main__":
    model = Wavenet()
    # import audio and quantize in to 255 integers
    inputs = import_audio('voice.wav')
    print('input lengh is ', len(inputs))
    inputs = np.reshape(inputs, [1, 1, np.shape(inputs)[0]])
    inputs = torch.from_numpy(inputs)
    model.train()
    model.cuda()

    inputs = Variable(inputs).float().cuda()
    outputs = model.forward(inputs)
    print(outputs.cuda())

