import torch.nn as nn
import torch.nn.functional as F
import torch

class MLP(nn.Module):
    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.view(-1, 28*28)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(64*12*12, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, 64*12*12)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class Transformer(nn.Module):
    def __init__(self, num_classes=10, d_model=64, nhead=8, num_encoder_layers=2):
        super(Transformer, self).__init__()
        self.conv = nn.Conv2d(1, d_model, kernel_size=7, stride=1, padding=3)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)
        self.fc = nn.Linear(d_model * 28 * 28, num_classes)

    def forward(self, x):
        x = self.conv(x).view(-1, 28, 28, 64)  # Flatten before Transformer
        x = x.permute(2, 0, 1)  # Shape (seq_len, batch, features)
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)  # (batch, seq_len, features)
        x = x.contiguous().view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x

def get_model(model_name, num_classes):
    if model_name == 'MLP':
        return MLP(num_classes=num_classes)
    elif model_name == 'CNN':
        return CNN(num_classes=num_classes)
    elif model_name == 'Transformer':
        return Transformer(num_classes=num_classes)
    else:
        raise ValueError(f"Model {model_name} not supported yet.")
