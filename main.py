import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchinfo import summary

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.ToTensor()

data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
loader = DataLoader(data)

images, _ = next(iter(loader))

mean = images.mean().item()
std = images.std().item()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((mean,), (std,))
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=1000)


class PredictiveCodingNet(nn.Module):
    def __init__(self, input_dim, h1_dim, h2_dim, h3_dim, output_dim, lateral_strength=0.1):
        super().__init__()
        self.W1 = nn.Parameter(torch.randn(h1_dim, input_dim) * 0.1)
        self.W2 = nn.Parameter(torch.randn(h2_dim, h1_dim) * 0.1)
        self.W3 = nn.Parameter(torch.randn(h3_dim, h2_dim) * 0.1)
        self.W4 = nn.Parameter(torch.randn(output_dim, h3_dim) * 0.1)
        self.L1 = nn.Parameter(torch.eye(h1_dim) * lateral_strength)
        self.L2 = nn.Parameter(torch.eye(h2_dim) * lateral_strength)
        self.L3 = nn.Parameter(torch.eye(h3_dim) * lateral_strength)
        self.b1 = nn.Parameter(torch.zeros(h1_dim))
        self.b2 = nn.Parameter(torch.zeros(h2_dim))
        self.b3 = nn.Parameter(torch.zeros(h3_dim))
        self.b4 = nn.Parameter(torch.zeros(output_dim))

    def forward(self, x, num_infer_steps=20, lr_state=0.2):
        batch_size = x.size(0)
        s1 = torch.zeros(batch_size, self.W1.shape[0], device=x.device)
        s2 = torch.zeros(batch_size, self.W2.shape[0], device=x.device)
        s3 = torch.zeros(batch_size, self.W3.shape[0], device=x.device)
        s4 = torch.zeros(batch_size, self.W4.shape[0], device=x.device)
        for _ in range(num_infer_steps):
            pred_s1 = F.linear(x, self.W1, self.b1) + torch.matmul(s1, self.L1)
            pred_s2 = F.linear(s1, self.W2, self.b2) + torch.matmul(s2, self.L2)
            pred_s3 = F.linear(s2, self.W3, self.b3) + torch.matmul(s3, self.L3)
            pred_s4 = F.linear(s3, self.W4, self.b4)
            e1 = s1 - pred_s1
            e2 = s2 - pred_s2
            e3 = s3 - pred_s3
            e4 = s4 - pred_s4
            s1 = F.relu(s1 - lr_state * e1)
            s2 = F.relu(s2 - lr_state * e2)
            s3 = F.relu(s3 - lr_state * e3)
            s4 = F.softmax(s4 - lr_state * e4, dim=1)
        return s1, s2, s3, s4

    def predict(self, x):
        _, _, _, s4= self.forward(x)
        return s4
    
