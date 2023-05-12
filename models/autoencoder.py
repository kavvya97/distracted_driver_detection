import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class AI(torch.nn.Module):
    def __init__(self):
      super().__init__()
      self.encoder = nn.Sequential(
        nn.Conv2d(3, 256, 3, 1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(256, 128, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(128, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(64, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(32, 3, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
      )
        
      self.decoder = nn.Sequential(
        nn.ConvTranspose2d(3, 32, kernel_size=3, stride=2, padding=0, output_padding=0),
        nn.ReLU(),
        nn.ConvTranspose2d(32, 64, kernel_size=3, stride=2, padding=0, output_padding=0),
        nn.ReLU(),
        nn.ConvTranspose2d(64, 128, kernel_size=3, stride=2, padding=0, output_padding=0),
        nn.ReLU(),
        nn.ConvTranspose2d(128, 256, kernel_size=3, stride=2, padding=0, output_padding=0),
        nn.ReLU(),
        nn.ConvTranspose2d(256, 3, kernel_size=3, stride=2, padding=0, output_padding=1),
      )
 
    def forward(self, x):
      encoded = self.encoder(x)
      decoded = self.decoder(encoded)
      return decoded