
import torch
import torch.nn as nn
from torchsummary import summary
from torch.utils.data import DataLoader

class Autoencoder(nn.Module):
	def __init__(self):
		super(Autoencoder, self).__init__()
		self.encoder = nn.Sequential(
			nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Flatten()
		)
		self.decoder = nn.Sequential(
			nn.ConvTranspose2d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1, output_padding=1),
			nn.ReLU(),
			nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=3, stride=2, padding=1, output_padding=1),
			nn.Sigmoid()
		)


	def forward(self, x):
		x = self.encoder(x)
		x = self.decoder(x)
		return x


class Conv_Encoder:

	def __init__(self):
		pass

	def encode(self, data):
		model = Autoencoder()

		data_tensor = torch.from_numpy(data).float()
		data_loader = DataLoader(data_tensor, batch_size=32, shuffle=True)

		# print(summary(model))

		exit(0)