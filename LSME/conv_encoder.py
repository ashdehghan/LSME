
import torch
import torch.nn as nn
from torchsummary import summary
from torch.utils.data import DataLoader

class Autoencoder(nn.Module):
	def __init__(self):
		super(Autoencoder, self).__init__()
		self.encoder = nn.Sequential(
			nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=2, padding=1),
			nn.ReLU(True),
			nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1),
			nn.ReLU(True),
			nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=2, padding=1),
			nn.ReLU(True),
		)

		self.flatten = nn.Flatten()
		self.fc_middle = nn.Linear(in_features=8*4*4, out_features=32)


		self.decoder = nn.Sequential(
			nn.ConvTranspose2d(in_channels=8, out_channels=16, kernel_size=3, stride=2),  # 8x8x16
			nn.ReLU(True),
			nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=3, stride=2, output_padding=1),  # 15x15x8
			nn.ReLU(True),
			nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=3, stride=2, output_padding=1),  # 30x30x1
			nn.Tanh()
		)


	def forward(self, x):
		encoded = self.encoder(x)
		encoded = self.flatten(encoded)
		encoded = self.fc_middle(encoded)
		decoded = self.decoder(encoded.view(-1, 8, 4, 4))
		return decoded


class Conv_Encoder:

	def __init__(self):
		pass

	def encode(self, data):
		model = Autoencoder()

		data_tensor = torch.from_numpy(data).float()
		data_loader = DataLoader(data_tensor, batch_size=32, shuffle=True)

		optimizer = torch.optim.Adam(model.parameters())


		num_epochs = 10
		for epoch in range(num_epochs):
			for batch_idx, data in enumerate(data_loader):
				data = data.view(-1, 1, 28, 28)
				optimizer.zero_grad()
				output = model(data)
				loss = nn.MSELoss()(output, data)
				loss.backward()
				optimizer.step()

		exit(0)