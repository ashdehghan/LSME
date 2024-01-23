
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchsummary import summary
from torch.utils.data import DataLoader

class Autoencoder(nn.Module):

	def __init__(self, emb_dim, encoder_out_size, channel_size):
		super(Autoencoder, self).__init__()
		# Class variables
		self.encoder_out_size = encoder_out_size
		self.channel_size = channel_size
		# Conv Encoder
		self.encoder = nn.Sequential(
			nn.Conv2d(in_channels=1, out_channels=self.channel_size, kernel_size=3, stride=1, padding=1),
			nn.ReLU(inplace=True)
		)
		# Dense Embedding Layer
		self.embedding_input = nn.Flatten()
		self.embedding_layer = nn.Linear(in_features=self.channel_size*self.encoder_out_size*self.encoder_out_size, out_features=emb_dim)
		self.embedding_output = nn.Linear(in_features=emb_dim, out_features=self.channel_size*self.encoder_out_size*self.encoder_out_size)
		# Conv Decoder
		self.decoder = nn.Sequential(
			nn.ConvTranspose2d(in_channels=self.channel_size, out_channels=1, kernel_size=3, stride=1, padding=1, output_padding=0),
			nn.Sigmoid()
		)


	def forward(self, x):
		encoded = self.encoder(x)
		encoded = self.embedding_input(encoded)
		embedding = self.embedding_layer(encoded)
		encoded = self.embedding_output(embedding)
		decoded = self.decoder(encoded.view(-1, self.channel_size, self.encoder_out_size, self.encoder_out_size))
		return [decoded, embedding]


class Conv_Encoder:

	def __init__(self):
		pass

	def encode(self, data, emb_dim):
		
		batch_size = 16
		if data.shape[0] <= 16:
			batch_size = 1

		channel_size = 8
		kernel_size = 3
		padding = 1
		stride = 1

		# Pre-calculate the encoder output size
		input_size = data.shape[1]
		encoder_out_size = int(((input_size + 2*padding - kernel_size)/(stride)) + 1)

		model = Autoencoder(emb_dim, encoder_out_size, channel_size)
		data_tensor = torch.from_numpy(data).float()
		data_loader = DataLoader(data_tensor, batch_size=batch_size, shuffle=False)
		optimizer = torch.optim.Adam(model.parameters())
		num_epochs = 10
		for epoch in tqdm(range(num_epochs), desc="Training Encoder"):
			for batch_idx, data in enumerate(data_loader):
				data = data.view(-1, 1, input_size, input_size)
				optimizer.zero_grad()
				output = model(data)

				loss = nn.MSELoss()(output[0], data)
				loss.backward()
				optimizer.step()
		
		# Compute Embeddings
		data_loader = DataLoader(data_tensor, batch_size=1, shuffle=False)
		embeddings = []
		for batch_idx, data in tqdm(enumerate(data_loader), desc="Building Embeddings"):
			# xx = data.detach().cpu().numpy()
			# plt.imshow(xx[0])
			# plt.xticks([])
			# plt.yticks([])
			# plt.show()

			data = data.view(-1, 1, input_size, input_size)
			pred = model(data)
			
			# yy = pred[0].detach().cpu().numpy()
			# plt.imshow(yy[0][0])
			# plt.xticks([])
			# plt.yticks([])
			# plt.show()

			# input("...")
			
			embedding =list(pred[1].detach().cpu().numpy().flatten())
			embeddings.append(embedding)

		return embeddings

