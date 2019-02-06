import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class MLP(nn.Module):
	def __init__(self, input_dim, num_hidden_layers, hidden_dim, dropout=0.5, activation_fn=nn.Tanh):
		super(MLP, self).__init__()
		self.num_hidden_layers = num_hidden_layers

		self.input_to_hidden = nn.Sequential(
			nn.Linear(in_features=input_dim, out_features=hidden_dim),
			activation_fn(),
			nn.Dropout(p=dropout)
		)
		init.xavier_normal(self.input_to_hidden[0].weight)
		self.input_to_hidden[0].bias.data.zero_()

		if num_hidden_layers > 1:
			self.hiddens = nn.ModuleList(
				[nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
							   activation_fn(),
							   nn.Dropout(p=dropout)
							   ) for i in range(num_hidden_layers - 1)]
			)
			for i in range(num_hidden_layers - 1):
				init.xavier_normal(self.hiddens[i][0].weight)
				self.hiddens[i][0].bias.data.zero_()

		self.output_logit = nn.Linear(in_features=hidden_dim, out_features=2)
		init.xavier_normal(self.output_logit.weight)
		self.output_logit.bias.data.zero_()

	def forward(self, x):
		x = self.input_to_hidden(x)
		if self.num_hidden_layers > 1:
			for hidden in self.hiddens:
				x = hidden(x)
		x = self.output_logit(x)
		return x


class RNN(nn.Module):
	def __init__(self, dim_input, dim_emb=256, dropout_emb=0.6, num_layer=2, dim_hidden=256, dropout_output=0.6, dim_output=2, batch_first=True):
		super(RNN, self).__init__()
		self.batch_first = batch_first

		self.embedding = nn.Sequential(
			nn.Linear(dim_input, dim_emb, bias=False),
			nn.Dropout(p=dropout_emb)
		)
		init.xavier_normal(self.embedding[0].weight)

		self.rnn = nn.GRU(input_size=dim_emb, hidden_size=dim_hidden, num_layers=num_layer, batch_first=True)

		self.output = nn.Sequential(
			nn.Dropout(p=dropout_output),
			nn.Linear(in_features=dim_hidden, out_features=dim_output)
		)
		init.xavier_normal(self.output[1].weight)
		self.output[1].bias.data.zero_()

	def forward(self, x, lengths):
		# if self.batch_first:
		# 	batch_size, max_len = x.size()[:2]
		# else:
		# 	max_len, batch_size = x.size()[:2]

		# emb -> batch_size X max_len X dim_emb
		emb = self.embedding(x)

		packed_input = pack_padded_sequence(emb, lengths, batch_first=self.batch_first)

		h_t, h_n = self.rnn(packed_input)

		# unpacked -> batch_size X max_len X dim_hidden
		h_unpacked, recovered_lengths = pad_packed_sequence(h_t, batch_first=self.batch_first)

		idx = (torch.LongTensor(recovered_lengths) - 1).view(-1, 1).expand(h_unpacked.size(0), h_unpacked.size(2)).unsqueeze(1)
		if next(self.parameters()).is_cuda:
			idx = idx.cuda()
		h_unpacked_last = h_unpacked.gather(1, Variable(idx)).squeeze().unsqueeze(1)  # get last hidden output of each sequence

		# without applying non-linearity
		logit = self.output(h_unpacked_last[:, -1, :])

		return logit
