import numpy as np
from scipy.sparse import coo_matrix

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from tqdm import tqdm


class AverageMeter(object):
	"""Computes and stores the average and current value"""

	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


""" Custom Dataset """


class VisitSequenceWithLabelDataset(Dataset):
	def __init__(self, seqs, labels, num_features, reverse):
		"""
		Args:
			seqs (list): list of patients (list) of visits (list) of codes (int) that contains visit sequences
			labels (list): list of labels (int)
			num_features (int): number of total features available
			reverse (bool): If true, reverse the order of sequence (for RETAIN)
		"""

		if len(seqs) != len(labels):
			raise ValueError("Seqs and Labels have different lengths")

		self.seqs = []
		# self.labels = []

		for seq, label in zip(seqs, labels):

			if reverse:
				sequence = list(reversed(seq)) #seq[::-1]  #
			else:
				sequence = seq

			row = []
			col = []
			val = []
			for i, visit in enumerate(sequence):
				for code in visit:
					if code < num_features:
						row.append(i)
						col.append(code)
						val.append(1.0)

			self.seqs.append(coo_matrix((np.array(val, dtype=np.float32), (np.array(row), np.array(col))), shape=(len(sequence), num_features)))
		self.labels = labels

	def __len__(self):
		return len(self.labels)

	def __getitem__(self, index):
		return self.seqs[index], self.labels[index]


""" Custom collate_fn for DataLoader"""


# @profile
def visit_collate_fn(batch):
	"""
	DataLoaderIter call - self.collate_fn([self.dataset[i] for i in indices])
	Thus, 'batch' is a list [(seq1, label1), (seq2, label2), ... , (seqN, labelN)]
	where N is minibatch size, seq is a SparseFloatTensor, and label is a LongTensor

	:returns
		seqs
		labels
		lengths
	"""
	batch_seq, batch_label = zip(*batch)

	num_features = batch_seq[0].shape[1]
	seq_lengths = list(map(lambda patient_tensor: patient_tensor.shape[0], batch_seq))
	max_length = max(seq_lengths)

	sorted_indices, sorted_lengths = zip(*sorted(enumerate(seq_lengths), key=lambda x: x[1], reverse=True))
	sorted_padded_seqs = []
	sorted_labels = []

	for i in sorted_indices:
		length = batch_seq[i].shape[0]

		if length < max_length:
			padded = np.concatenate(
				(batch_seq[i].toarray(), np.zeros((max_length - length, num_features), dtype=np.float32)), axis=0)
		else:
			padded = batch_seq[i].toarray()

		sorted_padded_seqs.append(padded)
		sorted_labels.append(batch_label[i])

	seq_tensor = np.stack(sorted_padded_seqs, axis=0)
	label_tensor = torch.LongTensor(sorted_labels)

	return torch.from_numpy(seq_tensor), label_tensor, list(sorted_lengths), list(sorted_indices)


def epoch(data_loader, model, output_activation=None, train=False, criterion=None, optimizer=None):
	if train:
		if not criterion or not optimizer:
			raise AttributeError("criterion and optimizer must be given for training")

	losses = AverageMeter()
	labels = []
	predictions = []

	# switch mode
	if train:
		model.train()
		mode = 'Train'
	else:
		model.eval()
		mode = 'Eval'

	for bi, batch in enumerate(tqdm(data_loader, desc="{} batches".format(mode), leave=False)):

		inputs, targets = batch

		input_var = torch.autograd.Variable(inputs)
		target_var = torch.autograd.Variable(targets)

		if next(model.parameters()).is_cuda:  # returns a boolean
			input_var = input_var.cuda()
			target_var = target_var.cuda()

		# compute output
		output = model(input_var)

		if output_activation:
			output = output_activation(output, dim=1)

		predictions.append(output.data)
		labels.append(targets)

		if criterion:
			loss = criterion(output, target_var)
			assert not np.isnan(loss.data[0]), 'Model diverged with loss = NaN'

			# measure accuracy and record loss
			losses.update(loss.data[0], inputs.size(0))

		if train:
			# compute gradient and do update step
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

	return torch.cat(labels, 0), torch.cat(predictions, 0), losses.avg


def rnn_epoch(loader, model, criterion, output_activation=None, optimizer=None, train=False):
	if train and not optimizer:
		raise AttributeError("Optimizer should be given for training")

	if train:
		model.train()
		mode = 'Train'
	else:
		model.eval()
		mode = 'Eval'

	losses = AverageMeter()
	labels = []
	outputs = []

	for bi, batch in enumerate(tqdm(loader, desc="{} batches".format(mode), leave=False)):
		inputs, targets, lengths, indices = batch

		input_var = torch.autograd.Variable(inputs)
		target_var = torch.autograd.Variable(targets)

		if next(model.parameters()).is_cuda:
			input_var = input_var.cuda()
			target_var = target_var.cuda()

		output = model(input_var, lengths)
		if output_activation:
			output = output_activation(output, dim=1)
		
		print(output)
		loss = criterion(output, target_var)
		print(loss)
		assert not np.isnan(loss.data[0]), 'Model diverged with loss = NaN'

		# compute gradient and do update step
		if train:
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

		sorted_indices, original_indices = zip(*sorted(enumerate(indices), key=lambda x: x[1], reverse=False))
		idx = torch.LongTensor(sorted_indices)

		labels.append(targets.gather(0, idx))
		outputs.append(output.data.cpu().gather(0, torch.stack((idx, idx), dim=1)))
		# record loss
		losses.update(loss.data[0], inputs.size(0))

	return torch.cat(labels, 0), torch.cat(outputs, 0), losses.avg


def batch_patient_tensor_to_list(batch_tensor, lengths, reverse):

	batch_size, max_len, num_features = batch_tensor.size()
	patients_list = []

	for i in range(batch_size):
		patient = []
		for j in range(lengths[i]):
			codes = torch.nonzero(batch_tensor[i][j])
			if codes.is_cuda:
				codes = codes.cpu()
			patient.append(sorted(codes.numpy().flatten().tolist()))

		if reverse:
			patients_list.append(list(reversed(patient)))
		else:
			patients_list.append(patient)

	return patients_list


def aggregate_seqs_list_to_csr(seqs, num_features):
	row = []
	col = []
	data = []

	for i, patient in enumerate(seqs):
		for visit in patient:
			for code in visit:
				if code < num_features:
					row.append(i)
					col.append(code)
					data.append(1)

	aggregated = coo_matrix((np.array(data, dtype=np.float32), (np.array(row), np.array(col))), shape=(len(seqs), num_features))
	aggregated = aggregated.tocsr()

	return aggregated