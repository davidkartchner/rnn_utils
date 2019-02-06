""" Matplotlib backend configuration """
import matplotlib
matplotlib.use('agg')  # generate postscript output by default

""" Imports """
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

import numpy as np

from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt

import sys
import os
import argparse
import pickle

from tqdm import trange

from utils import VisitSequenceWithLabelDataset, visit_collate_fn, rnn_epoch
from models import RNN

""" Arguments """
parser = argparse.ArgumentParser()

parser.add_argument('data_path', metavar='DATA_PATH', help="Path to the dataset")

parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, metavar='LR', help='learning rate (default: 1e-3)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run (default: 50)')
parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N', help='mini-batch size for train (default: 128)')
parser.add_argument('-eb', '--eval-batch-size', type=int, default=256, help='mini-batch size for eval (default: 128)')

parser.add_argument('--no-cuda', dest='cuda', action='store_false', help='NOT use cuda')
parser.add_argument('--no-plot', dest='plot', action='store_false', help='no plot')
parser.add_argument('--threads', type=int, default=16, help='number of threads for data loader to use (default: 2)')
parser.add_argument('--seed', type=int, default=0, help='random seed to use. Default=0')

parser.add_argument('--save', default='./', type=str, metavar='SAVE_PATH', help='path to save checkpoints (default: none)')
parser.add_argument('--resume', default='', type=str, metavar='LOAD_PATH', help='path to latest checkpoint (default: none)')

parser.set_defaults(cuda=True, plot=True)


""" Main function """


def main(argv):
	global args
	args = parser.parse_args(argv)
	if args.threads == -1:
		args.threads = torch.multiprocessing.cpu_count() - 1 or 1
	print('===> Configuration')
	print(args)

	cuda = args.cuda
	if cuda:
		if torch.cuda.is_available():
			print('===> {} GPUs are available'.format(torch.cuda.device_count()))
		else:
			raise Exception("No GPU found, please run with --no-cuda")

	# Fix the random seed for reproducibility
	# random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	if cuda:
		torch.cuda.manual_seed(args.seed)

	# Data loading
	print('===> Loading entire datasets')
	with open(args.data_path + 'train.seqs', 'rb') as f:
		train_seqs = pickle.load(f)
	with open(args.data_path + 'train.labels', 'rb') as f:
		train_labels = pickle.load(f)
	with open(args.data_path + 'valid.seqs', 'rb') as f:
		valid_seqs = pickle.load(f)
	with open(args.data_path + 'valid.labels', 'rb') as f:
		valid_labels = pickle.load(f)
	with open(args.data_path + 'test.seqs', 'rb') as f:
		test_seqs = pickle.load(f)
	with open(args.data_path + 'test.labels', 'rb') as f:
		test_labels = pickle.load(f)

	max_code = max(map(lambda p: max(map(lambda v: max(v), p)), train_seqs + valid_seqs + test_seqs))
	num_features = max_code + 1

	print("     ===> Construct train set")
	train_set = VisitSequenceWithLabelDataset(train_seqs, train_labels, num_features, reverse=False)
	print("     ===> Construct validation set")
	valid_set = VisitSequenceWithLabelDataset(valid_seqs, valid_labels, num_features, reverse=False)
	print("     ===> Construct test set")
	test_set = VisitSequenceWithLabelDataset(test_seqs, test_labels, num_features, reverse=False)

	train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True, collate_fn=visit_collate_fn, num_workers=args.threads)
	valid_loader = DataLoader(dataset=valid_set, batch_size=args.eval_batch_size, shuffle=False, collate_fn=visit_collate_fn, num_workers=args.threads)
	test_loader = DataLoader(dataset=test_set, batch_size=args.eval_batch_size, shuffle=False, collate_fn=visit_collate_fn, num_workers=args.threads)
	print('===> Dataset loaded!')

	# Create model
	print('===> Building a Model')

	model = RNN(dim_input=num_features, dim_emb=128, dim_hidden=128)

	if cuda:
		model = model.cuda()
	print(model)
	print('===> Model built!')

	weight_class0 = torch.mean(torch.FloatTensor(train_set.labels))
	weight_class1 = 1.0 - weight_class0
	weight = torch.FloatTensor([weight_class0, weight_class1])

	criterion = nn.CrossEntropyLoss(weight=weight)
	if args.cuda:
		criterion = criterion.cuda()

	optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, nesterov=False, weight_decay=args.weight_decay)
	scheduler = ReduceLROnPlateau(optimizer, 'min')

	best_valid_epoch = 0
	best_valid_loss = sys.float_info.max

	train_losses = []
	valid_losses = []

	if not os.path.exists(args.save):
		os.makedirs(args.save)

	for ei in trange(args.epochs, desc="Epochs"):
		# Train
		_, _, train_loss = rnn_epoch(train_loader, model, criterion=criterion, optimizer=optimizer, train=True)
		train_losses.append(train_loss)

		# Eval
		_, _, valid_loss = rnn_epoch(valid_loader, model, criterion=criterion)
		valid_losses.append(valid_loss)

		scheduler.step(valid_loss)

		is_best = valid_loss < best_valid_loss

		if is_best:
			best_valid_epoch = ei
			best_valid_loss = valid_loss

			# evaluate on the test set
			test_y_true, test_y_pred, test_loss = rnn_epoch(test_loader, model, criterion=criterion)

			if args.cuda:
				test_y_true = test_y_true.cpu()
				test_y_pred = test_y_pred.cpu()

			test_auc = roc_auc_score(test_y_true.numpy(), test_y_pred.numpy()[:, 1], average="weighted")
			test_aupr = average_precision_score(test_y_true.numpy(), test_y_pred.numpy()[:, 1], average="weighted")

			with open(args.save + 'train_result.txt', 'w') as f:
				f.write('Best Validation Epoch: {}\n'.format(ei))
				f.write('Best Validation Loss: {}\n'.format(valid_loss))
				f.write('Train Loss: {}\n'.format(train_loss))
				f.write('Test Loss: {}\n'.format(test_loss))
				f.write('Test AUROC: {}\n'.format(test_auc))
				f.write('Test AUPR: {}\n'.format(test_aupr))

			torch.save(model, args.save + 'best_model.pth')
			torch.save(model.state_dict(), args.save + 'best_model_params.pth')

		# plot
		if args.plot:
			plt.figure(figsize=(12, 9))
			plt.plot(np.arange(len(train_losses)), np.array(train_losses), label='Training Loss')
			plt.plot(np.arange(len(valid_losses)), np.array(valid_losses), label='Validation Loss')
			plt.xlabel('epoch')
			plt.ylabel('Loss')
			plt.legend(loc="best")
			plt.tight_layout()
			plt.savefig(args.save + 'loss_plot.eps', format='eps')
			plt.close()

	print('Best Validation Epoch: {}\n'.format(best_valid_epoch))
	print('Best Validation Loss: {}\n'.format(best_valid_loss))
	print('Train Loss: {}\n'.format(train_loss))
	print('Test Loss: {}\n'.format(test_loss))
	print('Test AUROC: {}\n'.format(test_auc))
	print('Test AUPR: {}\n'.format(test_aupr))


if __name__ == "__main__":
	main(sys.argv[1:])

