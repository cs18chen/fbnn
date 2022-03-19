import torch
import numpy as np
import argparse
import os
from tqdm import tqdm

from lib.fvi_seg import FVI_seg
from lib.utils.torch_utils import adjust_learning_rate
from lib.utils.fvi_seg_utils import test, numpy_metrics, run_runtime_seg
from dataloader import Skin_loader, dataset_normalized
from torch.utils.data import DataLoader
import pandas as pd

import util as util
import math

parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--batch_size_ft', type=int, default=1)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--randseed', type=int, default=0)
parser.add_argument('--start_epoch', type=int, default=1)
parser.add_argument('--n_epochs', type=int, default=100)
parser.add_argument('--final_epoch', type=int, default=100)
parser.add_argument('--ft_start', type=int, default=100, help='When to start fine-tuning on full-size images')
parser.add_argument('--dataset', type=str, default='skin', help='skin')

parser.add_argument('--beta_type', type=str, default='Blundell', help='Normal, Blundell, Soenderby, Standard')

parser.add_argument('--standard_cross_entropy', type=bool, default=False)
parser.add_argument('--add_cov_diag', type=bool, default=True, help='Add Diagonal component to Q covariance')
parser.add_argument('--f_prior', type=str, default='cnn_gp', help='Type of GP prior: cnn_gp')
parser.add_argument('--match_prior_mean', type=bool, default=False, help='Match Q mean with prior mean')
parser.add_argument('--x_inducing_var', type=float, default=0.1, help='Pixel-wise variance for inducing inputs')
parser.add_argument('--n_inducing', type=int, default=1, help='No. of inducing inputs, <= batch_size')
parser.add_argument('--save_results', type=int, default=100, help='save results every few epochs')#100
parser.add_argument('--base_dir', type=str, default='./', help='directory in which datasets are contained')
parser.add_argument('--training_mode', type=str, default='training_mode', help='store_true')
#parser.add_argument('--test_mode',type=str, default='test_mode', help='store_true')
#parser.add_argument('--test_runtime_mode', default='test_runtime_mode',help='store_true')
args = parser.parse_args()

num_classes= 2

np.random.seed(args.randseed)
torch.manual_seed(args.randseed)
torch.cuda.manual_seed(args.randseed)
H_crop, W_crop, H, W = 256, 256, 256, 256
if args.standard_cross_entropy:
	print('Using standard discrete likelihood')

####################################  Load Data #####################################
folder = './data/' 
tr_data = np.load(folder+'data_train.npy')
te_data = np.load(folder+'data_test.npy')
val_data = np.load(folder+'data_val.npy')

tr_mask = np.load(folder+'mask_train.npy')
te_mask = np.load(folder+'mask_test.npy')
val_mask = np.load(folder+'mask_val.npy')


tr_mask = np.expand_dims(tr_mask.astype(np.uint8), axis=3)
te_mask = np.expand_dims(te_mask.astype(np.uint8), axis=3)
val_mask = np.expand_dims(val_mask.astype(np.uint8), axis=3)


print('ISIC18 Dataset loaded')

tr_data = dataset_normalized(tr_data)
te_data = dataset_normalized(te_data)
val_data = dataset_normalized(val_data)

tr_data = tr_data/255
te_data = te_data/255
val_data = val_data/255

tr_mask = tr_mask / 255.
te_mask = te_mask / 255.
val_mask = val_mask / 255.

if args.dataset == 'skin':
	train_dataset = Skin_loader(tr_data, tr_mask)
	val_dataset = Skin_loader(val_data, val_mask)
	test_dataset = Skin_loader(te_data, te_mask)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size_ft, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)





if args.f_prior == 'cnn_gp':
	exp_name = '{}_{}_segmentation_gp_bnn'.format(args.dataset, args.beta_type)

from lib.elbo_seg import fELBO_seg as fELBO

def get_beta(batch_idx, m, beta_type, epoch, num_epochs):
    if beta_type == "Blundell":
        beta = 2 ** (m - (batch_idx + 1)) / (2 ** m - 1)
    elif beta_type == "Soenderby":
        if epoch is None or num_epochs is None:
            raise ValueError('Soenderby method requires both epoch and num_epochs to be passed.')
        beta = min(epoch / (num_epochs // 4), 1)
    elif beta_type == "Standard":
        beta = 1 / m
    elif beta_type == "Normal":
        beta = 1
    else:
        beta = 0
    return beta

m = math.ceil(len(train_loader) / args.batch_size)
#m1= len(train_loader)
def train(num_epochs, train_loader, FVI):
	FVI.train()

	ft_start_flag = 0
	for s in range(args.start_epoch, args.start_epoch + args.n_epochs):
		train_loss = 0.
		train_error = 0.
		FVI.train()
		if s >= args.ft_start and ft_start_flag == 0:
			from lib.prior.priors import f_prior_BNN
			print('Now fine-tuning on full size images')
		#	train_loader = ft_loader
			if args.f_prior == 'cnn_gp':
				FVI.prior = f_prior_BNN((H, W), device, num_channels_output=num_classes)
			ft_start_flag += 1
		for batch_idx, (X, Y) in enumerate(tqdm(train_loader)):
			x_t = X.to(device)
			y_t = Y.to(device)
			N_t = x_t.size(0)

			optimizer.zero_grad()

			f_samples, q_mean, q_cov, prior_mean, prior_cov = FVI(x_t)
			
			beta = get_beta(batch_idx, m, args.beta_type, s, args.n_epochs)

			loss = - fELBO(y_t, f_samples, q_mean, q_cov, prior_mean, prior_cov, beta, print_loss=True)

			loss.backward()
			train_loss += -loss.item()
			optimizer.step()
			_, _, train_acc_curr = numpy_metrics(FVI.predict(x_t, S=20).data.cpu().view(N_t, -1).numpy(), y_t.view(N_t, -1).data.cpu().numpy())
			train_error += 1 - train_acc_curr
			adjust_learning_rate(args.lr, 0.998, optimizer, s, args.final_epoch)
			del x_t, y_t, f_samples, q_mean, q_cov, prior_mean, prior_cov
		train_loss /= len(train_loader)
		train_error /= len(train_loader)
		print('Epoch: {} || Average Train Error: {:.5f} || Average Train Loss: {:.5f}'.format(s, train_error, train_loss))

		np.savetxt('output/CAFBNN/text_result/{}_{}_epoch_{}_average_train_loss.txt'.format(args.dataset, exp_name, s), [train_loss])
		np.savetxt('output/CAFBNN/text_result/{}_{}_epoch_{}_average_train_error.txt'.format(args.dataset, exp_name, s), [train_error])




		if s % args.save_results == 0 or s == args.final_epoch:
			val_error, val_mIOU = test(FVI, val_loader, num_classes, args.dataset, exp_name, plot_imgs=False)
			print('Epoch: {} || Validation Error: {:.5f} || Validation Mean IOU: {:.5f}'.format(s, val_error, val_mIOU))



			torch.save(FVI.state_dict(), 'output/CAFBNN/model_{}_{}.bin'.format(args.dataset, exp_name))
			torch.save(optimizer.state_dict(), 'output/CAFBNN/optimizer_{}_{}.bin'.format(args.dataset, exp_name))



if __name__ == '__main__':


	device = torch.device("cuda")

	keys = ('device', 'x_inducing_var', 'f_prior', 'n_inducing', 'add_cov_diag', 'standard_cross_entropy')
	values = (device, args.x_inducing_var, args.f_prior, args.n_inducing, args.add_cov_diag, args.standard_cross_entropy)
	fvi_args = dict(zip(keys, values))

	FVI = FVI_seg(x_size=(H_crop, W_crop), num_classes=num_classes, **fvi_args).to(device)
	optimizer = torch.optim.SGD(FVI.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)


	if args.training_mode:
		print('Training FVI_CV segmentation for {} epochs'.format(args.n_epochs))
		train(args.n_epochs, train_loader, FVI)

	if args.test_mode:
		print('Evaluating FVI_CV segmentation on test set')
		model_load_dir = os.path.join(args.base_dir, 'output/CAFBNN/model_{}_{}.bin'.format(args.dataset, exp_name))
		FVI.load_state_dict(torch.load(model_load_dir))
		error, mIOU= test(FVI, test_loader, num_classes, args.dataset, exp_name, plot_imgs=True, mkdir=False)

		np.savetxt('output/CAFBNN/text_result/{}_{}_epoch_{}_test_errorfull.txt'.format(args.dataset, exp_name, -1), [error])
		np.savetxt('output/CAFBNN/text_result/{}_{}_epoch_{}_test_mIOUfull.txt'.format(args.dataset, exp_name, -1), [mIOU])



	if args.test_runtime_mode:
		model_load_dir = os.path.join(args.base_dir, 'output/CAFBNN/model_{}_{}.bin'.format(args.dataset, exp_name))
		FVI.load_state_dict(torch.load(model_load_dir))
		run_runtime_seg(FVI, test_loader, exp_name, 10)
