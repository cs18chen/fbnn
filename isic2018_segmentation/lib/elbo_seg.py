"""
	Evidence Lower Bound (ELBO) for Functional Variational Inference (FVI_CV)
	Segmentation experiments. Functional ELBO and masked Cross Entropy loss
"""

import torch
import math
from .elbo_depth import kl_div
from torch.nn.functional import cross_entropy,binary_cross_entropy
import pandas as pd


def fELBO_seg(y_t, f_samples, q_mean, q_cov, prior_mean, prior_cov,  beta, void_class=2.0, print_loss=False):
	S, N, _, _ = f_samples.size()
	el = torch.ones_like(y_t) * void_class
	mask = torch.ne(y_t, el).float()
	N_non_void = mask.sum()
	y_t_masked = (y_t * mask).view(N, -1).unsqueeze(0).expand(S, -1, -1)
	loss = cross_entropy(f_samples.permute(0, 2, 1, 3), y_t_masked.long(), reduction="none").mean(0)
	log_lik = - loss.view(-1)[mask.view(-1).type(torch.bool)].mean()

	kl = kl_div(q_mean, q_cov, prior_mean, prior_cov)
	kl /= N_non_void.float()
	if print_loss:
		logger = 'Pixel-averaged Log Likelihood: {:.3f} || Pixel-averaged KL divergence: {:.3f}'.format(log_lik.item(), kl.item())
		print(logger)
	return log_lik - beta*kl

def masked_cross_entropy_seg(y_t, mean, num_classes=2, void_class=2., print_loss=False):
	N = y_t.size(0)
	el = torch.ones_like(y_t) * void_class
	mask = torch.ne(y_t, el).float()
	y_t_masked = (y_t * mask).view(N, -1)
	loss = cross_entropy(mean[:N,:].contiguous().view(N, num_classes, -1), y_t_masked.long(), reduction="none")
	loss = - loss.view(-1)[mask.view(-1).type(torch.bool)].mean()

	if print_loss:
		print('Cross entropy loss: {:.5f}'.format(loss.item()))

	return loss
