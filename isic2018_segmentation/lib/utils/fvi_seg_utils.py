import numpy as np
import torch
import matplotlib.pyplot as plt

def run_runtime_seg(model, test_set, exp_name, S):
	X_test = list(test_set)[0][0].cuda()
	model.eval()
	#First forward pass, ignore
	time_list = []
	for _ in range(100 + 1):
		end = model.predict_runtime(X_test, S)
		time_list.append(end)
	time_list = np.array(time_list)[1:]
	time_mean = time_list.mean()
	time_std = time_list.std()
	np.savetxt('output/CAFBNN/text_result/{}_mean_runtime_{}_samples.txt'.format(exp_name, S), [time_mean])
	np.savetxt('output/CAFBNN/text_result/std_runtime_{}_samples.txt'.format(exp_name, S), [time_std])
	print(time_list)
	print('Inference time elapsed (s), mean: {} || std: {}'.format(time_mean, time_std))
	model.train()



def plot_per_image(rgb, ground_truth, pred, pred_entropy, probs, mask, dataset, exp_name, idx, deterministic=False):
    if dataset == 'skin':
        H, W = 256, 256
    im_ratio = float(H/W)
    rgb = rgb.view(3, H, W).permute(1, 2, 0).numpy()
    ground_truth = ground_truth.view(H, W).numpy()

    pred = pred.view(H, W).numpy()
    fig = plt.figure(1,figsize=(12, 2))
    ax1 = plt.subplot(151)
    im1 = ax1.imshow(np.squeeze(np.uint8(rgb*255)))
    ax1.axis('off')
    ax2 = plt.subplot(152)
    im2 = ax2.imshow(np.squeeze(ground_truth), cmap='gray')
    ax2.axis('off')
    ax3 = plt.subplot(153)
    im3 = ax3.imshow(np.squeeze(np.uint8(pred)),  cmap='gray')
    ax3.axis('off')

    ax4 = plt.subplot(154)
    correctness_map = ground_truth-pred
    correctness_map[correctness_map != 0] = 1
    im4 = ax4.imshow(correctness_map, cmap='Greys')
    ax4.axis('off')

    if not deterministic:
        pred_entropy = pred_entropy.view(H, W).numpy()
        probs = probs.numpy()
        mask = mask.view(1, -1).numpy()
        ax5 = plt.subplot(155)
        im5 = ax5.imshow(1-pred_entropy, vmin=0., vmax=np.log(2), cmap='gray')
        ax5.axis('off')
        cb5 = fig.colorbar(im4, ax=ax5, fraction=0.046*im_ratio, pad=0.04)
        ax5.tick_params(labelsize=0)
        ax5.tick_params(size=0)
     
    plt.savefig('output/CAFBNN/fig/{}_{}_results_test_pred_{}.pdf'.format(dataset, exp_name, idx), bbox_inches='tight',pad_inches=0.1, dpi=1000)
    plt.show()
    plt.close()
import h5py

def test(model, test_loader, num_classes, dataset, exp_name, plot_imgs=True, mkdir=False):
    model.eval()
    H, W = 256,256
    if mkdir:
        import os
        new_dir = './results_{}'.format(exp_name)
        os.makedirs(new_dir, exist_ok=True)
        os.chdir(new_dir)
        n_save = len(test_loader)
    else:
        n_save = len(test_loader)
    with torch.no_grad():
        predictions = []
        entropy_uncertainty = []
        test_loss = 0
        test_error = 0
        I_tot = np.zeros(num_classes)
        U_tot = np.zeros(num_classes)

        for idx, (data, target) in enumerate(test_loader):
            data = data.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            print('Processing img {}'.format(idx))
            if idx < n_save and plot_imgs:
                output, entropy, probs  = model.predict(data, return_probs=True)
                mask_ = 2*torch.ones_like(target)
                mask = torch.ne(target, mask_)
                output = output.view(1,H, W)
                pre_label = output.view(1, 1, 256, 256)
                pred_entropy = entropy.view(1, 1, 256, 256)

                pre_label = pre_label.cpu().detach().numpy()
                pred_entropy = pred_entropy.cpu().detach().numpy()
                predictions.append(pre_label)
                entropy_uncertainty.append(pred_entropy)

                plot_per_image(data.cpu(), target.cpu(), output.cpu(), entropy.cpu(), probs.cpu(), mask.cpu(), dataset, exp_name, idx)
            else:
                output = model.predict(data)
            I, U, acc = numpy_metrics(
                output.view(target.size(0), -1).cpu().numpy(),
                target.view(target.size(0), -1).cpu().numpy(),
                n_classes=2,
                void_labels=[2],
            )
            I_tot += I
            U_tot += U
            test_error += 1 - acc


        test_error /= len(test_loader)
        m_jacc = np.mean(I_tot / U_tot)



 
        pred_all = np.concatenate(predictions, 0)
        entropy_all = np.concatenate(entropy_uncertainty, 0)
        file = h5py.File('output/CAFBNN/pred_result/' + 'skin_predict_results.h5', 'w')
        file.create_dataset('pred', data=pred_all)
        file.create_dataset('entropy', data=entropy_all)
        file.close()
        return test_error, m_jacc

from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
def numpy_metrics(y_pred, y_true, n_classes=2, void_labels=[2]):
    """
    Similar to theano_metrics to metrics but instead y_pred and y_true are now numpy arrays
    from: https://github.com/SimJeg/FC-DenseNet/blob/master/metrics.py
    void label is 11 by default
    """

    # Put y_pred and y_true under the same shape

    assert y_pred.shape == y_true.shape, "shapes do not match"

    # We use not_void in case the prediction falls in the void class of the groundtruth
    not_void = ~np.any([y_true == label for label in void_labels], axis=0)

    I = np.zeros(n_classes)
    U = np.zeros(n_classes)

    for i in range(n_classes):
        y_true_i = y_true == i
        y_pred_i = y_pred == i

        I[i] = np.sum(y_true_i & y_pred_i)
        U[i] = np.sum((y_true_i | y_pred_i) & not_void)

    accuracy = np.sum(I) / np.sum(not_void)


    return I, U, accuracy


