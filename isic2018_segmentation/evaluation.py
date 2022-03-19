
import h5py
import numpy as np
import os
import numpy as np
import scipy
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from dataloader import Skin_loader, dataset_normalized
folder = './data/'
tr_data = np.load(folder+'data_train.npy')
te_data = np.load(folder+'data_test.npy')
val_data = np.load(folder+'data_val.npy')

tr_mask = np.load(folder+'mask_train.npy')
te_mask = np.load(folder+'mask_test.npy')
val_mask = np.load(folder+'mask_val.npy')
te_mask0    = np.expand_dims(te_mask, axis=1)

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



h5f = h5py.File('output/CAFBNN/pred_result/'+ 'skin_predict_results.h5', 'r')

y_pred = h5f['pred'][:]

entropy_imgs = h5f['entropy'][:]


print(entropy_imgs.shape)
print(y_pred.shape)
print(te_mask.shape)

y_entropy = entropy_imgs.reshape(
    entropy_imgs.shape[0] * entropy_imgs.shape[1] * entropy_imgs.shape[2] * entropy_imgs.shape[3], 1)


predictions = y_pred

y_scores = predictions.reshape(
    predictions.shape[0] * predictions.shape[1] * predictions.shape[2] * predictions.shape[3], 1)


y_true = te_mask.reshape(te_mask.shape[0] * te_mask.shape[1] * te_mask.shape[2] * te_mask.shape[3], 1)

y_scores = np.where(y_scores > 0.5, 1, 0)
y_true = np.where(y_true > 0.5, 1, 0)

output_folder ='output/CAFBNN/metric/'

# Area under the ROC curve
fpr, tpr, thresholds = roc_curve((y_true), y_scores)
AUC_ROC = roc_auc_score(y_true, y_scores)
print("\nArea under the ROC curve: " + str(AUC_ROC))
roc_curve = plt.figure(figsize=(3,3))
plt.plot(fpr, tpr, '-', label='Area Under the Curve (AUC = %0.3f)' % AUC_ROC)
plt.title('ROC curve',fontsize=6)
plt.xlabel("FPR (False Positive Rate)",fontsize=6)
plt.ylabel("TPR (True Positive Rate)",fontsize=6)
plt.legend(loc='lower center', fontsize = 6)
plt.xticks(size=5)
plt.yticks(size=5)
plt.savefig(output_folder + "ROC.jpg", dpi=1000)
plt.savefig(output_folder + "ROC.pdf", dpi=1000)
plt.show()

# Precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
precision = np.fliplr([precision])[0]
recall = np.fliplr([recall])[0]
AUC_prec_rec = np.trapz(precision, recall)
print("\nArea under Precision-Recall curve: " + str(AUC_prec_rec))
prec_rec_curve = plt.figure(figsize=(3,3))
plt.plot(recall, precision, '-', label='Area Under the Curve (AUC = %0.3f)' % AUC_prec_rec)
plt.title('Precision - Recall curve',fontsize=6)
plt.xlabel("Recall",fontsize=6)
plt.ylabel("Precision",fontsize=6)
plt.legend(loc='lower center', fontsize = 6)
plt.xticks(size=5)
plt.yticks(size=5)
plt.savefig(output_folder + "Precision_recall.jpg", dpi=1000)
plt.savefig(output_folder + "Precision_recall.pdf", dpi=1000)
plt.show()
# Confusion matrix
threshold_confusion = 0.5
print("\nConfusion matrix:  Custom threshold (for positive) of " + str(threshold_confusion))
y_pred = np.empty((y_scores.shape[0]))
for i in range(y_scores.shape[0]):
    if y_scores[i] >= threshold_confusion:
        y_pred[i] = 1
    else:
        y_pred[i] = 0
confusion = confusion_matrix(y_true, y_pred)
print(confusion)
accuracy = 0
if float(np.sum(confusion)) != 0:
    accuracy = float(confusion[0, 0] + confusion[1, 1]) / float(np.sum(confusion))
print("Global Accuracy: " + str(accuracy))
specificity = 0
if float(confusion[0, 0] + confusion[0, 1]) != 0:
    specificity = float(confusion[0, 0]) / float(confusion[0, 0] + confusion[0, 1])
print("Specificity: " + str(specificity))
sensitivity = 0
if float(confusion[1, 1] + confusion[1, 0]) != 0:
    sensitivity = float(confusion[1, 1]) / float(confusion[1, 1] + confusion[1, 0])
print("Sensitivity: " + str(sensitivity))
precision = 0
if float(confusion[1, 1] + confusion[0, 1]) != 0:
    precision = float(confusion[1, 1]) / float(confusion[1, 1] + confusion[0, 1])
print("Precision: " + str(precision))

# Jaccard similarity index
jaccard_index = jaccard_score(y_true, y_pred)
print("\nJaccard similarity score: " + str(jaccard_index))

# F1 score
F1_score = f1_score(y_true, y_pred, labels=None, average='binary', sample_weight=None)
print("\nF1 score (F-measure): " + str(F1_score))

# Save the results
file_perf = open(output_folder + 'performances.txt', 'w')
file_perf.write("Area under the ROC curve: " + str(AUC_ROC)
                + "\nArea under Precision-Recall curve: " + str(AUC_prec_rec)
                + "\nJaccard similarity score: " + str(jaccard_index)
                + "\nF1 score (F-measure): " + str(F1_score)
                + "\n\nConfusion matrix:"
                + str(confusion)
                + "\nACCURACY: " + str(accuracy)
                + "\nSENSITIVITY: " + str(sensitivity)
                + "\nSPECIFICITY: " + str(specificity)
                + "\nPRECISION: " + str(precision)
                )
file_perf.close()




