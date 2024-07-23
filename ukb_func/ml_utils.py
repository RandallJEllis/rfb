import pandas as pd
import numpy as np
import sys
from sklearn.metrics import confusion_matrix, roc_auc_score, \
     precision_recall_fscore_support
from sklearn.metrics import precision_recall_curve, average_precision_score, \
     accuracy_score, balanced_accuracy_score, roc_curve, auc, matthews_corrcoef
from datetime import datetime
import pickle
import matplotlib.pyplot as plt
import os
from utils import save_pickle
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

def mcc_from_conf_mtx(tp, fp, tn, fn):
    return (tp * tn - fp * fn) / np.sqrt( (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) )

def encode_categorical_vars(df, catcols):
    enc = OneHotEncoder(drop='if_binary')
    enc.fit(df.loc[:, catcols])
    categ_enc = pd.DataFrame(enc.transform(df.loc[:, catcols]).toarray(),
                            columns=enc.get_feature_names_out(catcols))
    return categ_enc

def encode_ordinal_vars(df, ordvars):
    enc = OrdinalEncoder()
    enc.fit(df.loc[:, ordvars])
    ord_enc = pd.DataFrame(enc.transform(df.loc[:, ordvars]),
                            columns=enc.get_feature_names_out(ordvars))
    return ord_enc

def pick_threshold(y_true, y_probas, youden=False, beta=1):
    scores = []

    if youden is True:
        # calculate roc curve
        fpr, tpr, thresholds = roc_curve(y_true, y_probas)
        
        for i, t in enumerate(thresholds):
            # youden index = sensitivity + specificity - 1
            # AKA sensitivity + (1 - FPR) - 1 (NOTE: (1-FPR) = TNR)
            # AKA recall_1 + recall_0 - 1
            youdens_j = tpr[i] + (1 - fpr[i]) - 1
            scores.append(youdens_j)

    else:
        # calculate pr-curve
        precision, recall, thresholds = precision_recall_curve(y_true, y_probas)

        # convert to f score
        for i, t in enumerate(thresholds):
            fscore = ((1 + beta**2) * precision[i] * recall[i]) / \
                        ((beta**2 * precision[i]) + recall[i])
            scores.append(fscore)

    ix = np.nanargmax(scores)
    best_threshold = thresholds[ix]

    return best_threshold

def calc_results(metric, y_true, y_probas, youden=False, beta=1, threshold=None):
    auroc = roc_auc_score(y_true, y_probas)
    ap = average_precision_score(y_true, y_probas)

    if metric == 'roc_auc':
        youden = True
        
    return_threshold = False
    if threshold is None:
        threshold = pick_threshold(y_true, y_probas, youden, beta)
        return_threshold = True

    test_pred = (y_probas >= threshold).astype(int)
            
    tn, fp, fn, tp = confusion_matrix(y_true, test_pred).ravel()
    acc = accuracy_score(y_true, test_pred)
    bal_acc = balanced_accuracy_score(y_true,
                                      test_pred)
    prfs = precision_recall_fscore_support(y_true,
                                           test_pred, beta=beta)
    mcc = matthews_corrcoef(y_true, test_pred)

    # print(f'AUROC: {auroc}, AP: {ap}, Fscore: {best_fscore}, Accuracy: {acc}, Bal. Acc.: {bal_acc}, Best threshold: {best_threshold}')
    print(f'AUROC: {np.round(auroc, 4)}, AP: {np.round(ap, 4)}, \nAccuracy: {np.round(acc, 4)}, Bal. Acc.: {np.round(bal_acc, 4)}, \nBest threshold: {np.round(threshold, 4)}')
    print(f'Precision/Recall/Fscore: {prfs}')
    print('\n')
    res =  pd.Series(data=[auroc, ap, threshold, tp, tn, fp, fn, acc, bal_acc,
                           prfs[0][0], prfs[0][1], prfs[1][0], prfs[1][1],
                            prfs[2][0], prfs[2][1], mcc], 
                            index=['auroc', 'avg_prec', 'threshold', 'TP', 'TN', 'FP', 'FN',
                                   'accuracy', 'bal_acc', 'prec_n', 'prec_p', 'recall_n', 'recall_p',
                                    f'f{beta}_n', f'f{beta}_p', 'mcc'])
    if return_threshold == True:
        return res, threshold
    else:
        return res


def save_labels_probas(filepath, train_labels, train_probas, test_labels, test_probas, other_file_info=''):
    save_pickle(f'{filepath}/train_true_labels{other_file_info}.pkl', train_labels)
    save_pickle(f'{filepath}/train_probas{other_file_info}.pkl', train_probas)
    save_pickle(f'{filepath}/test_true_labels{other_file_info}.pkl', test_labels)
    save_pickle(f'{filepath}/test_probas{other_file_info}.pkl', test_probas)

def get_fold_number(fname):
    last_underscore = fname.rfind('_')
    last_period = fname.rfind('.')
    fold = fname[last_underscore+1:last_period]
    return fold

def sort_fold_results(fold_numbers, fold_results):
    # Pair strings with their corresponding numbers
    paired_list = list(zip(fold_numbers, fold_results))

    # Sort the paired list based on the numbers
    sorted_paired_list = sorted(paired_list)

    # Extract the sorted strings
    sorted_results = [fold_result for fold_number, fold_result in sorted_paired_list]
    sorted_results = pd.concat(sorted_results)
    
    return sorted_results
    
def concat_results(filepath):
    train_results = []
    train_fold = []
    test_results = []
    test_fold = []
    
    file_list = os.listdir(filepath)
    
    for fname in file_list:
        if fname[:2] == '._':
            continue
        if '.csv' in fname:
            if 'training_results' in fname and 'region' in fname:
                train_results.append(pd.read_csv(f'{filepath}/{fname}', index_col=0))
                
                fold = get_fold_number(fname)
                train_fold.append(fold)
                
            elif 'test_results' in fname and 'region' in fname:
                test_results.append(pd.read_csv(f'{filepath}/{fname}', index_col=0))
                
                fold = get_fold_number(fname)
                test_fold.append(fold)
    
    
    train_results = sort_fold_results(train_fold, train_results)
    test_results = sort_fold_results(test_fold, test_results)
    
    train_results.to_csv(f'{filepath}/train_results.csv')
    test_results.to_csv(f'{filepath}/test_results.csv')
    
#def iter_concat_results():
    
    
#if __name__ == "__main__":
#    if len(sys.argv) > 1:
#        function_name = sys.argv[1]
#        args = sys.argv[2:]
#        if function_name in globals():
#            globals()[function_name](*args)
#        else:
#            print(f"No function named '{function_name}' found.")
#    else:
#        print("No function name provided.")
        
