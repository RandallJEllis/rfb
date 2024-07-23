import numpy as np
from sklearn.metrics import auc, roc_curve, roc_auc_score, precision_recall_curve, average_precision_score, matthews_corrcoef
import os
import fnmatch
import pickle
import matplotlib.pyplot as plt
import argparse
import pandas as pd
from sklearn.calibration import calibration_curve
import scipy.stats as st 

def concat_labels_and_probas(dirpath):
    
    # In the provided code, `true_labels` is a list that contains the true labels for the data
    # points in a binary classification problem. These true labels are used with predicted 
    # probabilities to calculate the True Positive Rate (TPR) and False Positive Rate (FPR)
    # for generating Receiver Operating Characteristic (ROC) and calibration curves.
    
    true_labels = []
    probas = []

    # load true labels and probas
    for i in range(10):
        # if analyzing feature selection experiments, find the number of features for best performance
        if 'feature_selection' in dirpath:
            df = pd.read_csv(f'{dirpath}/test_results_region_{i}.csv')
            best_idx = df['auroc'].idxmax()
            
        tl = pickle.load(open(f'{dirpath}/test_true_labels_region_{i}.pkl', 'rb'))
        if 'feature_selection' not in dirpath:
            true_labels.append(tl[0])
        else:
            true_labels.append(tl[best_idx])
        
        p = pickle.load(open(f'{dirpath}/test_probas_region_{i}.pkl', 'rb'))    
        if 'feature_selection' not in dirpath:
            probas.append(p[0])
        else:
            probas.append(p[best_idx])
            
    return true_labels, probas

def choose_plot_title(dirpath):
    
    plot_title = {'age_only': 'Age', 'all_demographics': 'Demographics',
                    'proteins_only': 'Proteins', 'demographics_and_proteins': 'Demographics + Proteins'}
    
    if 'age_only' in dirpath:
        title = plot_title['age_only']
    elif 'all_demographics' in dirpath:
        title = plot_title['all_demographics']
    elif 'modality_only' in dirpath:
        if 'proteomics' in dirpath:
            title = 'Proteins'
        elif 'cognitive_test' in dirpath:
            title = 'Cognitive Tests'
        elif 'neuroimaging' in dirpath:
            title = 'IDPs'
            
        if 'feature_selection' in dirpath:
            title = f'FS {title}'
            
    elif 'demographics_and_modality' in dirpath:
        if 'proteomics' in dirpath:
            title = 'Demographics + Proteins'
        elif 'cognitive_test' in dirpath:
            title = 'Demographics + Cognitive Tests'
        elif 'neuroimaging' in dirpath:
            title = 'Demographics + IDPs'
        
        if 'feature_selection' in dirpath:
            title = f'FS Demographics + {title[15:]}'
    
    return title

def mean_roc_curve(true_labels_list, predicted_probs_list): 
    
    # Assuming you have a list of true labels and predicted probabilities
    # Initialize variables for mean ROC curve
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    auc_l = []

    for i, (true_labels, predicted_probs) in enumerate(zip(true_labels_list, predicted_probs_list)):
        rocauc = roc_auc_score(true_labels, predicted_probs)
        auc_l.append(rocauc)

        fpr, tpr, _ = roc_curve(true_labels, predicted_probs)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)

    # Compute mean and standard deviation of TPRs
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    std_tpr = np.std(tprs, axis=0)

    # Plot mean ROC curve
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(auc_l)

    return mean_tpr, std_tpr, mean_auc, std_auc

def mean_pr_curve(true_labels_list, predicted_probs_list):

    precision_list = []
    recall_list = []
    ap_list = []

    for i, (true_labels, predicted_probs) in enumerate(zip(true_labels_list, predicted_probs_list)):
        precision, recall, _ = precision_recall_curve(true_labels, predicted_probs)        
        precision_list.append(np.interp(np.linspace(0, 1, 100), recall[::-1], precision[::-1]))
        recall_list.append(np.linspace(0, 1, 100))

        ap = auc(recall, precision) #average_precision_score(true_labels, predicted_probs) #
        ap_list.append(ap)

    mean_precision = np.mean(precision_list, axis=0)
    std_precision = np.std(precision_list, axis=0)
    mean_recall = np.mean(recall_list, axis=0)
    mean_ap = np.mean(ap_list)
    std_ap = np.std(ap_list)

    return mean_precision, std_precision, mean_recall, mean_ap, std_ap

def cv_roc_curve(true_labels_list, predicted_probs_list, individual_label, title):
    # plot ROC curve
    
    # Assuming you have a list of true labels and predicted probabilities
    # Initialize variables for mean ROC curve
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []

    # Create a new figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))

    for i, (true_labels, predicted_probs) in enumerate(zip(true_labels_list, predicted_probs_list)):
        fpr, tpr, _ = roc_curve(true_labels, predicted_probs)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=1, alpha=0.3, label=f'{individual_label} {i+1} (AUC = {roc_auc:.2f})')

    # Compute mean and standard deviation of TPRs
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    std_tpr = np.std(tprs, axis=0)

    # Plot mean ROC curve
    mean_auc = auc(mean_fpr, mean_tpr)
    ax.plot(mean_fpr, mean_tpr, color='b', label=f'Mean ROC (AUC = {mean_auc:.2f})', lw=2, alpha=0.8)

    # Plot standard deviation
    tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
    tpr_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tpr_lower, tpr_upper, color='grey', alpha=0.2, label=r'$\pm$ 1 std. dev.')

    # Plot settings
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=0.8)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', weight='bold', size=14)
    ax.set_ylabel('True Positive Rate', weight='bold', size=14)
    ax.set_title(f'{title}', weight='bold', size=14)
    ax.legend(loc='lower right')

    return fig

def folder_recursive_cv_roc(filepath, metric, image_format):
    # iterate through folders and save individual ROC curves

    for dirpath, dirnames, filenames in os.walk(filepath):
        if 'test_true_labels_region_0.pkl' in filenames and metric in dirpath:
            print (f'{dirpath}') 
            
            true_labels, probas = concat_labels_and_probas(dirpath)

            title = choose_plot_title(dirpath)

            fig = cv_roc_curve(true_labels, probas, 'Fold', title)
            fig.savefig(f'{dirpath}/roc_curve.{image_format}')
            plt.close()
            
def multi_mean_roc_curve(filepath, metric, image_format, age65_cutoff=False):
    '''
    This function plots multiple mean ROC curves in one plot, from experiments:
    - age only
    - all demographics
    - all modality
    - all demographics + modality
    - feature selection of modality
    - feature selection of demographics + modality

    Input arguments:
    - filepath : str - filepath ending in a modality (proteomics, neuroimaging, cognitive_tests)
    - image_format: str - suffix for generated images (pdf, png, jpg)
    '''

    fig, ax = plt.subplots(figsize=(8, 6))
    
    mean_fpr = np.linspace(0, 1, 100)

    # Plot settings
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=0.8)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', weight='bold', size=14)
    ax.set_ylabel('True Positive Rate', weight='bold', size=14)
    
    colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']
    # for i,expt, in enumerate(['age_only', 'all_demographics', 'modality_only/feature_selection', 'demographics_and_modality/feature_selection', 'modality_only', 'demographics_and_modality', ]):
    for i,expt, in enumerate(['age_only', 'all_demographics', 'modality_only', 'demographics_and_modality', ]):
        if age65_cutoff==False:
            dirpath = f'{filepath}/{expt}/{metric}/'
        else:
            dirpath = f'{filepath}/{expt}/{metric}/agecutoff_65/'

        true_labels, probas = concat_labels_and_probas(dirpath)
        title = choose_plot_title(dirpath)

        mean_tpr, std_tpr, mean_auc, std_auc = mean_roc_curve(true_labels, probas)
        ax.plot(mean_fpr, mean_tpr, color=colors[i], label=f'{title}, AUC: {mean_auc:.2f} $\pm$ {std_auc:.2f}', lw=2, alpha=0.8)

        # Plot standard deviation
        tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
        tpr_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(mean_fpr, tpr_lower, tpr_upper, color=colors[i], alpha=0.2)#, label=r'$\pm$ 1 std. dev.')
    
    ax.legend(loc='lower right')

    fname = f'{filepath}/roc_curve_{metric}_all_expts_mean'
    if age65_cutoff is True:
        fname += '_age65cutoff'
    fname += f'.{image_format}'

    fig.savefig(fname)
    plt.close()

def multi_mean_pr_curve(filepath, image_format, age65_cutoff=False):
    '''
    This function plots multiple mean PR curves in one plot, from experiments:
    - age only
    - all demographics
    - all modality
    - all demographics + modality
    - feature selection of modality
    - feature selection of demographics + modality

    Input arguments:
    - filepath : str - filepath ending in a modality (proteomics, neuroimaging, cognitive_tests)
    - image_format: str - suffix for generated images (pdf, png, jpg)
    '''

    fig, ax = plt.subplots(figsize=(8, 6))
    
    # mean_fpr = np.linspace(0, 1, 100)

    # Plot settings
    # ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=0.8)
    # ax.set_xlim([0.0, 1.0])
    # ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', weight='bold', size=14)
    ax.set_ylabel('Precision', weight='bold', size=14)
    
    colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']
    for i,expt, in enumerate(['age_only', 'all_demographics', 'modality_only/feature_selection', 'demographics_and_modality/feature_selection', 'modality_only', 'demographics_and_modality', ]):
        if age65_cutoff==False:
            dirpath = f'{filepath}/{expt}/{metric}/'
        else:
            dirpath = f'{filepath}/{expt}/{metric}/agecutoff_65/'

        true_labels, probas = concat_labels_and_probas(dirpath)
        title = choose_plot_title(dirpath)

        # mean_tpr, std_tpr, mean_auc, ci = mean_roc_curve(true_labels, probas)
        mean_precision, std_precision, mean_recall, mean_ap, std_ap = mean_pr_curve(true_labels, probas)
        # ax.plot(mean_fpr, mean_tpr, color=colors[i], label=f'{title}, AUC: {mean_auc:.2f} $\pm$ {std_auc:.2f}', lw=2, alpha=0.8)


        ax.plot(mean_recall, mean_precision, color=colors[i], label=f'{title}, AP: {mean_ap:.2f} $\pm$ {std_ap:.2f})', lw=2, alpha=0.8)
        ax.fill_between(mean_recall, mean_precision - std_precision, mean_precision + std_precision, color=colors[i], alpha=0.2)

        # Plot standard deviation
        # tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
        # tpr_lower = np.maximum(mean_tpr - std_tpr, 0)
        # ax.fill_between(mean_fpr, tpr_lower, tpr_upper, color=colors[i], alpha=0.2)#, label=r'$\pm$ 1 std. dev.')
    
    ax.legend(loc='upper right')

    fname = f'{filepath}/pr_curve_all_expts_mean'
    if age65_cutoff is True:
        fname += '_age65cutoff'
    fname += f'.{image_format}'

    fig.savefig(fname)
    plt.close()

def plot_calibration_curve(true_labels, probas, n_bins):

    # # Compute Brier score
    # brier_score = brier_score_loss(true_labels, probas)
    # print(f'Brier Score: {brier_score}')

    # # Compute Log Loss
    # log_loss_score = log_loss(true_labels, probas)
    # print(f'Log Loss: {log_loss_score}')

    prob_true, prob_pred = calibration_curve(true_labels, probas, n_bins=n_bins)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    plt.plot(prob_pred, prob_true, marker='o')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('Predicted probability')
    plt.ylabel('True probability')
    plt.title('Calibration Curve')
    plt.show()

    # # Expected Calibration Error (ECE)
    # ece = np.sum(np.abs(prob_true - prob_pred) * len(y_true) / len(y_prob))
    # print(f'Expected Calibration Error (ECE): {ece}')

    # # Maximum Calibration Error (MCE)
    # mce = np.max(np.abs(prob_true - prob_pred))
    # print(f'Maximum Calibration Error (MCE): {mce}')
    
def folder_recursive_calibration_curve(filepath, metric, image_format):
    # iterate through folders and save individual ROC curves

    for dirpath, dirnames, filenames in os.walk(filepath):
        if 'test_true_labels_region_0.pkl' in filenames and metric in dirpath:
            print (f'{dirpath}') 
            
            true_labels, probas = concat_labels_and_probas(dirpath)

            title = choose_plot_title(dirpath)

            fig = mean_roc_curve(true_labels, probas, 'Fold', title)
            fig.savefig(f'{dirpath}/roc_curve.{image_format}')
            plt.close()
    
def figure_with_subplots(filepath):
    # save figure with subplots

    plot_title = {'age_only': 'Age Only', 'all_demographics': 'All Demographics',
                    'proteins_only': 'All Proteins', 'demographics_and_proteins': 'All Demographics + Proteins'}
    # fig_list = []
    # Create a new figure with 2x2 subplots
    fig, axs = plt.subplots(2, 2, figsize=(20, 16))

    # for experiment in list(plot_title.keys()):
    for i, ax in enumerate(axs.flat):
        test_true = pickle.load(open(f'{dirpath}/test_true_labels.pkl', 'rb'))
        test_probas = pickle.load(open(f'{dirpath}/test_probas.pkl', 'rb'))

        if 'age_only' in dirpath:
            title = plot_title['age_only']
        elif 'all_demographics' in dirpath:
            title = plot_title['all_demographics']
        elif 'proteins_only' in dirpath:
            title = plot_title['proteins_only']
        elif 'demographics_and_proteins' in dirpath:
            title = plot_title['demographics_and_proteins']

        # fig = mean_roc_curve(test_true, test_probas, 'Region fold', title)
        # fig_list.append(fig)
    

        # # Example data for each subplot
        # true_labels_list = [true_labels1, true_labels2, true_labels3, true_labels4]
        # predicted_probs_list = [predicted_probs1, predicted_probs2, predicted_probs3, predicted_probs4]
        # individual_label = "ROC Curve"
        # title = "Mean ROC Curve"

        

        # Call the function for each subplot
        for i, ax in enumerate(axs.flat):
            if i < len(true_labels_list):
                # Get data for the current subplot
                true_labels = true_labels_list[i]
                predicted_probs = predicted_probs_list[i]
                
                # Call the mean_roc_curve function
                mean_roc_curve([true_labels], [predicted_probs], 'Region fold', plot_title[experiment])

        # Adjust layout
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multi_mean_roc_curve with arguments")
    parser.add_argument('filepath', type=str, help="filepath")
    parser.add_argument('metric', type=str, help="metric")
    parser.add_argument('image_format', type=str, help="image format")
    # parser.add_argument('age65_cutoff', type=bool, help="use age cutoff of 65 (True/False)")
    
    args = parser.parse_args()
    multi_mean_roc_curve(args.filepath, args.metric, args.image_format)#, args.age65_cutoff)