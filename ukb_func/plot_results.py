import numpy as np
from sklearn.metrics import auc, roc_curve
import os
import fnmatch
import pickle
import matplotlib.pyplot as plt
import argparse
import pandas as pd

def mean_roc_curve(true_labels_list, predicted_probs_list, individual_label, title):
    # plot ROC curve
    
    # Assuming you have a list of true labels and predicted probabilities
    # Initialize variables for mean ROC curve
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []

    # Create a new figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))

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
    ax.set_xlabel('False Positive Rate', weight='bold', size=16)
    ax.set_ylabel('True Positive Rate', weight='bold', size=16)
    ax.set_title(f'{title}', weight='bold', size=16)
    ax.legend(loc='lower right')

    return fig

def folder_recursive_roc(filepath, image_format):
    # iterate through folders and save individual ROC curves

    plot_title = {'age_only': 'Age Only', 'all_demographics': 'All Demographics',
                    'proteins_only': 'All Proteins', 'demographics_and_proteins': 'All Demographics + Proteins'}
                    
    for dirpath, dirnames, filenames in os.walk(filepath):
        if 'test_true_labels_region_0.pkl' in filenames and 'roc_auc' in dirpath:
            print (f'{dirpath}') 
            # In the provided code, `true_labels` is a list that contains the true labels for the data
            # points in a binary classification problem. These true labels are used to calculate the
            # True Positive Rate (TPR) and False Positive Rate (FPR) for generating Receiver Operating
            # Characteristic (ROC) curves.
            # In the provided code, `true_labels` is a list that contains the true labels for the data
            # being used to generate ROC curves. It is used in the context of evaluating the
            # performance of a classification model by comparing the true labels of the data with the
            # predicted probabilities from the model.
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
                
            # train_true = pickle.load(f'{dirpath}/train_true_labels.pkl')
            # train_probas = pickle.load(f'{dirpath}/train_probas.pkl')
            # test_true = pickle.load(open(f'{dirpath}/test_true_labels.pkl', 'rb'))
            # test_probas = pickle.load(open(f'{dirpath}/test_probas.pkl', 'rb'))

            if 'age_only' in dirpath:
                title = plot_title['age_only']
            elif 'all_demographics' in dirpath:
                title = plot_title['all_demographics']
            elif 'modality_only' in dirpath:
                if 'proteomics' in dirpath:
                    title = 'All Proteins'
                elif 'cognitive_test' in dirpath:
                    title = 'All Cognitive Tests'
                elif 'neuroimaging' in dirpath:
                    title = 'All IDPs'
                    
                if 'feature_selection' in dirpath:
                    title = f'Feature Selected {title[4:]}'
                    
            elif 'demographics_and_modality' in dirpath:
                if 'proteomics' in dirpath:
                    title = 'All Demographics + Proteins'
                elif 'cognitive_test' in dirpath:
                    title = 'All Demographics + Cognitive Tests'
                elif 'neuroimaging' in dirpath:
                    title = 'All Demographics + IDPs'
                
                if 'feature_selection' in dirpath:
                    title = f'All Demographics + Feature Selected {title[19:]}'
                    

            fig = mean_roc_curve(true_labels, probas, 'Fold', title)
            fig.savefig(f'{dirpath}/roc_curve.{image_format}')
    
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
    parser = argparse.ArgumentParser(description="Run folder_recursive_roc with arguments")
    parser.add_argument('filepath', type=str, help="filepath")
    parser.add_argument('image_format', type=str, help="image format")
    
    args = parser.parse_args()
    folder_recursive_roc(args.filepath, args.image_format)