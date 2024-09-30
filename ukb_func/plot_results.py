import numpy as np
from sklearn.metrics import auc, roc_curve, roc_auc_score, precision_recall_curve, average_precision_score, matthews_corrcoef
import os
import pickle
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import matplotlib.patheffects as pe
from matplotlib.font_manager import FontProperties
import argparse
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss, mean_squared_error
import scipy.stats as st 
import sys
sys.path.append('./ukb_func')
from ml_utils import concat_labels_and_probas, probas_to_results
import seaborn as sns
import ptitprince as pt

# Set font properties
plt.rcParams.update({
    'font.size': 12,       # Set font size
    'font.weight': 'bold'  # Set font weight to bold
})

def _get_ages(filepath):
    if 'nacc' in filepath:
        return [None]
    else:
        return [None, 65]

def _choose_plot_title(dirpath):
    """
    Choose the appropriate plot title based on the directory path.

    Args:
    dirpath (str): The directory path containing experiment information.

    Returns:
    str: The chosen plot title.
    """
    
    print(dirpath)
    
    plot_title = {'age_only': 'Age', 'all_demographics': 'Demo'}

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
        elif 'csf' in dirpath:
            title = 'CSF'

        if 'feature_selection' in dirpath:
            title = f'FS {title}'

    elif 'demographics_and_modality' in dirpath:
        if 'proteomics' in dirpath:
            title = 'Demo + Proteins'
        elif 'cognitive_test' in dirpath:
            title = 'Demo + Cognitive Tests'
        elif 'neuroimaging' in dirpath:
            title = 'Demo + IDPs'
        elif 'csf' in dirpath:
            title = 'Demo + CSF'

        if 'feature_selection' in dirpath:
            title = f'FS Demo + {title[7:]}'

    elif 'age_sex_lancet2024' in dirpath:
        title = 'Age + Sex + Lancet'
    elif 'demographics_and_lancet2024' in dirpath:
        title = 'Demo + Lancet'
    elif 'demographics_modality_lancet2024' in dirpath:
        if 'proteomics' in dirpath:
            title = 'Demo + Proteins + Lancet'
        elif 'cognitive_test' in dirpath:
            title = 'Demo + Cognitive Tests + Lancet'
        elif 'neuroimaging' in dirpath:
            title = 'Demo + IDPs + Lancet'
        elif 'csf' in dirpath:
            title = 'Demo + CSF + Lancet'

        if 'feature_selection' in dirpath:
            title = f'FS Demo + {title[7:]}'
    return title


def _save_plot(fig, curve_type, filepath, metric, age_cutoff, image_format):
    """
    Save the plot figure to a file.

    Args:
    fig (matplotlib.figure.Figure): The figure to save.
    curve_type (str): Type of the curve (e.g., 'roc', 'pr').
    filepath (str): Base filepath for saving.
    metric (str): Metric used in the plot.
    age_cutoff (int or None): Age cutoff used, if any.
    image_format (str): Format to save the image (e.g., 'pdf', 'png').
    """
    fname = f'{filepath}/{curve_type}_curve_{metric}_all_expts_mean'
    if age_cutoff is not None:
        fname += f'_age{age_cutoff}cutoff'
    fname += f'.{image_format}'

    fig.savefig(fname, dpi=300)
    plt.close()


def _mean_roc_curve(true_labels_list, predicted_probs_list):
    """
    Calculate the mean ROC curve from multiple ROC curves.

    Args:
    true_labels_list (list): List of true label arrays.
    predicted_probs_list (list): List of predicted probability arrays.

    Returns:
    tuple: Mean TPR, std TPR, mean AUC, std AUC.
    """
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    auc_l = []

    for true_labels, predicted_probs in zip(true_labels_list,
                                            predicted_probs_list):
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

    # Calculate mean AUC and std AUC
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(auc_l)

    return mean_tpr, std_tpr, mean_auc, std_auc


# def cv_roc_curve(true_labels_list, predicted_probs_list, individual_label,
#                  title):
#     """
#     Plot ROC curves for cross-validation results.

#     Args:
#     true_labels_list (list): List of true label arrays.
#     predicted_probs_list (list): List of predicted probability arrays.
#     individual_label (str): Label for individual ROC curves.
#     title (str): Title for the plot.

#     Returns:
#     matplotlib.figure.Figure: The generated figure.
#     """
#     mean_fpr = np.linspace(0, 1, 100)
#     tprs = []

#     fig, ax = plt.subplots(figsize=(8, 6))

#     for i, (true_labels, predicted_probs) in enumerate(zip(
#                                                         true_labels_list,
#                                                         predicted_probs_list)):
#         fpr, tpr, _ = roc_curve(true_labels, predicted_probs)
#         interp_tpr = np.interp(mean_fpr, fpr, tpr)
#         interp_tpr[0] = 0.0
#         tprs.append(interp_tpr)
#         roc_auc = auc(fpr, tpr)
#         ax.plot(fpr, tpr, lw=1, alpha=0.3,
#                 label=f'{individual_label} {i+1} (AUC = {roc_auc:.2f})')

#     # Plot mean ROC curve
#     mean_tpr = np.mean(tprs, axis=0)
#     mean_tpr[-1] = 1.0
#     std_tpr = np.std(tprs, axis=0)
#     mean_auc = auc(mean_fpr, mean_tpr)
#     ax.plot(mean_fpr, mean_tpr, color='b',
#             label=f'Mean ROC (AUC = {mean_auc:.2f})', lw=2, alpha=0.8)

#     # Plot standard deviation
#     tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
#     tpr_lower = np.maximum(mean_tpr - std_tpr, 0)
#     ax.fill_between(mean_fpr, tpr_lower, tpr_upper, color='grey', alpha=0.2,
#                     label=r'$\pm$ 1 std. dev.')

#     # Set plot attributes
#     ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=0.8)
#     ax.set_xlim([0.0, 1.0])
#     ax.set_ylim([0.0, 1.05])
#     ax.set_xlabel('False Positive Rate', weight='bold', size=14)
#     ax.set_ylabel('True Positive Rate', weight='bold', size=14)
#     ax.set_title(f'{title}', weight='bold', size=14)
#     ax.legend(loc='lower right')

#     return fig


# def folder_recursive_cv_roc(filepath, metric, image_format):
#     """
#     Recursively plot ROC curves for all subdirectories in the given filepath.

#     Args:
#     filepath (str): Base filepath to start the recursive search.
#     metric (str): Metric used for evaluation.
#     image_format (str): Format to save the image (e.g., 'pdf', 'png').
#     """
#     for dirpath, dirnames, filenames in os.walk(filepath):
#         if 'test_true_labels_region_0.pkl' in filenames and metric in dirpath:
#             print(f'{dirpath}')

#             true_labels, probas = concat_labels_and_probas(dirpath)
#             title = _choose_plot_title(dirpath)

#             fig = cv_roc_curve(true_labels, probas, 'Fold', title)
#             fig.savefig(f'{dirpath}/roc_curve.{image_format}')
#             plt.close()


def _initialize_roc_plot():
    """
    Initialize a plot for ROC curves.

    Returns:
    tuple: Figure, Axes, and mean FPR array.
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    mean_fpr = np.linspace(0, 1, 100)

    # Set plot attributes
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=0.8)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', weight='bold', size=20)
    ax.set_ylabel('True Positive Rate', weight='bold', size=20)

    return fig, ax, mean_fpr


def multi_mean_roc_curve(filepath, metric, image_format, age65_cutoff=False):
    """
    Plot multiple mean ROC curves for different experiments.

    Args:
    filepath (str): Base filepath for the experiments.
    metric (str): Metric used for evaluation.
    image_format (str): Format to save the image (e.g., 'pdf', 'png').
    age65_cutoff (bool): Whether to use age 65 as a cutoff.
    """
    final_slash = filepath.rfind('/')
    modality = filepath[final_slash+1:]
    colors = ['#ff0000', '#ff7f00', '#ffae00', '#fff500', '#a2ff00', '#00ff29',
              '#00ffce', '#00c9ff', '#2700ff', '#ab00ff']

    ages = _get_ages(filepath)
    for age_cutoff in ages:
        fig, ax, mean_fpr = _initialize_roc_plot()

        experiments = ['age_only', 'age_sex_lancet2024', 'all_demographics',
                       'modality_only', 'demographics_and_lancet2024',
                       'modality_only/feature_selection',
                       'demographics_and_modality',
                       'demographics_and_modality/feature_selection',
                       'demographics_modality_lancet2024',
                       'demographics_modality_lancet2024/feature_selection']
        
        # if 'nacc' in filepath, remove experiments with 'feature_selection'
        if 'nacc' in filepath:
            experiments = [expt for expt in experiments if 'feature_selection' not in expt]

        for i, expt in enumerate(experiments):
            print(f'Age cutoff: {age_cutoff}, Experiment: {expt}')

            if age_cutoff is not None:
                dirpath = f'{filepath}/{expt}/{metric}/agecutoff_{age_cutoff}/'
            else:
                dirpath = f'{filepath}/{expt}/{metric}/'

            true_labels, probas = concat_labels_and_probas(dirpath)
            title = _choose_plot_title(dirpath)

            mean_tpr, std_tpr, mean_auc, std_auc = _mean_roc_curve(true_labels,
                                                                  probas)

            label = f'{title}\nAUC: {mean_auc:.3f} $\pm$ {std_auc:.3f}'
            ax.plot(mean_fpr, mean_tpr, color=colors[i], label=label, lw=2,
                    alpha=0.8)

            # Plot standard deviation
            tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
            tpr_lower = np.maximum(mean_tpr - std_tpr, 0)
            ax.fill_between(mean_fpr, tpr_lower, tpr_upper, color=colors[i],
                            alpha=0.2)

        font_prop = FontProperties(weight='bold', size=10)
        ax.legend(loc='lower right', prop=font_prop)

        plt.tight_layout()
        _save_plot(fig, f'{modality}_roc', filepath, metric, age_cutoff,
                  image_format)


def _initialize_pr_plot():
    """
    Initialize a plot for Precision-Recall curves.

    Returns:
    tuple: Figure and Axes objects.
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_xlabel('Recall', weight='bold', size=20)
    ax.set_ylabel('Precision', weight='bold', size=20)
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.25, 1.25])

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    return fig, ax


def _mean_pr_curve(true_labels_list, predicted_probs_list):
    """
    Calculate the mean Precision-Recall curve from multiple PR curves.

    Args:
    true_labels_list (list): List of true label arrays.
    predicted_probs_list (list): List of predicted probability arrays.

    Returns:
    tuple: Mean precision, std precision, mean recall, mean AP, std AP.
    """
    precision_list = []
    recall_list = []
    ap_list = []

    for true_labels, predicted_probs in zip(true_labels_list,
                                            predicted_probs_list):
        precision, recall, _ = precision_recall_curve(true_labels,
                                                      predicted_probs)
        precision_list.append(np.interp(np.linspace(0, 1, 100), recall[::-1],
                                        precision[::-1]))
        recall_list.append(np.linspace(0, 1, 100))

        ap = average_precision_score(true_labels, predicted_probs)
        ap_list.append(ap)

    mean_precision = np.mean(precision_list, axis=0)
    std_precision = np.std(precision_list, axis=0)
    mean_recall = np.mean(recall_list, axis=0)
    mean_ap = np.mean(ap_list)
    std_ap = np.std(ap_list)

    return mean_precision, std_precision, mean_recall, mean_ap, std_ap


# def plot_pr_curve(true_labels, probas):
#     """
#     Plot a Precision-Recall curve.

#     Args:
#     true_labels (array-like): True labels.
#     probas (array-like): Predicted probabilities.
#     """
#     fig, ax = _initialize_pr_plot()
#     mean_precision, std_precision, mean_recall, mean_ap, std_ap = \
#         mean_pr_curve(true_labels, probas)
#     ax.plot(mean_recall, mean_precision, color='blue',
#             label=f'AP: {mean_ap:.3f} $\pm$ {std_ap:.3f})',
#             lw=2, alpha=0.8)
#     ax.fill_between(mean_recall, mean_precision - std_precision,
#                     mean_precision + std_precision, color='blue', alpha=0.2)

#     ax.legend(loc='upper right')
#     plt.show()


def multi_mean_pr_curve(filepath, metric, image_format, save=True, experiments=None):
    '''
    This function plots multiple mean Precision-Recall curves in one plot for different experiments.

    Args:
    - filepath (str): filepath ending in a modality (proteomics, neuroimaging, cognitive_tests)
    - metric (str): the metric used for evaluation
    - image_format (str): suffix for generated images (pdf, png, jpg)
    - save (bool): whether to save the plot or not (default: True)
    - age_cutoffs (list): list of age cutoffs to use (default: [None, 65])
    - experiments (list): list of experiments to plot (default: None, which uses a predefined list)

    The function iterates through different experiments and age cutoffs, plotting PR curves for each.
    '''
    modality = filepath.split('/')[-1]

    # colors = plt.cm.rainbow(np.linspace(0, 1, 10))
    colors = ['#ff0000', '#ff7f00', '#ffae00', '#fff500', '#a2ff00', '#00ff29',
              '#00ffce', '#00c9ff', '#2700ff', '#ab00ff']

    if experiments is None:
        experiments = ['age_only', 'age_sex_lancet2024', 'all_demographics', 'modality_only', 
                       'demographics_and_lancet2024', 'modality_only/feature_selection', 'demographics_and_modality', 
                       'demographics_and_modality/feature_selection', 'demographics_modality_lancet2024',
                       'demographics_modality_lancet2024/feature_selection']
        
        # if 'nacc' in filepath, remove experiments with 'feature_selection'
        if 'nacc' in filepath:
            experiments = [expt for expt in experiments if 'feature_selection' not in expt]

    ages = _get_ages(filepath)

    for age_cutoff in ages:
        fig, ax = _initialize_pr_plot()

        for i, expt in enumerate(experiments):
            dirpath = f'{filepath}/{expt}/{metric}/' + (f'agecutoff_{age_cutoff}/' if age_cutoff is not None else '')

            true_labels, probas = concat_labels_and_probas(dirpath)
            title = _choose_plot_title(dirpath)

            mean_precision, std_precision, mean_recall, mean_ap, std_ap = _mean_pr_curve(true_labels, probas)

            ax.plot(mean_recall, mean_precision, color=colors[i], label=f'{title}\nAP: {mean_ap:.3f} $\pm$ {std_ap:.3f}', lw=2, alpha=0.8)
            ax.fill_between(mean_recall, mean_precision - std_precision, mean_precision + std_precision, color=colors[i], alpha=0.2)

        ax.legend(loc='upper right', prop=FontProperties(weight='bold', size=10))
        
        plt.tight_layout()

        if save:
            _save_plot(fig, f'{modality}_pr', filepath, metric, age_cutoff, image_format)
        else:
            plt.show()


def _initialize_calibration_curve_plot():
    '''
    Initialize a plot for calibration curves.

    Returns:
    - fig (matplotlib.figure.Figure): The figure object
    - ax (matplotlib.axes.Axes): The axes object

    This function sets up a new figure with appropriate labels, scales, and a diagonal reference line.
    '''
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_xlabel('Predicted Probability', weight='bold', size=20)
    ax.set_ylabel('Observed Fraction of Positives', weight='bold', size=20)
    
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    
    # Set logarithmic scales for both axes
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Add diagonal reference line
    ax.plot([0, 1], [0, 1], linestyle='--', color='red')
    
    return fig, ax 


def multi_calibration_curve(filepath, metric, image_format, n_bins=10):
    '''
    Plot calibration curves for multiple experiments in one plot.

    Args:
    - filepath (str): filepath ending in a modality (proteomics, neuroimaging, cognitive_tests)
    - metric (str): the metric used for evaluation
    - image_format (str): suffix for generated images (pdf, png, jpg)
    - n_bins (int): number of bins for calibration curve (default: 10)

    This function creates calibration curves for different experiments and age cutoffs,
    combining all folds for each experiment.
    '''
    # Extract modality from filepath
    final_slash = filepath.rfind('/')
    modality = filepath[final_slash+1:]
    
    # Define colors for different experiments
    colors = ['#ff0000', '#ff7f00', '#ffae00', '#fff500', '#a2ff00', '#00ff29', '#00ffce', '#00c9ff', '#2700ff', '#ab00ff']
    
    # Iterate through age cutoffs
    ages = _get_ages(filepath)   
    for age_cutoff in ages:
        pred_l = []
        true_l = []
        fig, ax = _initialize_calibration_curve_plot()
        
        experiments = ['age_only', 'age_sex_lancet2024', 'all_demographics',  'modality_only', 
                        'demographics_and_lancet2024', 'modality_only/feature_selection', 'demographics_and_modality', 
                        'demographics_and_modality/feature_selection', 'demographics_modality_lancet2024',
                        'demographics_modality_lancet2024/feature_selection']
        # if 'nacc' in filepath, remove experiments with 'feature_selection'
        if 'nacc' in filepath:
            experiments = [expt for expt in experiments if 'feature_selection' not in expt]
            
            
        # Iterate through different experiments
        for i, expt in enumerate(experiments):
            # Construct directory path based on age cutoff
            if age_cutoff is not None:
                dirpath = f'{filepath}/{expt}/{metric}/agecutoff_{age_cutoff}/'
            else:
                dirpath = f'{filepath}/{expt}/{metric}/'
            
            # Get true labels and probabilities
            true_labels, probas = concat_labels_and_probas(dirpath)
            true_labels = np.concatenate(true_labels)
            probas = np.concatenate(probas)
            
            # Calculate calibration curve
            prob_true, prob_pred = calibration_curve(true_labels, probas, n_bins=n_bins, strategy='quantile')
            title = _choose_plot_title(dirpath)

            # Remove zero values to avoid log(0) issues
            if min(prob_true) == 0:
                non_zero_indices = prob_true > 0
                prob_true = prob_true[non_zero_indices]
                prob_pred = prob_pred[non_zero_indices]

            pred_l.extend(prob_pred)
            true_l.extend(prob_true)
            
            # Plot calibration curve
            ax.plot(prob_pred, prob_true, color=colors[i], marker='s', linewidth=1,
                    label=f'{title}\nMSE: {mean_squared_error(true_labels, probas):.3f}, LL: {log_loss(true_labels, probas):.3f}',
                    path_effects=[pe.Stroke(linewidth=2, foreground='black'), pe.Normal()])
            
        # Set legend properties
        font_prop = FontProperties(weight='bold', size=10)
        ax.legend(loc='lower right', prop=font_prop)
        
        # Adjust plot limits
        ax.set_xlim(min(pred_l)*0.9, max(pred_l)*1.1)
        ax.set_ylim(min(true_l)*0.9, max(true_l)*1.1)

        plt.tight_layout()
        
        # Save the plot
        _save_plot(fig, f'{modality}_calibration', filepath, metric, age_cutoff, image_format)
    

# def folder_recursive_calibration_curve(filepath, metric, image_format):
#     '''
#     Recursively iterate through folders and save individual ROC curves.

#     Args:
#     - filepath (str): The root filepath to start the search
#     - metric (str): The metric used for evaluation
#     - image_format (str): The format to save the images (e.g., 'png', 'pdf')

#     This function walks through the directory structure, finds relevant files,
#     and generates ROC curves for each set of data found.
#     '''
#     for dirpath, dirnames, filenames in os.walk(filepath):
#         if 'test_true_labels_region_0.pkl' in filenames and metric in dirpath:
#             print(f'Processing: {dirpath}')
            
#             # Get true labels and probabilities
#             true_labels, probas = concat_labels_and_probas(dirpath)

#             # Choose plot title
#             title = _choose_plot_title(dirpath)

#             # Generate and save ROC curve
#             fig = _mean_roc_curve(true_labels, probas, 'Fold', title)
#             fig.savefig(f'{dirpath}/roc_curve.{image_format}')
#             plt.close()


def feature_importance_vals(filepath):
    '''
    Calculate and plot feature importance values.

    Args:
    - filepath (str): The filepath to the feature importance data

    Returns:
    - plot_df (pd.DataFrame): DataFrame with mean and std of feature importance

    This function reads feature importance data from multiple files,
    calculates mean and standard deviation, and prepares a DataFrame for plotting.
    '''
    df_l = []
    
    # Read feature importance data from 10 regions
    for i in range(10):
        df = pd.read_parquet(f'{filepath}/feature_importance_region_{i}.parquet')
        df_l.append(df)
        
    # Concatenate all DataFrames
    concatenated_df = pd.concat(df_l, ignore_index=True)

    # Calculate mean and standard deviation of importance for each feature
    result = concatenated_df.groupby('feature')['importance'].agg(['mean', 'std']).reset_index()

    # Rename columns for clarity
    result.columns = ['feature', 'mean_importance', 'std_importance']
    
    # Filter and sort features by mean importance
    plot_df = result[result.mean_importance > 0].sort_values('mean_importance')
    
    # Plotting code is commented out
    # plt.figure(figsize=(8, 6))
    # plt.barh(plot_df.feature, plot_df.mean_importance, xerr=plot_df.std_importance, color='skyblue')
    # plt.savefig(f'{filepath}/feature_importance.pdf')
    
    return plot_df

def _plot_conf_mtx(dff, title, remove_x_ticks=False):
    """
    Plot a confusion matrix with additional metrics.

    Args:
    - dff (pd.DataFrame): DataFrame containing TP, FP, TN, FN columns
    - title (str): Title for the plot
    - remove_x_ticks (bool): If True, removes y-axis ticks and labels

    Returns:
    - matplotlib.figure.Figure: The generated figure object

    This function creates a heatmap visualization of the confusion matrix
    with True Positive Rate (TPR), False Positive Rate (FPR),
    True Negative Rate (TNR), and False Negative Rate (FNR) annotations.
    """

    # Calculate FPR and FNR
    dff['FPR'] = dff.FP / (dff.FP + dff.TN)
    dff['FNR'] = dff.FN / (dff.FN + dff.TP)

    # Calculate confusion matrix values
    TP = np.sum(dff.TP)
    FP = np.sum(dff.FP)
    TN = np.sum(dff.TN)
    FN = np.sum(dff.FN)

    # Calculate rates
    TPR = 100 * (TP / (TP + FN))
    FPR = 100 * (FP / (FP + TN))
    TNR = 100 * (TN / (TN + FP))
    FNR = 100 * (FN / (FN + TP))

    # Create confusion matrix
    confusion_matrix = np.array([[TP, FN],
                                 [FP, TN]])

    # Create labels for annotations
    sub_labels = np.array([['TPR', 'FNR'],
                           ['FPR', 'TNR']])
    sub_label_vals = np.array([[f'{TPR:.2f}', f'{FNR:.2f}'],
                               [f'{FPR:.2f}', f'{TNR:.2f}']])

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))

    # Define colormap values
    colormap_values = np.array([[TPR, FNR],
                                [FPR, TNR]])

    # Plot heatmap
    heatmap = sns.heatmap(colormap_values, annot=False, cmap="Blues", 
                vmin=0, vmax=100,
                xticklabels=['Predicted\nPositive', 'Predicted\nNegative'], 
                yticklabels=['Actual\nPositive', 'Actual\nNegative'])

    # Make colorbar ticks bold
    # cbar = heatmap.collections[0].colorbar
    # cbar.ax.tick_params(labelsize=24, fontweight='bold')

    # Annotate the matrix
    for i in range(2):
        for j in range(2):
            value = confusion_matrix[i, j]
            ax.text(j + 0.5, i + 0.5, f"{int(value)}\n\n{sub_labels[i, j]}\n{sub_label_vals[i,j]}%",
                    ha='center', va='center', color='black', fontsize=30, weight='bold')

    # Set font sizes for tick labels and make them bold
    plt.xticks(fontsize=24, fontweight='bold')
    plt.yticks(fontsize=24, fontweight='bold')

    # Remove x-ticks if specified
    if remove_x_ticks:
        plt.xticks([])
        ax.tick_params(bottom=False)

    # Set title and adjust layout
    plt.title(f'{title}', fontweight='bold', fontsize=24)
    plt.tight_layout()

    return fig


def export_confusion_matrices(filepath):
    """
    Export confusion matrices as PDF files for specified experiments.

    This function generates and saves confusion matrices for a predefined set of experiments.
    It processes the results, creates confusion matrix plots, and saves them as PDF files.

    Args:
        filepath (str): The base filepath where experiment results and output files will be stored.

    The function performs the following steps for each experiment:
    1. Loads test results using probas_to_results function.
    2. Generates a confusion matrix plot using plot_conf_mtx function.
    3. Saves the confusion matrix plot as a PDF file.

    Note:
    - The function assumes the existence of helper functions: probas_to_results, plot_conf_mtx, and choose_plot_title.
    - Experiment paths containing '/' are modified for file naming purposes.
    """

    # List of experiments to process
    expts = [
        'demographics_and_lancet2024',
        'demographics_modality_lancet2024'
    ]
    
    ages = _get_ages(filepath)
    for age_cutoff in ages:
        for expt in expts:
            
            if age_cutoff is not None:
                dirpath = f'{filepath}/{expt}/log_loss/agecutoff_{age_cutoff}/'
            else:
                dirpath = f'{filepath}/{expt}/log_loss/'
        
                # Generate test results for the current experiment
            _, test_results = probas_to_results(f'{dirpath}', youden=True)
            
            # Create confusion matrix plot
            cm = _plot_conf_mtx(test_results, _choose_plot_title(f'{filepath}/{expt}'))

            # Modify experiment name if it contains '/' for file naming
            if '/' in expt:
                expt = expt.replace('/', '_')
            
            # Save the confusion matrix plot as a PDF file
            if age_cutoff is not None:
                fname = f'{filepath}/confusion_matrix_{expt}_agecutoff_{age_cutoff}.pdf'
            else:
                fname = f'{filepath}/confusion_matrix_{expt}.pdf'
            cm.savefig(fname)


def mcc_raincloud(filepath, orient='h'):
    """
    Generate a raincloud plot of Matthews Correlation Coefficient (MCC) values for multiple experiments.

    This function creates a raincloud plot to visualize the distribution of MCC values
    across different experiments and their folds.

    Args:
        filepath (str): Base filepath where experiment results are stored.

    The function performs the following steps:
    1. Define a list of experiments to analyze.
    2. Load test results for each experiment.
    3. Extract MCC values for each experiment.
    4. Prepare data for plotting.
    5. Create and customize the raincloud plot.
    6. Save the plot as a PDF file.

    Note:
    - The function assumes the existence of helper functions: probas_to_results and choose_plot_title.
    - The plot is saved with a filename that includes data modality and age cutoff, which are not defined within this function.
    """
    
    # Define list of experiments to analyze
    expts = ['age_only', 'age_sex_lancet2024', 'all_demographics',  'modality_only', 
            'demographics_and_lancet2024',
            'modality_only/feature_selection',
            'demographics_and_modality',
            'demographics_and_modality/feature_selection',
            'demographics_modality_lancet2024',
            'demographics_modality_lancet2024/feature_selection'
            ]
    # if 'nacc' in filepath, remove experiments with 'feature_selection'
    if 'nacc' in filepath:
        expts = [expt for expt in expts if 'feature_selection' not in expt]
        
    ages = _get_ages(filepath)
        
    for orient, fig_dimensions in zip(['v', 'h'], [(12, 8), (6, 8)]):
        if orient == 'h':
            continue
        for age_cutoff in ages:
            res_l = []
            titles_l = []
            for expt in expts:
                if age_cutoff is not None:
                    dirpath = f'{filepath}/{expt}/log_loss/agecutoff_{age_cutoff}/'
                else:
                    dirpath = f'{filepath}/{expt}/log_loss/'
                # Load test results for each experiment
                _, test_results = probas_to_results(f'{dirpath}', youden=True)
                res_l.append(test_results)
                titles_l.append(_choose_plot_title(f'{filepath}/{expt}'))
        
            # Extract MCC values for each experiment
            y = [i.mcc for i in res_l]

            # Prepare the data for plotting
            data = []
            for i, category in enumerate(titles_l):
                
                # insert a newline character after every plus sign
                if orient == 'v':
                    category = category.replace(' + ', '\n+\n')
                
                for value in y[i]:
                    data.append([category, value])
            data = pd.DataFrame(data, columns=["", "Matthews Correlation Coefficient"])

            # Define color palette for the plot
            colors = ['#ff0000', '#ff7f00', '#ffae00', '#fff500', '#a2ff00', '#00ff29', '#00ffce', '#00c9ff', '#2700ff', '#ab00ff']    

            # Create figure and axis objects
            fig, ax = plt.subplots(figsize=fig_dimensions)
        
            # Define the line width properties
            # line_width_props = {
            #     'whiskerprops': {'linewidth': 2},
            #     'capprops': {'linewidth': 2},
            #     'boxprops': {'linewidth': 2},
            # }

            # Generate the raincloud plot
            pt.RainCloud(x="", y="Matthews Correlation Coefficient", data=data, palette=colors, 
                        bw=.2, 
                        ax=ax, orient=orient,
                        box_linewidth=1,  # Modify the boxplot linewidth
                        offset=.2,  # Adjust the cloud position
                        move=.2,  # Adjust the rain position
                        width_viol=.6,  # Reduce violin width
                        )
            
            # Adjust the plot margins to prevent cutting off the rightmost rain only for vertical orientation
            if orient == 'v':
                plt.subplots_adjust(right=0.95)  # Increase right margin for vertical orientation

            # Adjust the y-axis limits to show the full peak
            if orient == 'v':
                y_min, y_max = ax.get_ylim()
                ax.set_ylim(y_min, y_max * 1.1)  # Increase the upper limit by 10%
            
            # Customize plot appearance
            ax.set_ylabel('MCC', fontweight='bold')
            # plt.title('Cross-validation - Matthews Correlation Coefficient', fontweight='bold')
            plt.tight_layout()

            # Save the plot as a PDF file
            # Note: data_modality and age_cutoff are not defined within this function
            if age_cutoff is not None:
                fname = f'{filepath}/mcc_raincloud_plot_agecutoff_{age_cutoff}.pdf'
            else:
                fname = f'{filepath}/mcc_raincloud_plot.pdf'
                
            if orient == 'v':
                fname = fname.replace('.pdf', '_vertical.pdf')
            elif orient == 'h':
                fname = fname.replace('.pdf', '_horizontal.pdf')
                
            fig.savefig(fname, dpi=300)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot ROC and PR curves")
    parser.add_argument('filepath', type=str, help="filepath")
    parser.add_argument('orient', type=str, help="orientation of the plot (h or v)")
    parser.add_argument('metric', type=str, help="metric")
    parser.add_argument('image_format', type=str, help="image format")
    # parser.add_argument('age65_cutoff', type=bool, help="use age cutoff of 65 (True/False)")
    
    args = parser.parse_args()
    multi_mean_roc_curve(args.filepath, args.metric, args.image_format)#, args.age65_cutoff)
    multi_mean_pr_curve(args.filepath, args.metric, args.image_format)#, args.age65_cutoff)
    multi_calibration_curve(args.filepath, args.metric, args.image_format)
    # The code is calling a function `export_confusion_matrices` with the argument `args.filepath`.
    # The function is likely designed to export confusion matrices to a file specified by the
    # `filepath` argument.
    export_confusion_matrices(args.filepath)
    mcc_raincloud(args.filepath, args.orient)