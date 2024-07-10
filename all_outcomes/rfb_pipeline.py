import argparse
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import precision_recall_curve, average_precision_score, confusion_matrix, roc_auc_score, precision_recall_fscore_support, accuracy_score, balanced_accuracy_score
import sys
from datetime import datetime

path = '../tidy_data'


def main():
    def bootstrap(true_labels, proba_predict, n_bootstrap, bs_w_replace=False):
        def append_results():
            yhat = proba_predict[bs_index]
            y_bs = true_labels.values[bs_index]

            auroc = roc_auc_score(y_bs,
                                  yhat)
            ap = average_precision_score(y_bs, yhat)
            # calculate pr-curve
            precision, recall, thresholds = precision_recall_curve(y_bs, yhat)
            # convert to f score
            fscore = (2 * precision * recall) / (precision + recall)
            # locate the index of the largest f score
            ix = np.nanargmax(fscore)
            best_threshold = thresholds[ix]
            best_fscore = fscore[ix]

            test_pred = (yhat > best_threshold).astype(int)

            tn, fp, fn, tp = confusion_matrix(y_bs,
                                              test_pred).ravel()
            acc = accuracy_score(y_bs, test_pred)
            bal_acc = balanced_accuracy_score(y_bs,
                                              test_pred)
            prfs = precision_recall_fscore_support(y_bs,
                                                   test_pred)
            nfeats_l.append(nfeat)
            proteins_l.append(p)
            outcome_l.append(oc)
            iter_l.append(i)
            bs_l.append(j)
            # true_labels_l.append(y_bs)
            # probas_l.append(yhat)
            auroc_l.append(auroc)
            ap_l.append(ap)
            best_threshold_l.append(best_threshold)
            best_fscore_l.append(best_fscore)
            tn_l.append(tn)
            fp_l.append(fp)
            fn_l.append(fn)
            tp_l.append(tp)
            accuracy_l.append(acc)
            balanced_accuracy_l.append(bal_acc)
            prec_n.append(prfs[0][0])
            prec_p.append(prfs[0][1])
            rec_n.append(prfs[1][0])
            rec_p.append(prfs[1][1])
            f1_n.append(prfs[2][0])
            f1_p.append(prfs[2][1])
        if bs_w_replace is False:
            # run 100 bootstraps of test set
            skf = StratifiedKFold(n_splits=n_bootstrap)
            # for j in range(100):
            for j, (bs_index, nonbs_index) in enumerate(
                    skf.split(proba_predict, true_labels)):
                append_results()
        else:
            bootstraps = np.array([np.random.choice(range(len(true_labels)),
                                   len(true_labels), replace=True)
                                   for _ in range(n_bootstrap)])
            for j, bs_index in enumerate(bootstraps):
                append_results()

    df = pd.read_parquet(f'{path}/proteomics_first_occurrences.parquet')
    proteins = pd.read_csv(f'{path}/protein_colnames.txt', header=None)
    proteins = proteins[0].tolist()
    outcomes = pd.read_csv(f'{path}/outcome_colnames.txt', header=None)
    outcomes = outcomes[0].tolist()
    demo = ['21003-0.0', '21003-0.0_squared', '31-0.0', '53-0.0', '54-0.0',]

    # Create the argument parser
    parser = argparse.ArgumentParser(description='Random feature baselines pipeline with parallelization')

    # Add arguments
    # parser.add_argument('njobs', type=str, help='Number of cores')
    parser.add_argument('--task_id', type=int, help='Task ID (i.e., njobs / number of outcomes per job)')

    # Parse the arguments
    args = parser.parse_args()
    task_id = args.task_id
    print('Running job number ', task_id)

    n_subset = 100
    n_bootstrap = 100
    bs_w_replace = True

    # set start and end indices
    chunk_size = 1
    start = (chunk_size * (task_id))
    end = chunk_size * (task_id+1)
    if start > len(outcomes):
        sys.exit('start index is greater than length of predictor vector')
    if end > len(outcomes):
        end = len(outcomes)
    print('Start ', start)
    print('End ', end)

    # process date column
    df['53-0.0'] = (df['53-0.0'] - min(df['53-0.0']))
    df['53-0.0'] = df['53-0.0'].apply(lambda x: x.days)

    # process sex and site columns
    enc = OneHotEncoder()
    enc.fit(df.loc[:, ['31-0.0', '54-0.0']])
    categ_enc = pd.DataFrame(enc.transform(df.loc[:, ['31-0.0', '54-0.0']]).toarray(), columns=enc.get_feature_names_out(['31-0.0', '54-0.0']))
    df = df.drop(columns=['31-0.0', '54-0.0'])
    df = df.join(categ_enc)

    '''
    Iterate over outcomes
    create labels
    subset df to take all proteins, all demo variables, and outcome
    Iterate over 10 80% train/20% test splits, stratifying by label
    Iterate over number of features [5, 50, 100, 500, 1000, all], do 100 rounds of
    features except when doing all features choose proteins
    fit HistGradientBoostingClassifier, calculate AUROC, accuracy,
    balanced accuracy, precision, recall, F1
    Store confusion matrix, metrics, number of features, which train/test split
    '''

    # store results
    nfeats_l = []
    proteins_l = []
    outcome_l = []
    iter_l = []
    bs_l = []
    tn_l = []
    fp_l = []
    fn_l = []
    tp_l = []
    auroc_l = []
    ap_l = []
    best_threshold_l = []
    best_fscore_l = []
    accuracy_l = []
    balanced_accuracy_l = []
    prec_n = []
    prec_p = []
    rec_n = []
    rec_p = []
    f1_n = []
    f1_p = []

    # remove non-encoded sex and site
    demo_nosite_nosex = [x for x in demo if x not in ['31-0.0', '54-0.0']]
    outcome_subset = outcomes[start:end]
    for i, oc in zip(list(range(start, end)), outcome_subset):
        print('outcome:', oc)
        df['label'] = df[oc].notna().astype(int)
        df_sub = df.loc[:, proteins + demo_nosite_nosex + ['label']]

        y = df_sub.label
        X_start = df_sub.drop(columns=['label'])

        # train-test split
        train_idx, test_idx, y_train, y_test = train_test_split(
                list(range(X_start.shape[0])), y, test_size=0.2, stratify=y, random_state=99)

        for nfeat in [5, 50, 100, 500, 1000, len(proteins)]:
            print('Number of features:', nfeat)
            if nfeat < len(proteins):
                for i in range(n_subset):
                    if i % 10 == 0:
                        now = datetime.now()
                        current_time = now.strftime("%H:%M:%S")
                        print(f'{nfeat}-protein set {i}, Current Time = {current_time}')

                    np.random.seed(seed=i)
                    p = np.random.choice(proteins, size=nfeat, replace=False)
                    p_drop = set(proteins).difference(set(p))
                    X = X_start.drop(columns=p_drop)

                    # fit to training set
                    clf = HistGradientBoostingClassifier(
                        class_weight='balanced').fit(X.iloc[train_idx], y_train)

                    # save probability predictions
                    proba_predict = clf.predict_proba(X.iloc[test_idx])

                    bootstrap(y_test, proba_predict[:, 1], n_bootstrap, bs_w_replace=True)

            else:
                np.random.seed(seed=0)
                X = X_start

                # fit to training set
                clf = HistGradientBoostingClassifier(
                    class_weight='balanced').fit(X.iloc[train_idx], y_train)

                # save probability predictions
                proba_predict = clf.predict_proba(X.iloc[test_idx])

                bootstrap(y_test, proba_predict[:, 1], n_bootstrap, bs_w_replace=True)

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Analyses complete. Current Time =", current_time)
    data = {'n_features': nfeats_l, 'proteins': proteins_l,
            'outcome': outcome_l,
            'iteration': iter_l, 'bootstrap': bs_l,
            'TN': tn_l, 'FP': fp_l,
            'FN': fn_l, 'TP': tp_l, 'auroc': auroc_l, 'avg_prec': ap_l,
            'best_thresh': best_threshold_l, 'best_f1': best_fscore_l,
            'accuracy': accuracy_l,
            'balanced_acc': balanced_accuracy_l, 'prec_neg': prec_n,
            'prec_pos': prec_p, 'rec_neg': rec_n, 'rec_pos': rec_p,
            'f1_neg': f1_n, 'f1_pos': f1_p}

    results = pd.DataFrame(data)

    if bs_w_replace == True:
        results.to_parquet(f'{path}/bootstrap/results_{oc}_bsWreplace_{task_id}.parquet')
    else:
        results.to_parquet(f'{path}/bootstrap/results_{oc}_{task_id}.parquet')

    # # Save all true labels
    # file_path = f'{path}/bootstrap/all_true_labels_{oc}_{task_id}.pkl'
    # with open(file_path, 'wb') as file:
    #     # Serialize and write the nested list to the file
    #     pickle.dump(true_labels_l, file)

    # # Save all predicted probabilities
    # file_path = f'{path}/bootstrap/all_probas_{oc}_{task_id}.pkl'
    # with open(file_path, 'wb') as file:
    #     # Serialize and write the nested list to the file
    #     pickle.dump(probas_l, file)

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Results saved. All done. Current Time =", current_time)


if __name__ == '__main__':
    main()
