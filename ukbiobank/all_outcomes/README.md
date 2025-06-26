The main script is rfp_pipeline.py. 

For each of 607 outcomes:
    Stratified 80/20 train/test split
    For n_features in [5, 10, 100, 500, 1000]
        100 random sets of n_features
        fit to train (hist gradient boosting classifier), generate predicted probabilities on test
        100 bootstraps of test set predicted probabilities
    One set of using all proteins
        100 bootstraps of test set predicted probabilities

The full results file is tidy_data/all_outcomes/bootstrap/full_bs_results.parquet
The columns of this file are: 
n_features - number of random features
outcome - outcome diagnosis
iteration - which random feature subset
bootstrap - which bootstrap of the predicted test set probabilities
TN	
FP	
FN	
TP	
auroc	
avg_prec	
best_thresh
best_f1	
accuracy	
balanced_acc	
prec_neg	
prec_pos	
rec_neg	
rec_pos	
f1_neg	
f1_pos

The individual files in: tidy_data/all_outcomes/bootstrap/individual_results
These have a proteins column, which has the Protein IDs in each individual iteration. This column was not included in the full_bs_results file because a column of lists is enormous for 30,410,700 rows.