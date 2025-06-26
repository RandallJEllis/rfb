import pandas as pd
import os


def anno_protein_set(proteins):
    script_dir = os.path.dirname(__file__)
    prot_annot = pd.read_csv(
        os.path.join(script_dir, "../../metadata/coding143.tsv"), sep="\t"
    )
    prot_annot[["protein_id", "meaning"]] = prot_annot["meaning"].str.split(
        ";", expand=True
    )
    prot_id = []
    for p in proteins:
        hyphen_idx = p.index("-")
        prot_id.append(int(p[:hyphen_idx]))
    sub_anno = prot_annot[prot_annot.coding.isin(prot_id)]
    return sub_anno


def pull_proteins(path, outcome, nf, iteration):
    """Pull the top proteins for a given outcome, number of features, and iteration.

    Args:
        path (str): The path to the directory containing the bootstrap results.
        outcome (str): The outcome to pull the top proteins for.
        nf (int): The number of features to pull the top proteins for.
        iteration (int): The iteration to pull the top proteins for.

    Returns:
        pd.DataFrame: A dataframe containing the top proteins for the given outcome, number of features, and iteration.

    Raises:
        ValueError: If the outcome is not found in the directory.

    Example:
        >>> pull_proteins('../tidy_data/bootstrap/individual_results/', 'outcome', 10, 1)
    """
    for f in os.listdir(path):
        if outcome in f:
            break
    df = pd.read_parquet(f"{path}/{f}")
    sub = df[
        (df.outcome == f"{outcome}-0.0")
        & (df.n_features == nf)
        & (df.iteration == iteration)
    ].iloc[0]
    prot_df = anno_protein_set(sub.proteins.tolist())
    return prot_df


def main():
    df = pd.read_parquet("../tidy_data/bootstrap/full_bs_results.parquet")
    top_rows = df.groupby(["n_features", "outcome"]).first().reset_index()
    top_rows = top_rows[top_rows.n_features != 2923]
    top_rows = top_rows.sort_values(by=["best_f1"], ascending=False)

    outcome_top_rows = top_rows.groupby(["outcome"]).first()
    outcome_top_rows = outcome_top_rows.sort_values(by=["best_f1"], ascending=False)

    df_l = []

    for i in range(outcome_top_rows.shape[0]):
        print(i)
        dfdf = pull_proteins(
            "../tidy_data/bootstrap/individual_results/",
            str(outcome_top_rows.index.values[i]),
            top_rows.n_features.values[i],
            top_rows.iteration.values[i],
        )
        dfdf["outcome"] = top_rows.outcome.values[i]
        dfdf["n_features"] = top_rows.n_features.values[i]
        dfdf["iteration"] = top_rows.iteration.values[i]
        df_l.append(dfdf)

        alldf = pd.concat(df_l)
        alldf.to_parquet("../tidy_data/bootstrap/top_proteins.parquet")


if __name__ == "__main__":
    main()
