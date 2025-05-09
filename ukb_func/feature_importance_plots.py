import ukb_utils
import plot_results
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set font properties
plt.rcParams.update(
    {"font.size": 12, "font.weight": "bold"}  # Set font size  # Set font weight to bold
)

protein_code = pd.read_csv(
    "../../../proj_idp/tidy_data/proteomics/coding143.tsv", sep="\t"
)
# Split the column by semicolon and expand into separate columns
split_columns = protein_code["meaning"].str.split(";", expand=True)

# Rename the new columns (optional)
split_columns.columns = [f"part_{i+1}" for i in range(split_columns.shape[1])]

# Concatenate the new columns with the original DataFrame (optional)
protein_code = pd.concat([protein_code, split_columns], axis=1)

# Drop the original column if no longer needed
protein_code = protein_code.drop("meaning", axis=1)

lookup_dict = {
    25731: "Discrepancy, T1 brain <->\nstandard-space brain template",
    25889: "Vol. GM, Amyg (R)",
    25928: "rfMRI component amplitudes,\ndim 100",
    25521: "Weighted-mean MD in\ntract parahipp. cingulum (L)",
    26586: "Vol. Inf-Lat-Vent (R)",
    26555: "Vol. Inf-Lat-Vent (L)",
    25602: "Weighted-mean L2 in parahipp. cingulum (L)",
    25753: "rfMRI par. corr. mtx, dim 100",
    25826: "Vol. GM, Lat-Occipital Ctx, inf. div. (L)",
    27077: "Mean thickness, BA4a (L)",
    26639: "Vol. Whole-hipp-body (L)",
    26718: "Vol. Sup. Cereb. Ped.",
    26604: "Vol. Central-nucleus (L)",
    25888: "Vol. GM, Amyg (L)",
    27272: "Mean thickness, inferiorparietal (R)",
    26643: "Vol. subiculum-body (R)",
    26539: "Mean intensity, Inf-Lat-Vent (L)",
    20157: "Time to complete\nalphanumeric trail #2",
    20159: "Correct symbol digit matches",
    20156: "Time to complete\nalphanumeric trail #1",
    20132: "Number of incorrect matches in round",
    20191: "Fluid intelligence score",
    23045: "Very nervous mood over last week",
    31: "Sex",
    30780: "LDL",
    22038: "Moderate physical activity",
    845: "Age completed full time education",
    24018: "Nitrogen dioxide air pollution; 2007",
    22038: "Weekly minutes moderate activity",
    24019: "Air pollution (pm10); 2007",
    24006: "Air pollution (pm2.5); 2010",
    20161: "Pack years of smoking",
    24012: "Distance to nearest major road",
    24011: "Traffic intensity, nearest major road",
    21003: "Age",
    24015: "Sum of major road length within 100m",
    25001: "Vol. peripheral cortical GM",
    26649: "Vol. of presubiculum-body (R)",
    26545: "Mean intensity of Pallidum (L)",
    27596: "Area of S-cingul-Marginalis (R)",
    26546: "Mean intensity of Hippocampus (L)",
    26647: "Vol. presubiculum-head (R)",
    25198: "Mean MO in tapetum on FA skeleton (R)",
    26709: "Vol. Pt (R)",
    26890: "Vol. bankssts (R)",
    26625: "Vol. presubiculum-head (L)",
    25167: "Mean MO in cerebral peduncle on FA skeleton (L)",
    26611: "Vol. Basal-nucleus (R)",
    27026: "Grey-white contrast in caudalanteriorcingulate (R)",
    27354: "Area of G-pariet-inf-Supramar (L)",
    25724: "Weighted-mean ISOVF, tract post. thal. radiation (R)",
    26570: "Mean intensity of Inf-Lat-Vent (R)",
    25044: "Median BOLD effect, faces activation",
    25801: "Vol. GM, Sup. Temp. Gyrus, posterior division (R)",
    25446: "Mean ISOVF, corticospinal tract on FA skeleton (R)",
    27463: "Mean thickness, S-oc-temp-med+Lingual (L)",
    23072: "Downhearted/depressed over last week",
    20240: "Maximum digits remembered correctly",
    25752: "rfMRI partial correlation matrix, dimension 25",
    26633: "Vol. GC-ML-DG-body (L)",
    25185: "Mean MO in external capsule on FA skeleton (L)",
    26641: "Vol. Whole-hippocampus (L)",
    26868: "Mean thickness of lingual (R)",
    25285: "Mean L2, cingulum hippocampus, FA skeleton (L)",
    25005: "Vol. GM (normalised for head size)",
    25603: "Weighted-mean L2, tract parahipp. cingulum (R)",
    27008: "Grey-white contrast in parstriangularis (L)",
    27741: "Vol. Pole-temporal (R)",
    26609: "Vol. Whole-amygdala (L)",
    26637: "Vol. CA3-head (L)",
    25824: "Vol. GM in Lat. Occ. Ctx, superior division (L)",
    25214: "Mean L1 in cerebral peduncle on FA skeleton (R)",
    26645: "Vol. subiculum-head (R)",
    27679: "Mean thickness of S-interm-prim-Jensen (R)",
    26528: "Vol. WM-hypointensities (whole brain)",
    27680: "Mean thickness of S-intrapariet+P-trans (R)",
    26612: "Vol. Accessory-Basal-nucleus (R)",
    25734: "Inverted signal-to-noise ratio in T1",
    26602: "Vol. Accessory-Basal-nucleus (L)",
    25450: "Mean ISOVF, inf. cerebellar peduncle, FA skeleton (R)",
    20131: "Number of correct matches in round",
}

for outcome in ["alzheimers", "dementia"]:
    for modality in ["proteomics", "neuroimaging", "cognitive_tests"]:
        # proteomics
        modality_path = f"../../results/UKBiobank/{outcome}/{modality}"
        main_path = f"{modality_path}/demographics_modality_lancet2024/log_loss/lgbm/"
        for age_cutoff in [0, 65]:
            if age_cutoff == 0:
                filepath = main_path
            elif age_cutoff == 65:
                filepath = main_path + "agecutoff_65"
            df = plot_results.feature_importance_vals(filepath)

            # if modality == "proteomics":
            df = df[-20:]
            print(df.feature.values)
            # elif modality == "neuroimaging":
            #     df = df[-10:]
            # elif modality == "cognitive_tests":
            #     df = df[-10:]

            ticks = []

            for f in df.feature.values:
                print(f)
                if f[-4:] == "-0.0" or f[-4:] == "-0.1":
                    hyphen_idx = f.index("-")
                    field_id = f[:hyphen_idx]
                    ticks.append(lookup_dict[int(field_id)])
                elif f[-2:] == "-0" and modality == "proteomics":
                    hyphen_idx = f.index("-")
                    prot_id = f[:hyphen_idx]
                    sym = protein_code.loc[
                        protein_code.coding == int(prot_id), "part_1"
                    ].values[0]
                    ticks.append(sym)
                elif "21003" in f:
                    ticks.append("Age")
                elif "max_educ" in f:
                    ticks.append("Max education")
                elif "apoe_" in f:
                    allele_num = f[-3]
                    ticks.append(f"APOE$\epsilon$4, {allele_num} alleles")
                elif "freq_friends_family_visit" in f:
                    ticks.append("Frequency of friends/family visits")
                elif "845" in f:
                    ticks.append("Age completed full time education")

                elif ".0" in f:
                    # print(f)
                    hyphen_idx = f.index("-")
                    if modality == "neuroimaging":
                        ticks.append(lookup_dict[int(f[:hyphen_idx])])
                    elif modality == "cognitive_tests":
                        ticks.append(lookup_dict[int(f[:hyphen_idx])])
                elif "_0_" in f:
                    underscore_idx = f.index("_")
                    if modality == "neuroimaging":
                        ticks.append(lookup_dict[int(f[:underscore_idx])])
                    elif modality == "cognitive_tests":
                        ticks.append(lookup_dict[int(f[:underscore_idx])])

                elif f == "head_injury":
                    ticks.append("Head injury")
                elif f == "depression":
                    ticks.append("Depression")
                elif f == "alcohol_consumption":
                    ticks.append("Alcohol consumption")
                elif f == "hypertension":
                    ticks.append("Hypertension")
                elif f == "diabetes":
                    ticks.append("Diabetes")
                elif f == "hearing_loss":
                    ticks.append("Hearing loss")
                else:
                    ticks.append(f)

            plt.figure(figsize=(8, 8))

            # Create a horizontal bar plot
            plt.barh(
                df.feature, df.mean_importance, xerr=df.std_importance, color="skyblue"
            )

            print(len(df.feature), len(ticks))
            plt.yticks(ticks=df.feature, labels=ticks)

            # if modality == "proteomics":
            #     x_ticks = np.arange(
            #         0, max(df.mean_importance) + 9, 4
            #     )  # Adjust the range and step as needed
            # elif modality == "neuroimaging":
            #     x_ticks = np.arange(
            #         0, max(df.mean_importance) + 9, 4
            #     )  # Adjust the range and step as needed
            # elif modality == "cognitive_tests":
            #     x_ticks = np.arange(
            #         0, max(df.mean_importance) + 9, 4
            #     )  # Adjust the range and step as needed
            # plt.xticks(ticks=x_ticks)
            plt.xlabel("Feature Importance")
            plt.tight_layout()
            if age_cutoff == 0:
                plt.savefig(f"{modality_path}/feature_importance_figure_2025.pdf")
                plt.savefig(
                    f"{modality_path}/feature_importance_figure_2025.png", dpi=300
                )
            elif age_cutoff == 65:
                plt.savefig(
                    f"{modality_path}/feature_importance_figure_2025_agecutoff_65.pdf"
                )
                plt.savefig(
                    f"{modality_path}/feature_importance_figure_2025_agecutoff_65.png",
                    dpi=300,
                )


# # neuroimaging
# main_path = (
#     "../../results/dementia/neuroimaging/demographics_and_lancet2024/log_loss/lgbm/"
# )

# for age_cutoff in [0, 65]:
#     if age_cutoff == 0:
#         filepath = main_path
#     elif age_cutoff == 65:
#         filepath = main_path + "agecutoff_65"
#     df = plot_results.feature_importance_vals(filepath)
#     df = df[-10:]

#     plt.figure(figsize=(8, 6))

#     # Create a horizontal bar plot
#     plt.barh(df.feature, df.mean_importance, xerr=df.std_importance, color="skyblue")

#     ticks = []

#     for f in df.feature.values:
#         if "apoe_" in f:
#             allele_num = f[-3]
#             ticks.append(f"APOE$\epsilon$4, {allele_num} alleles")
#         elif "21003" in f:
#             ticks.append("Age")
#         elif ".0" in f:
#             hyphen_idx = f.index("-")
#             ticks.append(idp_name_dict[int(f[:hyphen_idx])])
#         elif "_0_" in f:
#             underscore_idx = f.index("_")
#             ticks.append(idp_name_dict[int(f[:underscore_idx])])
#         else:
#             ticks.append(f)

#     plt.yticks(ticks=df.feature, labels=ticks)

#     x_ticks = np.arange(
#         0, max(df.mean_importance) + 1, 1
#     )  # Adjust the range and step as needed
#     plt.xticks(ticks=x_ticks)
#     plt.xlabel("Feature Importance")
#     plt.tight_layout()
#     plt.savefig(f"{filepath}/feature_importance_figure_2025.pdf")


# # cognitive tests
# main_path = "../../results/dementia/cognitive_tests/demographics_modality_lancet2024/log_loss/lgbm/"

# for age_cutoff in [0, 65]:
#     if age_cutoff == 0:
#         filepath = main_path
#     elif age_cutoff == 65:
#         filepath = main_path + "agecutoff_65"
#     df = plot_results.feature_importance_vals(filepath)
#     df = df[-10:]

#     plt.figure(figsize=(8, 6))

#     # Create a horizontal bar plot
#     plt.barh(df.feature, df.mean_importance, xerr=df.std_importance, color="skyblue")

#     ticks = []

#     for f in df.feature.values:
#         if "apoe_" in f:
#             allele_num = f[-3]
#             ticks.append(f"APOE$\epsilon$4, {allele_num} alleles")
#         elif "21003" in f:
#             ticks.append("Age")
#         elif ".0" in f:
#             hyphen_idx = f.index("-")
#             ticks.append(cognitive_tick_dict[int(f[:hyphen_idx])])
#         elif "_0_" in f:
#             underscore_idx = f.index("_")
#             ticks.append(cognitive_tick_dict[int(f[:underscore_idx])])
#         elif f == "head_injury":
#             ticks.append("Head injury")
#         elif f == "depression":
#             ticks.append("Depression")
#         elif f == "alcohol_consumption":
#             ticks.append("Alcohol consumption")
#         elif f == "hypertension":
#             ticks.append("Hypertension")
#         elif f == "diabetes":
#             ticks.append("Diabetes")
#         else:
#             ticks.append(f)

#     # cognitive_ticks = ['Age',
#     #
#     #
#     #                     f'APOE$\epsilon$4, 2 alleles',
#     #
#     #
#     #                     'Maximum digits remembered correctly',
#     #
#     #
#     #                     'Sex'
#     #                         ]

#     plt.yticks(ticks=df.feature, labels=ticks)

#     x_ticks = np.arange(
#         0, max(df.mean_importance) + 2, 4
#     )  # Adjust the range and step as needed
#     plt.xticks(ticks=x_ticks)
#     plt.xlabel("Feature Importance")
#     plt.tight_layout()
#     plt.savefig(f"{filepath}/feature_importance_figure_2025.pdf")
