import os
import pandas as pd
import numpy as np

import sys
from datetime import datetime
import pickle

import matplotlib.pyplot as plt

sys.path.append("./ukb_func")
from plot_results import *
import ukb_utils
import utils
import ml_utils
import argparse


parser = argparse.ArgumentParser()
parser.add_argument(
    "--modality", type=str, choices=["neuroimaging", "proteomics", "cognitive_tests"]
)
args = parser.parse_args()
modality = args.modality

demo_lancet = pd.read_csv(
    f"../../results/UKBiobank/alzheimers/{modality}/demographics_and_lancet2024/log_loss/lgbm/test_results.csv"
)
_, demo_modality_lancet = probas_to_results(
    f"../../results/UKBiobank/alzheimers/{modality}/demographics_modality_lancet2024/feature_selection/log_loss/lgbm/"
)

# print mean, median, STD of MCC for each dataset
print(
    "demo_lancet",
    np.mean(demo_lancet["mcc"]),
    np.std(demo_lancet["mcc"]),
    np.median(demo_lancet["mcc"]),
)
print(
    "FS demo_modality_lancet",
    np.mean(demo_modality_lancet["mcc"]),
    np.std(demo_modality_lancet["mcc"]),
    np.median(demo_modality_lancet["mcc"]),
)
