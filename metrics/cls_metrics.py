"""
This module is used to measure the metrics of the Classification models. It
measures:
- Accuracy
- Precision
- Recall
- F1 Score
x ROC Curve
x AUC Score

Author: Deja S.
Version: 1.0.0
Created: 23-04-2024
Last Edit: 23-04-2024
"""

import os
import tqdm
import argparse
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, confusion_matrix

if __name__ == "__main__":
    # Argument Parser
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--file", required=True, type=str, help="The CSV file path.")
    ap.add_argument("-n", "--name", required=False, type=str, default=None, help="Name of the model.")
    opts = vars(ap.parse_args())

    file_path = opts['file']
    name = opts['name']
    output_file_path = f"./runs/metrics/classification_metric_{name}_output_{datetime.now()}.txt" if name is not None else f"./runs/metrics/classification_metric_output_{datetime.now()}.txt"

    # Check if the file path is valid
    if not os.path.exists(file_path):
        print(f"ERROR: file path '{file_path}' is not found or does not exist!")
        exit()

    # Read file
    data = pd.read_csv(file_path)

    # Get the true and predicted labels
    true_labels = data.loc[:, "true"]
    predicted_labels = data.loc[:, "predicted"]

    # Calculate the metrics
    acc_score = accuracy_score(true_labels, predicted_labels)
    pre_score = precision_score(true_labels, predicted_labels, average='macro')
    rec_score = recall_score(true_labels, predicted_labels, average='macro')
    f1_score = f1_score(true_labels, predicted_labels, average='macro')

    # Display the metrics
    print("Metrics:")
    print("=" * 80)
    print(f"Accuracy Score:\t{acc_score:.3f}")
    print(f"Precision Score:\t{pre_score:.3f}")
    print(f"Recall Score:\t{rec_score:.3f}")
    print(f"F1 Score:\t{f1_score:.3f}")
    print("=" * 80)

    # Writing results to a file
    with open(name, w) as file:
        file.write("Metrics:\n")
        file.write("=" * 80)
        file.write(f"\nAccuracy Score:\t{acc_score:.3f}\n")
        file.write(f"Precision Score:\t{pre_score:.3f}\n")
        file.write(f"Recall Score:\t{rec_score:.3f}\n")
        file.write(f"F1 Score:\t{f1_score:.3f}\n")
        file.write("=" * 80)
    print("Done.")
