# evaluation.py

import pandas as pd
import os

def compute_accuracy(results, method):
    correct = 0
    for item in results:
        model_choice = item[method]
        correct_choice = item["Answer"]
        if model_choice.isdigit():
            if chr(65 + int(model_choice) - 1) == correct_choice:
                correct += 1
        elif model_choice == correct_choice:
            correct += 1
    return correct / len(results)

def evaluate_results(results, folder_name):
    accuracy = {
        "CP_Raw": compute_accuracy(results, "CP_Raw"),
        "CP_UN": compute_accuracy(results, "CP_UN"),
        "CP_LN": compute_accuracy(results, "CP_LN"),
        "MCP": compute_accuracy(results, "MCP_output")
    }

    accuracy_df = pd.DataFrame(list(accuracy.items()), columns=["Method", "Accuracy"])
    evaluation_csv_path = os.path.join(folder_name, 'evaluation.csv')
    accuracy_df.to_csv(evaluation_csv_path, index=False)

    return accuracy_df
