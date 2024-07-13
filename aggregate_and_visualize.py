# aggregate_and_visualize.py

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def collect_results(base_dir):
    results = []
    dataset_name = "CommonQA"
    for model_size in ["0.5B", "1.5B"]:
        for k_shot in ["0-shot", "1-shot", "2-shot", "5-shot"]:
            folder_name = f"qwen-Qwen2-{model_size}-Instruct_{dataset_name}_{k_shot}"
            eval_file = os.path.join(base_dir, folder_name, "evaluation.csv")

            if os.path.exists(eval_file):
                eval_df = pd.read_csv(eval_file)
                eval_df["Model"] = f"Qwen2-{model_size}"
                eval_df["k-shot"] = k_shot
                results.append(eval_df)

    return pd.concat(results, ignore_index=True)

def save_aggregate_results(results_df, output_file):
    results_df.to_csv(output_file, index=False)

def plot_results(results_df):
    # Convert 'k-shot' to numerical values for plotting
    results_df['k-shot'] = results_df['k-shot'].str.extract(r'(\d+)-shot')[0].astype(int)

    # Plotting
    plt.figure(figsize=(14, 8))
    sns.set(style="whitegrid")

    # Plot each method
    methods = results_df['Method'].unique()
    for method in methods:
        subset = results_df[results_df['Method'] == method]
        sns.lineplot(data=subset, x='k-shot', y='Accuracy', hue='Model', marker='o', label=method)

    plt.title('Model Accuracy by k-shot Examples and Method')
    plt.xlabel('Number of k-shot Examples')
    plt.ylabel('Accuracy')
    plt.legend(title='Method')
    plt.grid(True)
    plt.tight_layout()

    plt.savefig("accuracy_by_k_shot_and_method.png")
    plt.show()

if __name__ == "__main__":
    base_dir = "./experiments"  
    aggregate_csv_path = os.path.join(base_dir, f"aggregate_results.csv")

    # Collect and save results
    aggregated_results = collect_results(base_dir)
    save_aggregate_results(aggregated_results, aggregate_csv_path)

    # Plot results
    plot_results(aggregated_results)
