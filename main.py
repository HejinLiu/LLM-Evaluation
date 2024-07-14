# main.py

import argparse
from config import MODELS, DATASETS
from data_processing import load_data, get_examples
from model_inference import run_inference
from evaluation import evaluate_results
import os
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Run k-shot inference and evaluation.")
    parser.add_argument("--model_name", choices=MODELS, required=True, help="Model name to use for inference, like'qwen/Qwen2-0.5B-Instruct'.")
    parser.add_argument("--k", type=int, required=True, help="Number of k-shot examples to use.")
    parser.add_argument("--dataset_name", choices=DATASETS, required=True, help="Dataset name to use.")

    args = parser.parse_args()

    folder_name = f"experiments/{args.model_name.replace('/', '-')}_{args.dataset_name}_{args.k}-shot"
    os.makedirs(folder_name, exist_ok=True)

    # load data
    data = load_data(args.dataset_name)
    dev_data = load_data(args.dataset_name, is_dev=True)

    # get examples for k-shot evaluation
    examples = get_examples(dev_data, args.k)

    # inference
    results = run_inference(data, examples, args.model_name, folder_name)

    # save to CSV
    results_df = pd.DataFrame(results)
    results_csv_path = os.path.join(folder_name, 'results.csv')
    results_df.to_csv(results_csv_path, index=False)

    # integrate results
    accuracy_df = evaluate_results(results, folder_name)
    print("\n--------Evaluation Completed--------")
    print(accuracy_df)

if __name__ == "__main__":
    main()
