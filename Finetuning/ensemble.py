import pandas as pd
import os
import glob
import argparse
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
import numpy as np
from sklearn.metrics import accuracy_score  # Retained for reference if needed

def load_csvs(csv_dir, pattern="*_predictions.csv", prediction_threshold=0.5):
    """
    Loads all CSV files matching the pattern from the specified directory and calculates individual file accuracies and AUC.

    Args:
        csv_dir (str): Directory containing the CSV files.
        pattern (str, optional): Glob pattern to match CSV files.
        prediction_threshold (float, optional): Threshold to convert probabilities to binary predictions.

    Returns:
        list of pd.DataFrame: List containing DataFrames of each CSV.
    """
    csv_files = glob.glob(os.path.join(csv_dir, pattern))
    if not csv_files:
        raise ValueError(f"No CSV files found in {csv_dir} with pattern {pattern}")

    dataframes = []
    for file in csv_files:
        df = pd.read_csv(file)
        required_columns = {'image_path', 'prob', 'prediction', 'label'}
        if not required_columns.issubset(df.columns):
            raise ValueError(f"CSV file {file} is missing required columns.")

        # Make predictions based on the specified probability threshold
        df['prediction'] = (df['prob'] >= 0.3).astype(int)
        
        # Calculate individual file balanced accuracy and AUC
        balanced_acc = balanced_accuracy_score(df['label'], df['prediction'])
        try:
            auc = roc_auc_score(df['label'], df['prob'])
        except ValueError:
            auc = float('nan')  # In case all true labels are the same and AUC can't be computed

        print(f"Loaded {file} with {len(df)} entries. Balanced Accuracy: {balanced_acc:.4f}, AUC: {auc:.4f}")

        # if balanced_acc > 0.57:
        #     print("Added")
        dataframes.append(df)
        # else:
        #     print("Not Added")

    return dataframes

def merge_csvs(dataframes):
    """
    Merges multiple DataFrames on 'image_path' and 'label'.

    Args:
        dataframes (list of pd.DataFrame): List of DataFrames to merge.

    Returns:
        pd.DataFrame: Merged DataFrame containing all predictions and labels.
    """
    merged_df = dataframes[0].copy()
    merged_df = merged_df.rename(columns={
        'prob': 'prob_1',
        'prediction': 'pred_1'
    })

    for i, df in enumerate(dataframes[1:], start=2):
        merged_df = pd.merge(
            merged_df,
            df.rename(columns={
                'prob': f'prob_{i}',
                'prediction': f'pred_{i}'
            }),
            on=['image_path', 'label'],
            how='inner'
        )
        print(f"Merged CSV {i} with {len(df)} entries.")

    print(f"Total merged entries: {len(merged_df)}")
    return merged_df

def average_ensemble(merged_df, num_models, threshold=0.5):
    """
    Performs average ensemble by averaging probabilities and applying threshold.

    Args:
        merged_df (pd.DataFrame): Merged DataFrame containing all predictions.
        num_models (int): Number of models involved in the ensemble.
        threshold (float): Threshold to convert averaged probability to binary prediction.

    Returns:
        tuple: (final_pred, averaged_prob)
    """
    prob_cols = [f'prob_{i}' for i in range(1, num_models + 1)]
    averaged_prob = merged_df[prob_cols].mean(axis=1)
    final_pred = (averaged_prob >= threshold).astype(int)
    return final_pred, averaged_prob


def evaluate_ensemble(merged_df, num_models, thresholds_avg):
    """
    Evaluates ensemble methods across different thresholds.

    Args:
        merged_df (pd.DataFrame): Merged DataFrame containing all predictions.
        num_models (int): Number of models involved in the ensemble.
        thresholds_avg (list of float): Thresholds for average ensemble.
        thresholds_vote (list of float): Thresholds for majority voting ensemble.

    Returns:
        dict: Dictionary containing balanced accuracy and AUC results for each method and threshold.
    """
    results = {
        'Average_Ensemble': {},
    }

    true_labels = merged_df['label'].values

    # Average Ensemble Evaluation
    for thresh in thresholds_avg:
        preds, averaged_prob = average_ensemble(merged_df, num_models, thresh)
        balanced_acc = balanced_accuracy_score(true_labels, preds)
        try:
            auc = roc_auc_score(true_labels, averaged_prob)
        except ValueError:
            auc = float('nan')  # In case AUC cannot be computed
        results['Average_Ensemble'][thresh] = {'Balanced Accuracy': balanced_acc, 'AUC': auc}
        print(f"Average Ensemble - Threshold: {thresh:.2f}, Balanced Accuracy: {balanced_acc:.4f}, AUC: {auc:.4f}")

    return results

def main():
    """
    Main function to load CSVs, perform ensemble methods, and evaluate balanced accuracy and AUC.
    """
    parser = argparse.ArgumentParser(description="Ensemble Model Evaluation with Balanced Accuracy")
    parser.add_argument(
        "-c", "--csv_dir",
        type=str,
        default="predictions/fine_tuned_models",
        help="Directory containing the prediction CSV files."
    )
    parser.add_argument(
        "-p", "--pattern",
        type=str,
        default="*_predictions.csv",
        help="Glob pattern to match prediction CSV files."
    )
    parser.add_argument(
        "--threshold_start",
        type=float,
        default=0.1,
        help="Starting threshold value for average ensemble."
    )
    parser.add_argument(
        "--threshold_end",
        type=float,
        default=0.9,
        help="Ending threshold value for average ensemble."
    )
    parser.add_argument(
        "--threshold_step",
        type=float,
        default=0.1,
        help="Step size for threshold values."
    )
    parser.add_argument(
        "--prediction_threshold",
        type=float,
        default=0.5,
        help="Threshold to convert probabilities to binary predictions for individual models."
    )

    args = parser.parse_args()

    # Load CSVs with the specified prediction threshold
    dataframes = load_csvs(args.csv_dir, args.pattern, prediction_threshold=args.prediction_threshold)
    num_models = len(dataframes)
    print(f"Number of models loaded: {num_models}")

    # Merge CSVs
    merged_df = merge_csvs(dataframes)

    # Define threshold ranges for average ensemble
    thresholds_avg = np.arange(args.threshold_start, args.threshold_end + args.threshold_step, args.threshold_step)
    thresholds_avg = [round(t, 2) for t in thresholds_avg]


    print(f"Average Ensemble Thresholds: {thresholds_avg}")

    # Evaluate Ensemble Methods
    results = evaluate_ensemble(merged_df, num_models, thresholds_avg)

    # Display Results
    print("\n=== Ensemble Evaluation Results ===")
    print("\nAverage Ensemble:")
    for thresh, metrics in results['Average_Ensemble'].items():
        print(f"Threshold: {thresh:.2f} -> Balanced Accuracy: {metrics['Balanced Accuracy']:.4f}, AUC: {metrics['AUC']:.4f}")


    # Prepare DataFrame for Saving Results
    records = []
    for thresh, metrics in results['Average_Ensemble'].items():
        records.append({
            'Method': 'Average_Ensemble',
            'Threshold': thresh,
            'Balanced Accuracy': metrics['Balanced Accuracy'],
            'AUC': metrics['AUC']
        })
    results_df = pd.DataFrame(records)

    # Define the output directory and ensure it exists
    ensemble_results_dir = os.path.join(args.csv_dir, "ensemble_evaluation")
    os.makedirs(ensemble_results_dir, exist_ok=True)

    # Save the results to a CSV
    results_csv_path = os.path.join(ensemble_results_dir, "ensemble_evaluation_results.csv")
    try:
        results_df.to_csv(results_csv_path, index=False)
        print(f"\nSaved ensemble evaluation results to '{results_csv_path}'.")
    except Exception as e:
        print(f"Error saving ensemble evaluation results: {e}")

if __name__ == "__main__":
    main()
