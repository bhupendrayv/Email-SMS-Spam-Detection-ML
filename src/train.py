"""
Comprehensive Training Pipeline:
Trains and evaluates all Classic ML and Deep Learning models across both
Bag-of-Words (BoW) and TF-IDF text representations.
"""

import os
import time
import json
from copy import deepcopy
import pandas as pd
import joblib

from src.dataset import load_dataset, get_train_test_data, DATASET_INFO
from src.preprocessor import TextPreprocessor
from src.feature_extraction import FeatureExtractionPipeline
from src.models.classic_models import get_classic_models
from src.models.deep_models import get_deep_learning_models
from src.evaluate import evaluate_model, create_comparison_table

import argparse


PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")


def train_and_evaluate_all(dataset_key: str = "sms"):
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    ds_name = DATASET_INFO.get(dataset_key, {}).get("name", dataset_key)

    print("=" * 75)
    print(f"      SPAM DETECTION BENCHMARK — {ds_name}")
    print("=" * 75)

    # 1. Load data
    print(f"\n[1/5] Loading and splitting dataset ({dataset_key})...")
    X_train_raw, X_test_raw, y_train, y_test = get_train_test_data(dataset_key=dataset_key)
    print(f"Total training samples: {len(X_train_raw)} (Spam: {sum(y_train)}, Ham: {len(y_train) - sum(y_train)})")
    print(f"Total testing samples:  {len(X_test_raw)} (Spam: {sum(y_test)}, Ham: {len(y_test) - sum(y_test)})")

    # 2. Text Preprocessing
    print("\n[2/5] Preprocessing text data (cleaning, entity normalization, stemming)...")
    preprocessor = TextPreprocessor()
    t0 = time.time()
    X_train_clean = [preprocessor.clean_text(t) for t in X_train_raw]
    X_test_clean = [preprocessor.clean_text(t) for t in X_test_raw]
    print(f"Text preprocessing completed in {time.time() - t0:.2f}s")

    # 3. Feature Extraction: BoW and TF-IDF
    print("\n[3/5] Extracting features: Bag-of-Words (BoW) & TF-IDF...")
    extractor = FeatureExtractionPipeline(max_features=5000, ngram_range=(1, 2))
    extractor.fit(X_train_clean)
    extractor.save(MODELS_DIR)

    X_train_bow = extractor.transform_bow(X_train_clean)
    X_test_bow = extractor.transform_bow(X_test_clean)

    X_train_tfidf = extractor.transform_tfidf(X_train_clean)
    X_test_tfidf = extractor.transform_tfidf(X_test_clean)

    print(f"BoW Matrix shape:   {X_train_bow.shape} (Sparsity: {100.0 * (1.0 - X_train_bow.nnz / (X_train_bow.shape[0] * X_train_bow.shape[1])):.2f}%)")
    print(f"TF-IDF Matrix shape:{X_train_tfidf.shape} (Sparsity: {100.0 * (1.0 - X_train_tfidf.nnz / (X_train_tfidf.shape[0] * X_train_tfidf.shape[1])):.2f}%)")

    # 4. Prepare Models
    classic_models = get_classic_models()
    deep_models = get_deep_learning_models()
    
    all_models_dict = {**classic_models, **deep_models}

    all_results = []
    trained_artifacts = {}

    # 5. Training loop over representations
    representations = [
        ("Bag-of-Words (BoW)", X_train_bow, X_test_bow, "bow"),
        ("TF-IDF", X_train_tfidf, X_test_tfidf, "tfidf")
    ]

    print("\n[4/5] Training and evaluating models across BoW vs TF-IDF...")
    total_runs = len(representations) * len(all_models_dict)
    run_idx = 1

    for rep_name, X_tr, X_te, rep_slug in representations:
        print(f"\n--- Running Representation: {rep_name} ---")
        for model_name, model_blueprint in all_models_dict.items():
            print(f"[{run_idx}/{total_runs}] Training {model_name} with {rep_name}...", end=" ", flush=True)
            model_instance = deepcopy(model_blueprint)

            start_t = time.perf_counter()
            model_instance.fit(X_tr, y_train)
            train_duration = time.perf_counter() - start_t

            # Evaluate
            metrics = evaluate_model(
                model=model_instance,
                X_test=X_te,
                y_test=y_test,
                model_name=model_name,
                feature_rep=rep_name,
                training_time=train_duration
            )
            all_results.append(metrics)

            # Save model artifact
            safe_name = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
            model_file = f"{rep_slug}_{safe_name}.joblib"
            joblib.dump(model_instance, os.path.join(MODELS_DIR, model_file))

            print(f"Done in {train_duration:.2f}s | Acc: {metrics['accuracy']:.4f} | Prec: {metrics['precision']:.4f} | F1: {metrics['f1_score']:.4f}")
            run_idx += 1

    # 6. Save results
    print("\n[5/5] Compiling and saving benchmark results...")
    comparison_df = create_comparison_table(all_results)
    csv_path = os.path.join(RESULTS_DIR, "metrics_summary.csv")
    json_path = os.path.join(RESULTS_DIR, "comparison_report.json")

    comparison_df.to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print nicely formatted summary table
    print("\n" + "=" * 90)
    print("                     MODEL COMPARISON LEADERBOARD")
    print("=" * 90)
    print(comparison_df.to_string(index=False))
    print("=" * 90)

    # Top recommendations
    best_f1_row = max(all_results, key=lambda x: x["f1_score"])
    best_prec_row = max(all_results, key=lambda x: (x["precision"], x["f1_score"]))
    
    print("\nKEY FINDINGS & HIGHLIGHTS:")
    print(f"1. Highest F1-Score: {best_f1_row['model_name']} ({best_f1_row['feature_representation']}) -> F1: {best_f1_row['f1_score']:.4f}, Accuracy: {best_f1_row['accuracy']:.4f}")
    print(f"2. Highest Precision: {best_prec_row['model_name']} ({best_prec_row['feature_representation']}) -> Precision: {best_prec_row['precision']:.4f} (Crucial for 0 False Positives)")
    print(f"3. Results saved to: {csv_path} and {json_path}")

    return all_results, comparison_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and benchmark spam detection models")
    parser.add_argument(
        "--dataset", "-d",
        choices=["sms", "email", "phishing", "urls"],
        default="sms",
        help="Dataset to train on (default: sms)"
    )
    args = parser.parse_args()
    train_and_evaluate_all(dataset_key=args.dataset)

