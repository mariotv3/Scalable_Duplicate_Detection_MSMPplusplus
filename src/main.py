# src/main.py
import json
import argparse

from preprocessing.cleaning import prepare_datasets
from lsh.minhashing import build_minhash_for_brands 
from evaluation.lsh_eval import tune_lsh_parameters
from lsh.lsh import run_lsh_for_all_brands  
from msm.msm import generate_small_brand_candidate_pairs
from evaluation.msm_eval import tune_msm_params, run_msm_and_evaluate

def main(args):
    with open(args.path) as f:
        raw_data = json.load(f)

    seed = 123
    #testing:
    raw_data = dict(list(raw_data.items())[:400])

    datasets = prepare_datasets(raw_data, seed=3)
    data_train = datasets["data_train"]
    data_test = datasets["data_test"]
    brand_blocks_train = datasets["brand_blocks_train"]
    brand_blocks_test = datasets["brand_blocks_test"]
    known_brands = datasets["known_brands"]

    print("Data cleaned and split correctly")
    print("\nNo clusters in the train set: ", len(data_train))
    NUM_PERM = 128
    # ---------- LSH on train ----------
    brand_signatures_train, small_brand_offers_train, _ = build_minhash_for_brands(brand_blocks_train, num_perm=NUM_PERM, seed=seed)
    print("Minhash done.")

    best_lsh_params, best_lsh_metrics = tune_lsh_parameters(
        brand_signatures=brand_signatures_train,
        data=raw_data,
        num_perm=NUM_PERM,
        max_FC=0.8
    )

    recall = best_lsh_metrics["PC"]
    precision = best_lsh_metrics["PQ"]
    FC = best_lsh_metrics["FC"]

    b = best_lsh_params["b"]
    r = best_lsh_params["r"]
    brand_lsh_candidates_train = run_lsh_for_all_brands(brand_signatures_train, b, r)

    # add small brands (all pairs)
    small_brand_candidates_train, _ = generate_small_brand_candidate_pairs(small_brand_offers_train)
    brand_candidates_train = {**brand_lsh_candidates_train, **small_brand_candidates_train}

    print("\nLSH done.")
    print(f"\nUsing b = {b} and r = {r} the recall = {recall}, the precision = {precision}, and the FC = {FC}")
    train_clusters = set(data_train.keys())

    # ---------- MSM tuning on train ----------
    gamma_grid = [0.2, 0.3, 0.4, 0.5]
    epsilon_grid = [0.2, 0.3, 0.4]
    mu_grid = [0.3, 0.5, 0.7]
    alpha_grid = [0.5, 0.6, 0.7]

    best_msm_params, best_msm_metrics = tune_msm_params(
        brand_candidates=brand_candidates_train,
        data=raw_data,
        brands=known_brands,
        train_cluster_ids=train_clusters,
        gamma_values=gamma_grid,
        epsilon_values=epsilon_grid,
        mu_values=mu_grid,
        alpha_values=alpha_grid,
    )

    print("Best MSM params:", best_msm_params)
    print("Train F1:", best_msm_metrics["F1"])

    # ---------- Run LSH + MSM on TEST with tuned params ----------
    brand_signatures_test, small_brand_offers_test = build_minhash_for_brands(brand_blocks_test)
    brand_lsh_candidates_test = run_lsh_for_all_brands(
        brand_signatures_test,
        b=best_lsh_params["b"],
        r=best_lsh_params["r"],
    )
    small_brand_candidates_test, _ = generate_small_brand_candidate_pairs(small_brand_offers_test)
    brand_candidates_test = {**brand_lsh_candidates_test, **small_brand_candidates_test}

    test_clusters = set(data_test.keys())

    test_metrics = run_msm_and_evaluate(
        brand_candidates=brand_candidates_test,
        data=raw_data,
        gamma=best_msm_params["gamma"],
        epsilon=best_msm_params["epsilon"],
        mu=best_msm_params["mu"],
        alpha=best_msm_params["alpha"],
        brands=known_brands,
        allowed_cluster_ids=test_clusters,
    )

    print("TEST metrics:")
    for k, v in test_metrics.items():
        print(k, ":", v)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="data/TVs-all-merged.json")
    args = parser.parse_args()
    main(args)