# src/evaluation/curves.py

from __future__ import annotations

from typing import Dict, List, Tuple, Any

import numpy as np

from lsh.lsh import run_lsh_for_all_brands, factor_pairs
from evaluation.lsh_eval import evaluate_lsh_global
from msm.msm import generate_small_brand_candidate_pairs
from evaluation.msm_eval import run_msm_and_evaluate


def eval_full_model_for_lsh_configs(
    brand_signatures_test: Dict[str, Tuple[List[str], np.ndarray]],
    small_brand_offers_test: Dict[str, List[Tuple[str, dict]]],
    cleaned_data: Dict[str, List[dict]],
    known_brands,
    test_clusters,
    msm_params: Dict[str, float],
    num_perm: int,
    max_delta: int = 2,
) -> List[Dict[str, Any]]:


    rows: List[Dict[str, Any]] = []

    # vary k_eff between num_perm - max_delta and num_perm + max_delta,
    # but LSH only uses as many rows as we actually give it in signatures.
    # For k_eff < num_perm we just slice the first k_eff rows.
    # (For k_eff > num_perm we skip, because we don't have that many rows.)
    k_candidates = range(num_perm - max_delta, num_perm + 1)

    for k_eff in k_candidates:
        if k_eff <= 0:
            continue
        if k_eff > num_perm:
            continue

        for b, r in factor_pairs(k_eff):
            if b * r != k_eff:
                continue

            # --- Build a k_eff-sliced version of brand_signatures ---
            brand_sigs_k: Dict[str, Tuple[List[str], np.ndarray]] = {}
            for brand, (offer_ids, sigs) in brand_signatures_test.items():
                if sigs.shape[0] < k_eff:
                    continue
                brand_sigs_k[brand] = (offer_ids, sigs[:k_eff, :])

            if not brand_sigs_k:
                continue

            # --- LSH evaluation for this config ---
            lsh_metrics = evaluate_lsh_global(
                brand_signatures=brand_sigs_k,
                data=cleaned_data,
                b=b,
                r=r,
            )
            
            PQ = lsh_metrics["PQ"]
            PC = lsh_metrics["PC"]
            F1_star = lsh_metrics["F1*"]

            # --- Build candidate pairs for this LSH config ---
            brand_lsh_candidates = run_lsh_for_all_brands(
                brand_sigs_k,
                b=b,
                r=r,
            )
            small_brand_candidates, _ = generate_small_brand_candidate_pairs(
                small_brand_offers_test
            )
            brand_candidates = {
                **brand_lsh_candidates,
                **small_brand_candidates,
            }

            # # numerator: all candidate pairs across all brands
            total_cand = sum(len(pairs) for pairs in brand_candidates.values())

            # denominator: all possible pairs per brand, summed
            total_possible = 0

            for brand, (offer_ids, _) in brand_sigs_k.items():
                n = len(offer_ids)
                if n >= 2:
                    total_possible += n * (n - 1) // 2
            
            for brand, items in small_brand_offers_test.items():
                n = len(items)
                if n >= 2:
                    total_possible += n * (n - 1) // 2

            FC = total_cand / total_possible if total_possible > 0 else 0.0

            # --- Run MSM with *fixed* params on these candidates ---
            msm_metrics = run_msm_and_evaluate(
                brand_candidates=brand_candidates,
                data=cleaned_data,
                gamma=msm_params["gamma"],
                epsilon=msm_params["epsilon"],
                mu=msm_params["mu"],
                alpha=msm_params["alpha"],
                brands=known_brands,
                allowed_cluster_ids=test_clusters,
            )
            F1_full = msm_metrics["F1"]

            rows.append(
                {
                    "k_eff": k_eff,
                    "b": b,
                    "r": r,
                    "FC": FC,
                    "PQ": PQ,
                    "PC": PC,
                    "F1*": F1_star,
                    "F1": F1_full,
                }
            )

    return rows