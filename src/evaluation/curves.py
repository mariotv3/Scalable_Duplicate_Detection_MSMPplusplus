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
    test_clusters,
    msm_params: Dict[str, float],
    num_perm: int,
    max_delta: int = 2,
) -> List[Dict[str, Any]]:


    rows: List[Dict[str, Any]] = []

    k_candidates = range(num_perm - max_delta, num_perm + 1)

    for k_eff in k_candidates:
        if k_eff <= 0:
            continue
        if k_eff > num_perm:
            continue

        for b, r in factor_pairs(k_eff):
            if b * r != k_eff:
                continue

            brand_sigs_k: Dict[str, Tuple[List[str], np.ndarray]] = {}
            for brand, (offer_ids, sigs) in brand_signatures_test.items():
                if sigs.shape[0] < k_eff:
                    continue
                brand_sigs_k[brand] = (offer_ids, sigs[:k_eff, :])

            if not brand_sigs_k:
                continue

            lsh_metrics = evaluate_lsh_global(
                brand_signatures=brand_sigs_k,
                data=cleaned_data,
                b=b,
                r=r,
            )
            
            PQ = lsh_metrics["PQ"]
            PC = lsh_metrics["PC"]
            F1_star = lsh_metrics["F1*"]

        
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

            total_cand = sum(len(pairs) for pairs in brand_candidates.values())

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

            msm_metrics = run_msm_and_evaluate(
                brand_candidates=brand_candidates,
                data=cleaned_data,
                gamma=msm_params["gamma"],
                epsilon=msm_params["epsilon"],
                mu=msm_params["mu"],
                alpha=msm_params["alpha"],
                beta=msm_params["beta"],
                delta=msm_params["delta"],
                eta=msm_params["eta"],
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

    full_brand_candidates: Dict[str, List[Tuple[str, str]]] = {}


    for brand, (offer_ids, _) in brand_signatures_test.items():
        n = len(offer_ids)
        if n < 2:
            continue
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append((offer_ids[i], offer_ids[j]))
        full_brand_candidates[brand] = pairs


    for brand, items in small_brand_offers_test.items():
        offer_ids = [oid for oid, _ in items]
        n = len(offer_ids)
        if n < 2:
            continue
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append((offer_ids[i], offer_ids[j]))
        if brand in full_brand_candidates:
            full_brand_candidates[brand].extend(pairs)
        else:
            full_brand_candidates[brand] = pairs

    msm_metrics_full = run_msm_and_evaluate(
        brand_candidates=full_brand_candidates,
        data=cleaned_data,
        gamma=msm_params["gamma"],
        epsilon=msm_params["epsilon"],
        mu=msm_params["mu"],
        alpha=msm_params["alpha"],
        beta=msm_params["beta"],
        delta=msm_params["delta"],
        eta=msm_params["eta"],
        allowed_cluster_ids=test_clusters,
    )

    rows.append(
        {
            "k_eff": None, 
            "b": None,
            "r": None,
            "FC": 1.0,
            "PQ": 0.0001,
            "PC": 1.0,
            "F1*": 0.0001,
            "F1": msm_metrics_full["F1"],
        }
    )
    return rows