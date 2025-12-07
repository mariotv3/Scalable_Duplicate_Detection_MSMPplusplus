# src/evaluation/msm_eval.py
from collections import defaultdict
from itertools import product

from msm.msm import msm_for_all_brands, clusters_to_pairs_by_brand  # adjust import path


def get_model_id(offer_id, data):
    cid, idx = offer_id.split("#", 1)
    return data[cid][int(idx)]["modelID"]


def evaluate_predicted_pairs(predicted_pairs, data, allowed_cluster_ids=None):
    model_by_offer = {}
    offers_in_scope = set()

    if allowed_cluster_ids is not None:
        for cid in allowed_cluster_ids:
            if cid not in data:
                continue
            for idx, offer in enumerate(data[cid]):
                oid = f"{cid}#{idx}"
                offers_in_scope.add(oid)
                model_by_offer[oid] = offer["modelID"]
    else:
        for cid, offers in data.items():
            for idx, offer in enumerate(offers):
                oid = f"{cid}#{idx}"
                offers_in_scope.add(oid)
                model_by_offer[oid] = offer["modelID"]

    pred_set = set()
    for oid1, oid2 in predicted_pairs:
        if oid1 in offers_in_scope and oid2 in offers_in_scope:
            a, b = sorted((oid1, oid2))
            pred_set.add((a, b))

    model_to_offers = defaultdict(list)
    for oid, mid in model_by_offer.items():
        model_to_offers[mid].append(oid)

    true_pairs = set()
    for oids in model_to_offers.values():
        if len(oids) >= 2:
            oids = sorted(oids)
            for i in range(len(oids)):
                for j in range(i + 1, len(oids)):
                    true_pairs.add((oids[i], oids[j]))

    TP_set = pred_set & true_pairs
    FP_set = pred_set - true_pairs
    FN_set = true_pairs - pred_set

    TP = len(TP_set)
    FP = len(FP_set)
    FN = len(FN_set)

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    F1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    PQ = precision
    PC = recall
    F1_star = F1

    return {
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "n_true_pairs": len(true_pairs),
        "n_pred_pairs": len(pred_set),
        "precision": precision,
        "recall": recall,
        "F1": F1,
        "PQ": PQ,
        "PC": PC,
        "F1*": F1_star,
    }


def run_msm_and_evaluate(brand_candidates, data, gamma, epsilon, mu, alpha, brands, allowed_cluster_ids=None):
    clusters_by_brand, _ = msm_for_all_brands(
        brand_candidates=brand_candidates,
        brands=brands,
        data=data,
        gamma=gamma,
        epsilon=epsilon,
        mu=mu,
        alpha=alpha,
    )
    _, all_pairs_pred = clusters_to_pairs_by_brand(clusters_by_brand)
    metrics = evaluate_predicted_pairs(
        predicted_pairs=all_pairs_pred,
        data=data,
        allowed_cluster_ids=allowed_cluster_ids,
    )
    return metrics


def tune_msm_params(
    brand_candidates,
    data,
    brands,
    train_cluster_ids=None,
    gamma_values=None,
    epsilon_values=None,
    mu_values=None,
    alpha_values=None,
):
    if gamma_values is None:
        gamma_values = [0.3, 0.5, 0.7]
    if epsilon_values is None:
        epsilon_values = [0.2, 0.3, 0.4]
    if mu_values is None:
        mu_values = [0.3, 0.5, 0.7]
    if alpha_values is None:
        alpha_values = [0.5, 0.6, 0.7]

    best_F1 = -1.0
    best_params = None
    best_metrics = None

    for gamma, epsilon, mu, alpha in product(gamma_values, epsilon_values, mu_values, alpha_values):
        metrics = run_msm_and_evaluate(
            brand_candidates=brand_candidates,
            data=data,
            gamma=gamma,
            epsilon=epsilon,
            mu=mu,
            alpha=alpha,
            brands=brands,
            allowed_cluster_ids=train_cluster_ids,
        )
        F1 = metrics["F1"]
        print(
            f"gamma={gamma}, epsilon={epsilon}, mu={mu}, alpha={alpha} "
            f"-> F1={F1:.4f}, precision={metrics['precision']:.4f}, recall={metrics['recall']:.4f}"
        )
        if F1 > best_F1:
            best_F1 = F1
            best_params = {"gamma": gamma, "epsilon": epsilon, "mu": mu, "alpha": alpha}
            best_metrics = metrics

    return best_params, best_metrics