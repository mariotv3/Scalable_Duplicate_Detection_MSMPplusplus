import optuna
from collections import defaultdict
from itertools import product
from optuna.samplers import TPESampler

from msm.msm import msm_for_all_brands, clusters_to_pairs_by_brand


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
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    F1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0


    return {
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "n_true_pairs": len(true_pairs),
        "n_pred_pairs": len(pred_set),
        "precision": precision,
        "recall": recall,
        "F1": F1
    }

def run_msm_and_evaluate(brand_candidates, data, gamma, epsilon, mu, alpha, beta, delta, eta, allowed_cluster_ids=None):
    clusters_by_brand, _ = msm_for_all_brands(
        brand_candidates=brand_candidates,
        beta=beta,
        delta=delta,
        eta=eta,
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
    train_cluster_ids=None,
    n_trials=30,
    timeout=None,
    gamma_range=(0.2, 0.8),
    epsilon_range=(0.1, 0.5),
    mu_range=(0.3, 0.9),
    alpha_range=(0.4, 0.9),
    beta_range=(0.0,0.2),
    eta_range=(0.3,0.7),
    delta_range=(0.3,0.7),
    seed: int | None = None,
):

    sampler = optuna.samplers.TPESampler(seed=seed)

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name="msm_hparam_tuning",
    )

    def objective(trial: optuna.Trial) -> float:
        gamma = trial.suggest_float("gamma", gamma_range[0], gamma_range[1])
        epsilon = trial.suggest_float("epsilon", epsilon_range[0], epsilon_range[1])
        mu = trial.suggest_float("mu", mu_range[0], mu_range[1])
        alpha = trial.suggest_float("alpha", alpha_range[0], alpha_range[1])
        beta = trial.suggest_float("beta", beta_range[0], beta_range[1])
        delta = trial.suggest_float("delta", delta_range[0],delta_range[1])
        eta = trial.suggest_float("eta", eta_range[0], eta_range[1])
        
        metrics = run_msm_and_evaluate(
            brand_candidates=brand_candidates,
            data=data,
            gamma=gamma,
            epsilon=epsilon,
            mu=mu,
            alpha=alpha,
            beta=beta,
            delta=delta,
            eta=eta,
            allowed_cluster_ids=train_cluster_ids,
        )

        
        trial.set_user_attr("metrics", metrics)

        return metrics["F1"] 

    study.optimize(objective, n_trials=n_trials, timeout=timeout, n_jobs=1)

    best_trial = study.best_trial
    best_params = {
        "gamma": best_trial.params["gamma"],
        "epsilon": best_trial.params["epsilon"],
        "mu": best_trial.params["mu"],
        "alpha": best_trial.params["alpha"],
        "beta":best_trial.params["beta"],
        "eta":best_trial.params["eta"],
        "delta":best_trial.params["delta"]
    }
    best_metrics = best_trial.user_attrs["metrics"]

    return best_params, best_metrics