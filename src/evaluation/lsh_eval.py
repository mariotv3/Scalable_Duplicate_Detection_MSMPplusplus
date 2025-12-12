import numpy as np
from itertools import combinations


from lsh.lsh import lsh_for_brand_block, factor_pairs


def _model_id(oid, data):
    cid, idx_str = oid.split("#", 1)
    return data[cid][int(idx_str)]["modelID"]


def evaluate_lsh_pairs_for_brand(pairs_offer_ids, offer_ids, data):
    tp = 0
    for o1, o2 in pairs_offer_ids:
        if _model_id(o1, data) == _model_id(o2, data):
            tp += 1

    total_possible = len(offer_ids) * (len(offer_ids) - 1) // 2

    true_overall = 0
    for o1, o2 in combinations(offer_ids, 2):
        if _model_id(o1, data) == _model_id(o2, data):
            true_overall += 1

    return {
        "lsh_true_pairs": tp,              
        "total_possible_pairs": total_possible,
        "true_pairs_overall": true_overall 
    }


def evaluate_lsh_global(brand_signatures, data, b, r):
    total_tp = 0          
    total_true = 0         
    total_cand = 0         
    total_possible = 0     

    total_offers = 0
    for _, (offer_ids, sigs) in brand_signatures.items():
        k, n = sigs.shape
        if n < 2:
            continue
        if k != b * r:
            raise ValueError(f"k={k} but b*r={b*r}")
        pairs, _ = lsh_for_brand_block(offer_ids, sigs, b, r)
        
        total_offers += len(offer_ids)

        m = evaluate_lsh_pairs_for_brand(
            pairs_offer_ids=pairs,
            offer_ids=offer_ids,
            data=data
        )

        total_tp += m["lsh_true_pairs"]
        total_true += m["true_pairs_overall"]
        total_possible += m["total_possible_pairs"]
        total_cand += len(pairs)

    global_total_possible = total_offers * (total_offers-1) // 2
    PC = total_tp / total_true if total_true else 0.0     
    PQ = total_tp / total_cand if total_cand else 0.0    
    FC = total_cand / total_possible if total_possible else 0.0  

    F1_star = 2 * PQ * PC / (PQ + PC) if (PQ + PC) else 0.0

    return {
        "TP": total_tp,
        "total_true": total_true,
        "cand_pairs": total_cand,
        "total_possible_pairs": total_possible,
        "PQ": PQ,
        "PC": PC,
        "FC": FC,
        "F1*": F1_star,
    }


def tune_lsh_parameters(brand_signatures, data, num_perm, min_PC=0.9):
    best_params = (64, 2)
    best_metrics = None
    best_FC = 1.0

    for b, r in factor_pairs(num_perm):
        if b * r != num_perm:
            continue
        m = evaluate_lsh_global(brand_signatures, data, b, r)

        if m["FC"] < best_FC and m["PC"] > min_PC:
            best_FC = m["FC"]
            best_params = (b, r)
            best_metrics = m

    return best_params, best_metrics


def build_brand_lsh_candidates(brand_signatures, b, r):
    brand_lsh_candidates = {}
    brand_lsh_doc_cands = {}

    for brand, (offer_ids, sigs) in brand_signatures.items():
        k, n = sigs.shape
        if n < 2 or k != b * r:
            continue
        pairs, doc_to_cand = lsh_for_brand_block(offer_ids, sigs, b, r)
        brand_lsh_candidates[brand] = pairs
        brand_lsh_doc_cands[brand] = doc_to_cand

    return brand_lsh_candidates, brand_lsh_doc_cands