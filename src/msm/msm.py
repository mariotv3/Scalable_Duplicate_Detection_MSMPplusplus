# src/msm/msm.py
import re
from collections import defaultdict
from itertools import combinations, product

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from ordered_set import OrderedSet  # if you used OrderedSet before

def generate_small_brand_candidate_pairs(small_brand_offers):
    small_brand_candidates = {}
    all_pairs_flat = []
    for brand, items in small_brand_offers.items():
        offer_ids = [oid for oid, _ in items]
        pairs = []
        if len(offer_ids) >= 2:
            for i, j in combinations(range(len(offer_ids)), 2):
                pair = (offer_ids[i], offer_ids[j])
                pairs.append(pair)
                all_pairs_flat.append(pair)
        small_brand_candidates[brand] = pairs
    return small_brand_candidates, all_pairs_flat


def cosineSim(s1, s2):
    if not s1 and not s2:
        return 0.0
    t1 = s1.lower().split()
    t2 = s2.lower().split()
    vocab = set(t1) | set(t2)
    if not vocab:
        return 0.0
    v1 = np.array([t1.count(tok) for tok in vocab], dtype=float)
    v2 = np.array([t2.count(tok) for tok in vocab], dtype=float)
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom == 0:
        return 0.0
    return float(np.dot(v1, v2) / denom)


def q_gram_similarity(string_1, string_2, q=3):
    s1 = f" {str(string_1).lower()} "
    s2 = f" {str(string_2).lower()} "
    q1 = {s1[i:i + q] for i in range(len(s1) - q + 1)}
    q2 = {s2[i:i + q] for i in range(len(s2) - q + 1)}
    if not q1 and not q2:
        return 0.0
    return len(q1 & q2) / len(q1 | q2)


def levenshtein(a, b):
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        prev = dp[:]
        dp[0] = i
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            dp[j] = min(prev[j] + 1, dp[j - 1] + 1, prev[j - 1] + cost)
    return dp[-1]


def norm_lv(a, b):
    if not a and not b:
        return 0.0
    d = levenshtein(a, b)
    m = max(len(a), len(b))
    if m == 0:
        return 0.0
    return 1 - d / m


def split_numeric(token):
    token = token.lower()
    m = re.match(r"(\d+)(\D*)", token)
    if m:
        return m.group(2), m.group(1)
    m = re.match(r"(\D+)(\d+)", token)
    if m:
        return m.group(1), m.group(2)
    return token, ""


def avg_lv_sim(model_words_1, model_words_2, mw=False):
    if not model_words_1 or not model_words_2:
        return 0.0
    sims = []
    for w1 in model_words_1:
        for w2 in model_words_2:
            if mw:
                w1_non, _ = split_numeric(w1)
                w2_non, _ = split_numeric(w2)
                sims.append(norm_lv(w1_non, w2_non))
            else:
                sims.append(norm_lv(w1, w2))
    return sum(sims) / len(sims)


def get_original_offer(offer_id, data):
    cid, idx_str = offer_id.split("#", 1)
    return data[cid][int(idx_str)]


def same_shop(p1, p2, debug=False):
    return p1.get("shop") == p2.get("shop")


def same_brand(p1, p2, brands, debug=False):
    b1 = "NA"
    b2 = "NA"
    fm1 = p1.get("featuresMap") or {}
    fm2 = p2.get("featuresMap") or {}
    if fm1.get("Brand") is not None:
        b1 = fm1.get("Brand").lower()
    if fm2.get("Brand") is not None:
        b2 = fm2.get("Brand").lower()
    t1 = (p1.get("title") or "").lower()
    t2 = (p2.get("title") or "").lower()
    if b1 == "NA":
        for key in brands:
            if re.search(rf"\b{re.escape(key.lower())}\b", t1):
                b1 = key.lower()
                break
    if b2 == "NA":
        for key in brands:
            if re.search(rf"\b{re.escape(key.lower())}\b", t2):
                b2 = key.lower()
                break
    return b1 == b2 or b1 == "NA" or b2 == "NA"


def same_resolution(p1, p2, debug=False):
    r1 = "NA"
    r2 = "NA"
    regex = r"(?<!\S)\d{3,}\s*[x]\s*\d{3,}(?!\S)"
    fm1 = p1.get("featuresMap") or {}
    fm2 = p2.get("featuresMap") or {}
    for k, v in fm1.items():
        if "resolution" in k.lower():
            m = re.search(regex, str(v))
            if m is not None:
                r1 = m.group(0)
                break
    for k, v in fm2.items():
        if "resolution" in k.lower():
            m = re.search(regex, str(v))
            if m is not None:
                r2 = m.group(0)
                break
    return r1 == r2 or r1 == "NA" or r2 == "NA"


def extract_model_words(features, keys):
    pattern = r"^\d+\.\d+|\b\d+:\d+\b|(?<!\S)\d{3,}\s*[x]\s*\d{3,}(?!\S)"
    model_words = OrderedSet()
    for k in keys:
        if k in features:
            matches = re.findall(pattern, str(features.get(k, "")))
            model_words.update(matches)
    return model_words


def title_comp(t1, t2, alpha, beta, delta, approx):
    title_regex = r"([a-zA-Z0-9]*((\d*\.)?\d+[^0-9, ]+)[a-zA-Z0-9]*)"
    name_cos = cosineSim(t1, t2)
    if name_cos > alpha:
        return 1
    mw1 = OrderedSet(x[0] for x in re.findall(title_regex, t1))
    mw2 = OrderedSet(x[0] for x in re.findall(title_regex, t2))
    similar_mw = False
    for w1 in mw1:
        non1, num1 = split_numeric(w1)
        for w2 in mw2:
            non2, num2 = split_numeric(w2)
            a = norm_lv(non1, non2)
            if a > approx and num1 != num2:
                return -1
            elif a > approx and num1 == num2:
                similar_mw = True
    final_sim = beta * name_cos + (1 - beta) * avg_lv_sim(mw1, mw2, mw=False)
    if similar_mw:
        final_sim = delta * avg_lv_sim(mw1, mw2, mw=True) + (1 - delta) * final_sim
    return final_sim


def msm_pair_dissimilarity(p1, p2, brands, gamma, mu, alpha, debug=False):
    if brands is not None:
        if not same_brand(p1, p2, brands, debug=False):
            return 1.0
    if same_shop(p1, p2, debug=False):
        return 1.0
    if not same_resolution(p1, p2, debug=False):
        return 1.0
    sim = 0.0
    mean_sim = 0.0
    m = 0
    w = 0
    f1 = p1.get("featuresMap") or {}
    f2 = p2.get("featuresMap") or {}
    no_1 = list(f1.keys())
    no_2 = list(f2.keys())
    for k1 in list(f1.keys()):
        for k2 in list(no_2):
            ks = q_gram_similarity(k1, k2, q=3)
            if ks > gamma:
                vs = q_gram_similarity(str(f1.get(k1, "")), str(f2.get(k2, "")), q=3)
                sim += ks * vs
                m += 1
                w += ks
                if k1 in no_1:
                    no_1.remove(k1)
                if k2 in no_2:
                    no_2.remove(k2)
                break
    if w > 0:
        mean_sim = sim / w
    mw1 = extract_model_words(f1, no_1)
    mw2 = extract_model_words(f2, no_2)
    u = len(mw1.union(mw2))
    mw_pct = 0.0 if u == 0 else len(mw1.intersection(mw2)) / u
    t_sim = title_comp(p1.get("title", ""), p2.get("title", ""),
                       alpha=alpha, beta=0.0, delta=0.5, approx=0.5)
    if t_sim == -1:
        theta1 = m / max(1, min(len(f1), len(f2)))
        theta2 = 1 - theta1
        h_sim = theta1 * mean_sim + theta2 * mw_pct
    else:
        theta1 = (1 - mu) * m / max(1, min(len(f1), len(f2)))
        theta2 = 1 - mu - theta1
        h_sim = theta1 * mean_sim + theta2 * mw_pct + mu * t_sim
    return 1.0 - h_sim


def msm_clustering_for_brand(brand, brands, candidate_pairs, data, gamma, epsilon, mu, alpha):
    if not candidate_pairs:
        return [], pd.DataFrame()
    offer_ids = sorted({oid for pair in candidate_pairs for oid in pair})
    n = len(offer_ids)
    dissim = pd.DataFrame(np.ones((n, n), dtype=float), index=offer_ids, columns=offer_ids)
    np.fill_diagonal(dissim.values, 0.0)
    for oid1, oid2 in candidate_pairs:
        p1 = get_original_offer(oid1, data)
        p2 = get_original_offer(oid2, data)
        d = msm_pair_dissimilarity(p1, p2, brands=brands, gamma=gamma, mu=mu, alpha=alpha)
        if d > 0.4:
            d = 1.0
        dissim.loc[oid1, oid2] = d
        dissim.loc[oid2, oid1] = d
    if n == 1:
        return [[offer_ids[0]]], dissim
    clustering = AgglomerativeClustering(
        metric="precomputed",
        linkage="complete",
        distance_threshold=epsilon,
        n_clusters=None,
    )
    clustering.fit(dissim.values)
    labels = clustering.labels_
    clusters_dict = defaultdict(list)
    for oid, lbl in zip(offer_ids, labels):
        clusters_dict[lbl].append(oid)
    clusters = list(clusters_dict.values())
    return clusters, dissim


def msm_for_all_brands(brand_candidates, brands, data, gamma, epsilon, mu, alpha):
    clusters_by_brand = {}
    dissim_by_brand = {}
    for brand, pairs in brand_candidates.items():
        clusters, dissim = msm_clustering_for_brand(
            brand=brand,
            brands=brands,
            candidate_pairs=pairs,
            data=data,
            gamma=gamma,
            epsilon=epsilon,
            mu=mu,
            alpha=alpha,
        )
        clusters_by_brand[brand] = clusters
        dissim_by_brand[brand] = dissim
    return clusters_by_brand, dissim_by_brand


def clusters_to_pairs_by_brand(clusters_by_brand):
    brand_pairs = {}
    all_pairs = set()
    for brand, clusters in clusters_by_brand.items():
        pairs = []
        for cluster in clusters:
            if len(cluster) < 2:
                continue
            cluster = sorted(cluster)
            for i, j in combinations(cluster, 2):
                pair = (i, j)
                pairs.append(pair)
                all_pairs.add(pair)
        brand_pairs[brand] = pairs
    return brand_pairs, all_pairs