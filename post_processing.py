"""
This script adds information after having run the experiments.
1. for each condition, it creates dataframe with a baseline of random clusters,
   using the information from the clustering in the experiments.
2. (optional) for each condition, it re-creates the clusters and calculates each clusters'
   standard deviation of shift vector length. The outputs are updated with this info.
"""


import utils
import eval_utils
import numpy as np
from SpacePair import SpacePair
import exp_unsup

from pandas import DataFrame

from typing import List, Dict, Tuple


def make_random_clusters(cluster_sizes: List[int]) -> List[int]:
    """
    Assign cluster labels randomly
    :param cluster_sizes: list where each index is the cluster label
    :return: list with an assigned cluster label per point
    """
    labels = list(range(len(cluster_sizes)))  # to iterate over clusters
    vector_indices = list(range(sum(cluster_sizes)))  # this will be consumed
    result = list(np.zeros(len(vector_indices)))

    for label in labels:
        c_size = cluster_sizes[label]
        for i in range(c_size):  # for as many points as the cluster of this label has,
            v_index = vector_indices.pop(np.random.randint(len(vector_indices)))  # randomly draw a vector index
            result[v_index] = label  # and assign the label to this index
    return result


def make_cluster_baseline(cluster_dataframe):
    """
    Use the information about cluster sizes and number of clusters to make a
    dataframe of random clusters. The distribution of cluster sizes follows the
    standard normal distribution.
    :param cluster_dataframe: an output from one of the experiments
    :return: DataFrame with measurements, such as the one that was  passed.
    """

    # make random clusters like the AP ones and crop them
    r_cluster_sizes = np.random.normal(loc=np.mean(cluster_dataframe["cluster_size"]),
                                       scale=np.std(cluster_dataframe["cluster_size"]),
                                       size=len(cluster_dataframe))
    # positive integer cluster sizes, minimum 2
    r_cluster_sizes = [int(min(np.round(abs(s)), 2)) for s in r_cluster_sizes]
    # adjust cluster sizes to fit the number of shift vectors
    overload = int(sum(r_cluster_sizes) - sum(cluster_dataframe["cluster_size"]))
    n_clusters = len(r_cluster_sizes)
    if overload > 0:  # crop clusters
        while overload > 0:
            idx = np.random.randint(n_clusters)
            if r_cluster_sizes[idx] > 2:  # don't allow clusters of size 1 or 0
                r_cluster_sizes[idx] -= 1
                overload -= 1
    elif overload < 0:  # pad clusters
        for i in range(-overload):
            idx = np.random.randint(n_clusters)
            r_cluster_sizes[idx] += 1

    r_labels = make_random_clusters(r_cluster_sizes)
    r_clusters, r_cluster_words = exp_unsup.reorganize(r_labels, D, ind_D)

    r_lengths_max = [exp_unsup.cluster_length(cluster, "max") for cluster in r_clusters]
    r_lengths_mean = [exp_unsup.cluster_length(cluster, "mean") for cluster in r_clusters]
    r_lengths_median = [exp_unsup.cluster_length(cluster, "median") for cluster in r_clusters]
    r_lengths_std = [exp_unsup.cluster_length(cluster, "std") for cluster in r_clusters]
    r_inner_dists = [exp_unsup.inner_distance(cluster) for cluster in r_clusters]

    r_lengths_max_normed = eval_utils.z_scores(r_lengths_max)
    r_lengths_mean_normed = eval_utils.z_scores(r_lengths_mean)
    r_lengths_median_normed = eval_utils.z_scores(r_lengths_median)
    r_inner_dists_normed = eval_utils.z_scores(r_inner_dists)

    df = DataFrame({
        "max_length": r_lengths_max,
        "mean_length": r_lengths_mean,
        "median_length": r_lengths_median,
        "std_length": r_lengths_std,
        "max_length_zscore": r_lengths_max_normed,
        "mean_length_zscore": r_lengths_mean_normed,
        "median_length_zscore": r_lengths_median_normed,
        "inner_distance": r_inner_dists,
        "inner_dist_zscore": r_inner_dists_normed,  # normalized among all clusters
        "cluster_size": r_cluster_sizes
    })

    return df



if __name__ == '__main__':

    d1 = "outputs/shift_experiments_apshifts/"
    d2 = "outputs/shift_experiments_apsource/"
    d3 = "outputs/shift_experiments_noalign_apshifts/"
    d4 = "outputs/shift_experiments_noalign_apsource/"

    y1 = "1740_1770/"
    y2 = "1860_1890/"

    e1 = "unsup_bi/"
    e2 = "unsup_mono/"
    e3 = "dis_tech/"

    s1 = "all"
    s2 = "all_discourse"
    s3 = "all_technical"

              # unsup_bi        # unsup_mono      # discourse       technical
    combos = [(d1, y1, e1, s1), (d1, y1, e2, s1), (d1, y1, e3, s2), (d1, y1, e3, s3),  # 1740 APshifts
              (d2, y1, e1, s1), (d2, y1, e2, s1), (d2, y1, e3, s2), (d2, y1, e3, s3),  # 1740 APfirst

              (d1, y2, e1, s1), (d1, y2, e2, s1), (d1, y2, e3, s2), (d1, y2, e3, s3),  # 1860 APshifts
              (d2, y2, e1, s1), (d2, y2, e2, s1), (d2, y2, e3, s2), (d2, y2, e3, s3),  # 1860 APfirst

                                (d3, y2, e2, s1), (d3, y2, e3, s2), (d3, y2, e3, s3),  # noalign APshifts
                                (d4, y2, e2, s1), (d4, y2, e3, s2), (d4, y2, e3, s3)]  # noalign APfirst


    for round, (d, y, e, s) in enumerate(combos):
        print(f"\n\nROUND {round+1} OF {len(combos)}\n\n\n")

        tuples = True if e==e1 or e==e2 else False   # true for unsup_bi and unsup_mono
        stats, df_dist, df_clust = eval_utils.read_results(d + y + e, s, e, tuples=tuples, with_baseline=False)

        # load space pair and project X
        sp = SpacePair.from_config(d+y+"spacepair.cfg", init_all=True)
        PX = np.array([sp.P.dot(x) for x in sp.X])

        # extract word pairs from df_dist and make difference vectors
        try:
            pairs = [(w1, w2) for w1, w2 in zip(df_dist["src_word"].tolist(), df_dist["trg_word"].tolist())]
        except KeyError:
            pairs = [(w, w) for w in df_dist["word"].tolist()] # for the dis_tech experiment

        from_vecs = np.array([  PX[sp.voc_x[u]] for u,_ in pairs])
        to_vecs   = np.array([sp.Y[sp.voc_y[v]] for _,v in pairs])
        D = utils.shift_directions(from_vecs, to_vecs, norm=False) # norm the vectors
        D = exp_unsup.normalize_shifts_by_frequency(D, pairs, sp.freq_x, sp.freq_y)
        voc_D = {p: i for i, p in enumerate(pairs)}
        ind_D = {i: p for i, p in enumerate(pairs)}


        # use the info from previous clustering
        baseline_df = make_cluster_baseline(df_clust)
        baseline_df = baseline_df.sort_values("inner_distance", ascending=False, ignore_index=True)
        out_filepath = d + y + e + s + "_clustering_baseline.tsv"
        print("saving DataFrame to", out_filepath)
        baseline_df.to_csv(out_filepath, sep='\t')

        del sp
        del PX


        """ OPTIONAL: re-create clusters in order to add further measurements."""
        add_info = False
        if add_info:
            # The center_ids are not the same as previously, because the difference vectors are sorted differently.
            # However, the clustering is the same (it's just where the vectors are in the array that's different).
            try:
                center_ids = [voc_D[centroid] for centroid in df_clust["centroid"]]
            except KeyError:
                center_ids = [voc_D[(c,c)] for c in df_clust["centroid"]] # for the dis_tech experiment

            labels = np.zeros(len(pairs))
            for i, label in enumerate(center_ids):  # iterate over clusters
                for member in df_clust.iloc[i]["cluster_words"]:  # treat each member of a cluster
                    try:
                        labels[voc_D[member]] = label
                    except KeyError:
                        labels[voc_D[(member,member)]] = label # for the dis_tech experiment

            clusters, cluster_words = exp_unsup.reorganize(labels, D, ind_D)

            # calculate standard deviation of length per cluster, add that information to the
            # DataFrame, and overwrite the old one with this updated version
            lengths_std = [exp_unsup.cluster_length(cluster, "std") for cluster in clusters]
            df_clust["std_length"] = lengths_std

            out_filepath = d+y+e+s+"_shift_clusters.tsv"
            print("saving DataFrame to", out_filepath)
            df_clust.to_csv(out_filepath, sep='\t')


            del D
            del clusters


print("done.")
