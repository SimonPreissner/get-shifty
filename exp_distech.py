import numpy as np
from pandas import DataFrame
import utils
import eval_utils
import os
from typing import Dict, List, Tuple
import SpacePair


import exp_unsup




def read_3clusters(filenme:str) -> (List[str], List[str]):
    dis_words = []
    tech_words = []
    with open(filenme, "r") as f:
        lines = f.readlines()
        dis_words.extend(lines[1].rstrip().split())
        dis_words.extend(lines[3].rstrip().split())
        tech_words.extend(lines[5].rstrip().split())

    return dis_words, tech_words

def read_wordlist(filename:str, column:int=0, uniquify=True) -> List[str]:
    """
    Read a word list in one- or multi-column format.
    :param column: choose the column containing the words if it's a multicolumn file
    :param uniquify: delete duplicate words
    """
    woi = []
    with open(filename, "r") as f:
        for line in f:
            line = line.rstrip().split()
            if line:
                woi.append(line[column])
    if uniquify:
        return list(set(woi)) # make sure that each word is in the list only once
    else:
        return woi

def output_dists(sp:SpacePair, PX:np.ndarray, woi:List[str],
                 neighbors:int=10, use_csls:bool=False,
                 out_dir:str=None, list_name:str=""):
    """
    For all given words u, compute the cosine distance between
    Px_u and y_u, and find the nearest neighbors of x_u in X and y_u in Y.
    Writes results to a file if out_dir is specified.
    Prefixes output files with list_name if specified.
    :param sp: SpacePair object holding X and Y as well as the word pairs T
    :param PX: X, projected onto Y (i.e. aligned)
    :param neighbors: number of nearest neighbors to be reported
    :param use_csls: rank nearest neighbors not by cosine, but by CSLS instead
    :param out_dir: needs to end with "/"
    :param list_name: used to distinguish different lists of words of interest
    """

    dists = {}
    for w in woi:
        dists[w] = utils.pairwise_cos_dist(np.array([PX[sp.voc_x[w]]]),
                                           np.array([sp.Y[sp.voc_y[w]]]),
                                           no_zero_dists=False)[0]

    dist_ranked_words = sorted(dists, key=dists.get, reverse=True)

    # just some printouts
    top = 10
    print(f"\nDistances (top {top} {list_name} words):\n"
          f"{'dcos(Px_w,y_w)'} {'word':<12}")
    for w in dist_ranked_words[:top]:
        print(f"{dists[w]:<13.5f}  {w:<12}")


    print(f"\nFinding the {neighbors} nearest neighbors in each space")
    src_nbs = exp_unsup.find_closest_concepts(sp.X[[sp.voc_x[w] for w in woi]],
                                              sp.X, sp.voc_x,
                                              k=neighbors, csls=use_csls)
    trg_nbs = exp_unsup.find_closest_concepts(sp.Y[[sp.voc_y[w] for w in woi]],
                                              sp.Y, sp.voc_y,
                                              k=neighbors, csls=use_csls)


    if out_dir is not None:
        if not os.path.isdir(out_dir): os.makedirs(out_dir)

        filepath = out_dir + list_name + "_dists.tsv"
        print(f"writing pair distances to {filepath}...\n")
        df = DataFrame({"distance":[dists[w] for w in dist_ranked_words],
                        "word": dist_ranked_words,
                        "src_neighbors":src_nbs,
                        "trg_neighbors":trg_nbs})
        df.to_csv(filepath, sep='\t')





def run(sp:SpacePair, PX:np.ndarray, woi:List[str], list_name:str, ap_source:bool=False,
        out_dir:str=None,
        min_count:int=1, spaces_mincount:int=0,
        dist_nbs:int=10, dir_k:int=1, pairdist_csls:bool=False,
        options:utils.ConfigReader=None):
    """
    This is similar to exp_unsup.py, but it works on the basis of words of interest,
    which might not appear in the space pair's vocabularies.
    :param sp: SpacePair (holds X, U, Y, V, and T)
    :param PX: projected space (using SpacePair.P to project X onto Y)
    :param woi: list of words of interest.
    :param ap_source: either cluster source vectors (True) or cluster difference vectors (False)
    :param out_dir: output destination (creates multiple files)
    :param min_count: usually 5, 10, 15 -- require min. occurrence from pairs' words
    :param spaces_mincount: usually 1, 5, 10 -- require min. occ. of embeddings' words
    :param dir_k: number of NNs to a centroid
    :param pairdist_csls: use csls for nearest neighbors search (in the pairdists part)
    :param signal: list of translation pair (in the monolingual scenario: just word pairs)
    :param options: use this for more convenient parameter passing. expects to
     hold min_count, spaces_mincount, dist_nbs, dir_k, and pairdist_csls.
    """

    if options is not None:
        if options("exp_distech_min_wordcount") is not None:
            min_count = options("exp_distech_min_wordcount")
        if options("exp_distech_spaces_mincount") is not None:
            spaces_mincount = options("exp_distech_spaces_mincount")
        if options("exp_distech_neighbors") is not None:
            dist_nbs = options("exp_distech_neighbors")
        if options("exp_distech_clusterlabels") is not None:
            dir_k = options("exp_distech_clusterlabels")
        if options("exp_distech_use_csls") is not None:
            pairdist_csls = options("exp_distech_use_csls")


    # 1. 'prune' the word pairs to throw out unreliable embeddings
    woi, sp = exp_unsup.reduce_bilingual_signal([(w,w) for w in woi], sp,
                                               min_count=min_count,
                                               spaces_mincount=spaces_mincount) # this is to save memory
    woi = [p[0] for p in woi]
    print(f"Reduced vocabulary to words with min. {min_count} corpus occurrences. "
          f"Continuing with {len(woi)} pairs.")
    if spaces_mincount>0:
        print(f"Also reduced spaces to concepts with {spaces_mincount} corpus "
              f"occuccences in order to save memory. new sizes (X/Y): "
              f"{sp.X.shape[0]}/{sp.Y.shape[0]}.")


    # 2. Calculate difference vectors
    from_vecs = np.array([  PX[sp.voc_x[u]] for u in woi])
    to_vecs   = np.array([sp.Y[sp.voc_y[v]] for v in woi])
    D = utils.shift_directions(from_vecs, to_vecs, norm=False) # norm the vectors, just to be safe.
    D = exp_unsup.normalize_shifts_by_frequency(D, [(w,w) for w in woi], sp.freq_x, sp.freq_y)
    voc_D = {w:i for i,w in enumerate(woi)}


    cluster_timer = utils.Timer()
    print(f"\nStarting shift detection on '{list_name}' words.")

    # Distances between Px and Y (no shift directions involved)
    output_dists(sp, PX, woi,
                 neighbors=dist_nbs, use_csls=pairdist_csls,
                 out_dir=out_dir, list_name=list_name)
    cluster_timer("distances")


    # 3.2 Clustering of difference vetors
    selected_D = D[[voc_D[w] for w in woi]]
    ind_sD = {i: w for i, w in enumerate(woi)}

    # value of labels = index in center_ids
    # index of labels = key in ind_sD
    # value of center_ids = key in ind_sD
    # index of center_ids = cluster label (= values of labels)
    if ap_source is True:
        print("Clustering source vectors ...")
        labels, center_ids, convergence_it = utils.affprop_clusters(np.array(sp.X[[sp.voc_x[u] for u in woi]]))
    else:
        print("Clustering shift vectors ...")
        labels, center_ids, convergence_it = utils.affprop_clusters(selected_D)
    cluster_timer("AP_clustering")

    # make arrays of vectors and lists of word pairs which belong to the same cluster
    clusters, cluster_words = exp_unsup.reorganize(labels, selected_D, ind_sD)
    cluster_timer("re-organization")

    # 3.3 Cluster sizes, cluster lengths, inner distances
    cluster_sizes = [len(c) for c in clusters]

    lengths_max =     [exp_unsup.cluster_length(cluster, "max") for cluster in clusters]
    lengths_mean =    [exp_unsup.cluster_length(cluster, "mean") for cluster in clusters]
    lengths_median = [exp_unsup.cluster_length(cluster, "median") for cluster in clusters]
    lengths_std = [exp_unsup.cluster_length(cluster, "std") for cluster in clusters]
    inner_dists = [exp_unsup.inner_distance(cluster) for cluster in clusters]

    lengths_max_normed    = eval_utils.z_scores(lengths_max)
    lengths_mean_normed   = eval_utils.z_scores(lengths_mean)
    lengths_median_normed = eval_utils.z_scores(lengths_median)
    lengths_std_normed    = eval_utils.z_scores(lengths_std)
    inner_dists_normed    = eval_utils.z_scores(inner_dists)

    cluster_timer("lengths_and_inner_dist")

    del clusters  # to save memory

    # Nearest Neighbors: find the vector(s) in Y most similar to a cluster's centroid
    direction_labels = exp_unsup.find_closest_concepts(selected_D[center_ids], sp.Y, sp.voc_y, k=dir_k)
    cluster_timer("closest_concepts")

    # report everything
    df = DataFrame({
        "max_length" :       lengths_max,
        "mean_length":       lengths_mean,
        "median_length":     lengths_median,
        "std_length":        lengths_std,
        "max_length_zscore": lengths_max_normed,
        "mean_length_zscore": lengths_mean_normed,
        "median_length_zscore": lengths_median_normed,
        "std_length_zscore": lengths_std_normed,
        "inner_distance" :   inner_dists,
        "inner_dist_zscore": inner_dists_normed, # normalized among all clusters
        "cluster_size" :     cluster_sizes,
        "centroid" :         [ind_sD[center] for center in center_ids], # these are word pairs
        "direction_label" :  direction_labels, # these are tuples (word, distance)
        "cluster_words" :    cluster_words
    })
    # rank by shift size
    df = df.sort_values("inner_distance", ascending=False, ignore_index=True)
    df.to_csv(out_dir+list_name+"_shift_clusters.tsv", sep='\t')

    cluster_timer.total()

    # additional information
    with open(out_dir+list_name+"_clustering_stats", "w") as f:
        f.write(f"number_of_clusters\t{len(center_ids)}\n"
                f"convergence_criterion\t{convergence_it}\n"
                f"param_min_count\t{min_count}\n"
                f"param_dir_k\t{dir_k}\n"
                f"param_reduce_spaces\t{spaces_mincount}\n"
                f"size_X\t{sp.X.shape[0]}\n"
                f"size_Y\t{sp.Y.shape[0]}\n"
                f"words_of_interest\t{len(woi)}\n"
                f"\ntime_taken:\n{cluster_timer}")


