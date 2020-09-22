"""
the core part of this experiment is to first take differences between times in
order to obtain conceptual shifts and then cluster these shift vectors. This way
we can observe
- whether certain shifts belong to a shared topic
-
"""


import numpy as np
from pandas import DataFrame
import utils
import eval_utils
import os
import math
from typing import Dict, List, Tuple
import SpacePair


def output_pairdists(sp:SpacePair, PX:np.ndarray,
                     word_pairs:List[Tuple[str,str]]=None,
                     neighbors:int=10, use_csls:bool=False,
                     out_dir:str=None, partition_name:str=""):
    """
    For all given word pairs (u,v), compute the cosine distance between
    Px_u and y_v, and find the nearest neighbors of x_u in X and y_v in Y.
    Writes results to a file if out_dir is specified.
    Prefixes output files with partition_name if specified.
    Uses word_pairs instead of SpacePair.T if specified.
    :param sp: SpacePair object holding X and Y as well as the word pairs T
    :param PX: X, projected onto Y (i.e. aligned)
    :param neighbors: number of nearest neighbors to be reported
    :param use_csls: rank nearest neighbors not by cosine, but by CSLS instead
    :param out_dir: needs to end with "/"
    :param partition_name: used to distinguish matching/mismatching word pairs
    """
    if word_pairs is None:
        word_pairs = sorted(sp.T, key=sp.T.get, reverse=True)

    pairdists = {}
    for u,v in word_pairs:
        pairdists[(u,v)] = utils.pairwise_cos_dist(np.array([PX[sp.voc_x[u]]]),
                                                   np.array([sp.Y[sp.voc_y[v]]]))[0]

    dist_ranked_pairs = sorted(pairdists, key=pairdists.get, reverse=True)

    # just some printouts
    top = 10
    print(f"\nPair Distances (top {top} {partition_name}):\n"
          f"{'dcos(Px_u,y_v)'} {'word_u':<12}  {'word_v':<12}")
    for u,v in dist_ranked_pairs[:top]:
        print(f"{pairdists[(u,v)]:<13.5f}  {u:<12}  {v:<12}")


    print(f"\nFinding the {neighbors} nearest neighbors for each u and v...")
    U, V = zip(*word_pairs)
    src_nbs = find_closest_concepts(sp.X[[sp.voc_x[u] for u in U]], sp.X, sp.voc_x, k=neighbors, csls=use_csls)
    trg_nbs = find_closest_concepts(sp.Y[[sp.voc_y[v] for v in V]], sp.Y, sp.voc_y, k=neighbors, csls=use_csls)


    if out_dir is not None:
        if not os.path.isdir(out_dir): os.makedirs(out_dir)

        filepath = out_dir + partition_name + "_pairdists.tsv"
        print(f"writing pair distances to {filepath}...\n")
        df = DataFrame({"distance":[pairdists[pair] for pair in dist_ranked_pairs],
                        "src_word": [pair[0] for pair in dist_ranked_pairs],
                        "trg_word": [pair[1] for pair in dist_ranked_pairs],
                        "src_neighbors":src_nbs,
                        "trg_neighbors":trg_nbs})
        df.to_csv(filepath, sep='\t')

def reduce_bilingual_signal(pairs:List[Tuple[str,str]],
                            spacepair:SpacePair,
                            min_count:int=1,
                            spaces_mincount:int=0):
    """
    Reduce the list of word pairs (and the embedding spaces) to those words which
    occur at least min_count (and spaces_mincount) times in the underlying corpus.
    This function uses an approximation method to check whether the frequencies in
    the SpacePair objects have been log-flattened; it can thus only handle raw or
    log-flattened frequencies. Returns the word pairs and, if altered, the SpacePair.
    :param min_count: usual values: 5, 10, 20
    :param spaces_mincount: usual values: 1, 3, 5, 10
    :return: List[Tuple[str,str]], optionally: SpacePair
    """

    # reconstruct whether frequencies have been flattened
    flattening_applies = True
    overlap = list(set([p[0] for p in pairs]).intersection(set(spacepair.freq_x.keys())))
    for i in range(min(len(overlap),10)):
        reconstructed_freq = math.e**spacepair.freq_x[overlap[i]]
        if not np.isclose(reconstructed_freq, round(reconstructed_freq), atol=1e-5):
            flattening_applies = False
    if flattening_applies:
        min_count = np.log(min_count)

    src_words = [w for w,f in spacepair.freq_x.items() if f >= min_count]
    trg_words = [w for w,f in spacepair.freq_y.items() if f >= min_count]
    reduced_signal = [pair for pair in pairs if pair[0] in src_words and pair[1] in trg_words]

    if spaces_mincount > 0:
        if flattening_applies:
            spaces_mincount = np.log(spaces_mincount)
        src_words = [w for w, f in spacepair.freq_x.items() if f >= spaces_mincount]
        trg_words = [w for w, f in spacepair.freq_y.items() if f >= spaces_mincount]
        spacepair.X, spacepair.voc_x = utils.select_subspace(spacepair.X, spacepair.voc_x, src_words)
        spacepair.Y, spacepair.voc_y = utils.select_subspace(spacepair.Y, spacepair.voc_y, trg_words)
        return reduced_signal, spacepair
    else:
        return reduced_signal

def reorganize(labels:List[int], selected_D:np.ndarray,
               ind_sD:Dict[int,str])->(List[np.ndarray], List[List[str]]):
    """
    From the information about which embedding belongs to which cluster, create
    one 2D-array of embeddings per cluster and a list of lists with the words
    instead of the embeddings. Return both structures.
    :param labels: Cluster labels (length: m)
    :param selected_D: Embeddings (shape: (m,d))
    :param ind_sD: mapping of embedding/label indices to words (size: m)
    :return: List[np.ndarray]], List[List[str]]
    """
    clusters = {}
    cluster_words = {}
    for i, label in enumerate(labels): # the index becomes the key

        if label not in clusters:
            clusters[label] = [selected_D[i]]
        else:
            clusters[label].append(selected_D[i])

        if label not in cluster_words:
            cluster_words[label] = [ind_sD[i]]
        else:
            cluster_words[label].append(ind_sD[i])

    # sort by label to be better accessible
    clusters = [np.array(clusters[i]) for i in sorted(clusters.keys())]
    cluster_words = [cluster_words[i] for i in sorted(cluster_words.keys())]

    return clusters, cluster_words

def normalize_shifts_by_frequency(D:np.ndarray, sorted_pairs:List[Tuple[str,str]],
                                  freq1:Dict[str,int], freq2:Dict[str,int]) -> np.ndarray:
    """
    Normalize difference vectors with the logarithm of the difference of the
    original words' corpus frequencies. This is inspired by Cafagna et al. (2019)
    :param D: difference vectors (= shift vectors)
    :param sorted_pairs: word pairs in the same order as the shifts in D
    :param freq1: corpus frequencies of the first words
    :param freq2: corpus frequencies of the second words
    :return: shift vectors with 'normalized' lengths
    """
    flattening_applies = True
    overlap = list(set([p[0] for p in sorted_pairs]).intersection(set(freq1.keys())))
    for i in range(min(len(overlap),10)): # have a look at the first couple of words
        reconstructed_freq = math.e ** freq1[overlap[i]]
        if not np.isclose(reconstructed_freq, round(reconstructed_freq), atol=1e-5):
            flattening_applies = False # no flattening if any of the 'reconstructed' frequencies is not a whole number
    if flattening_applies: # restore absolute frequencies, but only of the needed entries
        freq1 = {w: round(math.e ** freq1[w], 0) for w, _ in sorted_pairs}
        freq2 = {w: round(math.e ** freq2[w], 0) for _, w in sorted_pairs}


    norms = np.array([max(np.log(abs(freq2[w2]-freq1[w1])),1) for w1,w2 in sorted_pairs])
    return (D.T/norms).T



def cluster_length(cluster:np.ndarray, length_metric:str) -> float:
    """
    Compute the average length of a cluster using different averaging methods.
    All vectors are measured with L2 norm first (np.linalg.norm()).
    :param cluster: m-sized cluster of shape (m,d)
    :param length_metric: one of 'mean', 'max', 'median', 'std'
    :return: average length
    """
    sizes = np.linalg.norm(cluster, axis=1)
    if length_metric == "max":      return float(np.max(sizes))
    elif length_metric == "mean":   return float(np.mean(sizes))
    elif length_metric == "median": return float(np.median(sizes))
    elif length_metric == "std": return float(np.std(sizes))
    else: raise NotImplementedError(f"{length_metric} is not implemented. "
                                    f"Use max/mean/median/std instead.")

def inner_distance(cluster:np.ndarray, triu:bool=True) -> float:
    """
    Compute the average pairwise distance between embeddings of a cluster,
    similarly to Bizzoni et al. (2019, "Grammar and Meaning"). While they count
    all distances (whole matrix), this (by default) counts each distance only
    once and also excludes the self-distances.
    :param cluster: m-sized cluster of shape (m,d)
    :param triu: optionally only count each distance once
    :return: average cosine distance.
    """
    if len(cluster.shape) == 1:
        return 0
    else:
        cosines = utils.cosine_matrix(cluster, cluster)
        # count each value only once = upper triangle without the diagonal
        if triu==True:
            cosines = cosines[np.triu(cosines, 1).nonzero()]
        # convert similarities to distances
        return float(np.mean(np.ones_like(cosines) - cosines))

def find_closest_concepts(centroids:np.ndarray,
                          space:np.ndarray, voc:Dict[str,int],
                          k:int=1,
                          csls:bool=False) -> List[List[Tuple[str,float]]]:
    """
    Compare one or more vectors (registered, but possibly not in the space's
    vocabulary) to all vectors in a space and return the nearest neighbors for
    each of them as a list of tuples (label, distance).
    CAUTION: this is memory-intensive if the space is large.
    :param centroids: one or more vectors, shape (n,d)
    :param space: embeddings of shape (u,d)
    :param voc: maps words to indices; length: u
    :param k: number of nearest neighbors
    :param csls: optionally use CSLS instead of cosine as a measure of distance.
    :return: List of nearest neighbors (and the neighbor's distance) per given centroid.
    """
    words = sorted(voc, key=voc.get)
    real_k = min(centroids.shape[0], k) # make sure that k is nor larger than the matrix

    # in case only one centroid is given
    if len(centroids.shape) == 1:
        centroids = np.expand_dims(centroids, 0)

    # The centroids have to be normed!
    sims_mat = utils.cosine_matrix(centroids, space, norm=False) # (n,d), (m,d) -> (n,m)
    if csls is True:
        sims_mat = utils.csls(sims_mat, knn=real_k)

    output = {}
    for i, sims in enumerate(sims_mat): # iterate over centroids
        for sim_idx in sims.argsort()[::-1][1:(1 + real_k)]: # taken from Jacob Eisenstein: https://github.com/jacobeisenstein/language-change-tutorial/blob/master/naacl-notebooks/DirtyLaundering.ipynb
            if i not in output:
                output[i] = [(words[sim_idx], 1-sims[sim_idx])]  # convert cosine to distance at the last second
            else:
               output[i].append((words[sim_idx], 1-sims[sim_idx]))

    # returns a list of lists, sorted like center_ids
    return [output[i] for i in sorted(output)]


def run(sp:SpacePair, PX:np.ndarray, monolingual:bool=False, ap_source:bool=False,
        out_dir:str=None,
        min_count:int=1, spaces_mincount:int=0, dist_nbs:int=10,
        dir_k:int=1, pairdist_csls:bool=False, signal:List[Tuple[str,str]]=None,
        options:utils.ConfigReader=None):
    """
    Perform unsupervised shift detection via cosine measures and clustering.
    Given a source space X, a target space Y, their vocabularies U and V, a set
    T of word pairs, and the Y-aligned source space PX, carry out the following
    processing steps:
    1. reduce the amount of data (T, X, U, Y, and V) to save memory. PX is not
        affected, but will only be accessed via the reduced U.
    2. calculate the set D of difference vectors: D = {y_v - Px_u | (u,v) in T}
    3. carry out the following points once each for the following partitions T':
        string-matching pairs; mismatching pairs; all pairs combined.
        3.1 calculate the pair distance dcos(Px_u, y_v) for all (u,v) in T'.
            - report the distances.
            - report the nearest neighbors (NN) of x_u in X and y_v in Y.
        3.2 cluster the difference vectors with affinity propagation (AP) and
            report the clusters and their centroids (one word pair per shift)
        3.3 report the size, average length, and inner distance of clusters
        3.4 report the NNs (in Y) of the clusters' centroids
        3.5 report some additional information (e.g. point of AP convergence)
    :param sp: SpacePair (holds X, U, Y, V, and T)
    :param PX: projected space (using SpacePair.P to project X onto Y)
    :param monolingual: use the SpacePair's vocabulary overlap as word pairs.
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
        if options("exp_unsup_min_wordcount") is not None:
            min_count = options("exp_unsup_min_wordcount")
        if options("exp_unsup_spaces_mincount") is not None:
            spaces_mincount = options("exp_unsup_spaces_mincount")
        if options("exp_unsup_neighbors") is not None:
            dist_nbs = options("exp_unsup_neighbors")
        if options("exp_unsup_clusterlabels") is not None:
            dir_k = options("exp_unsup_clusterlabels")
        if options("exp_unsup_use_csls") is not None:
            pairdist_csls = options("exp_unsup_use_csls")


    if signal is not None:
        sorted_pairs = signal # if we want to compare differing words across time
    elif monolingual is True:
        # make monolingual signal: most frequent shared words
        shared_freq = {}
        for w in set(sp.freq_x.keys()).intersection(set(sp.freq_y.keys())):
            shared_freq[(w, w)] = sp.freq_x[w] + sp.freq_y[w]
        sorted_pairs = sorted(shared_freq, key=shared_freq.get, reverse=True)
    else:
        sorted_pairs = sorted(sp.T, key=sp.T.get, reverse=True)

    # 1. 'prune' the word pairs to throw out unreliable embeddings
    sorted_pairs, sp = reduce_bilingual_signal(sorted_pairs, sp,
                                               min_count=min_count,
                                               spaces_mincount=spaces_mincount) # this is to save memory
    print(f"Reduced vocabulary to words with min. {min_count} corpus occurrences. "
          f"Continuing with {len(sorted_pairs)} pairs.")
    if spaces_mincount>0:
        print(f"Also reduced spaces to concepts with {spaces_mincount} corpus "
              f"occuccences in order to save memory. new sizes (X/Y): "
              f"{sp.X.shape[0]}/{sp.Y.shape[0]}.")

    # Make partitions
    matching_T = []
    mismatch_T = []
    for u, v in sorted_pairs:
        if u == v: matching_T.append((u, v))
        else:      mismatch_T.append((u, v))

    if len(mismatch_T) == 0 or len(matching_T) == 0:
        partitions = [("all", sorted_pairs)]
    else:
        partitions = [("matching", matching_T),
                      ("mismatch", mismatch_T),
                      ("all", sorted_pairs)]


    # 2. Calculate difference vectors
    from_vecs = np.array([  PX[sp.voc_x[u]] for u,_ in sorted_pairs])
    to_vecs   = np.array([sp.Y[sp.voc_y[v]] for _,v in sorted_pairs])
    D = utils.shift_directions(from_vecs, to_vecs, norm=False) # norm the vectors, just to be safe.
    D = normalize_shifts_by_frequency(D, sorted_pairs, sp.freq_x, sp.freq_y)

    voc_D = {p:i for i,p in enumerate(sorted_pairs)}


    # 3.
    for partition_name, pairs in partitions:
        cluster_timer = utils.Timer()
        print(f"\nStarting unsupervised shift detection on {partition_name} word pairs.")

        # 3.1 Pair distances
        output_pairdists(sp, PX, word_pairs=pairs,
                         neighbors=dist_nbs, use_csls=pairdist_csls,
                         out_dir=out_dir, partition_name=partition_name)
        cluster_timer("pair_distances")


        # 3.2 Clustering of difference vetors
        selected_D = D[[voc_D[pair] for pair in pairs]]
        ind_sD = {i: p for i, p in enumerate(pairs)}

        # value of labels = index in center_ids
        # index of labels = key in ind_sD
        # value of center_ids = key in ind_sD
        # index of center_ids = cluster label (= values of labels)
        if ap_source is True:
            print("Clustering source vectors ...")
            labels, center_ids, convergence_it = utils.affprop_clusters(np.array(sp.X[[sp.voc_x[u] for u,_ in pairs]]))
        else:
            print("Clustering shift vectors ...")
            labels, center_ids, convergence_it = utils.affprop_clusters(selected_D)
        cluster_timer("AP_clustering")

        # make arrays of vectors and lists of word pairs which belong to the same cluster
        clusters, cluster_words = reorganize(labels, selected_D, ind_sD)
        cluster_timer("re-organization")

        # 3.3 Cluster sizes, cluster lengths, inner distances
        cluster_sizes = [len(c) for c in clusters]

        lengths_max =     [cluster_length(cluster, "max") for cluster in clusters]
        lengths_mean =    [cluster_length(cluster, "mean") for cluster in clusters]
        lengths_median =  [cluster_length(cluster, "median") for cluster in clusters]
        lengths_std =     [cluster_length(cluster, "std") for cluster in clusters]
        inner_dists = [inner_distance(cluster) for cluster in clusters]

        lengths_max_normed =     eval_utils.z_scores(lengths_max)
        lengths_mean_normed =    eval_utils.z_scores(lengths_mean)
        lengths_median_normed =  eval_utils.z_scores(lengths_median)
        lengths_std_normed =     eval_utils.z_scores(lengths_std)
        inner_dists_normed =     eval_utils.z_scores(inner_dists)
        cluster_timer("lengths_and_inner_dist")

        del clusters  # to save memory


        # 3.4 Nearest Neighbors: find the vector(s) in Y most similar to a cluster's center exemplar
        direction_labels = find_closest_concepts(selected_D[center_ids],
                                                 sp.Y, sp.voc_y, k=dir_k)
        cluster_timer("closest_concepts")

        # 3.5 report everything
        df = DataFrame({
            "max_length" :          lengths_max,
            "mean_length":          lengths_mean,
            "median_length":        lengths_median,
            "std_length":           lengths_std,
            "max_length_zscore":    lengths_max_normed,
            "mean_length_zscore":   lengths_mean_normed,
            "median_length_zscore": lengths_median_normed,
            "std_length_zscore":    lengths_std_normed,
            "inner_distance" :      inner_dists,
            "inner_dist_zscore":    inner_dists_normed, # normalized among all clusters
            "cluster_size" :        cluster_sizes,
            "centroid" :            [ind_sD[center] for center in center_ids], # these are word pairs
            "direction_label" :     direction_labels, # these are tuples (word, distance)
            "cluster_words" :       cluster_words
        })
        # rank by shift size
        df = df.sort_values("inner_distance", ascending=False, ignore_index=True)
        df.to_csv(out_dir+partition_name+"_shift_clusters.tsv", sep='\t')

        cluster_timer.total()

        # additional information
        with open(out_dir+partition_name+"_clustering_stats", "w") as f:
            f.write(f"number_of_shifts\t{selected_D.shape[0]}\n"
                    f"number_of_clusters\t{len(center_ids)}\n"
                    f"convergence_criterion\t{convergence_it}\n"
                    f"\nparam_min_count\t{min_count}\n"
                    f"param_dist_nbs\t{dist_nbs}\n"
                    f"param_dir_k\t{dir_k}\n"
                    f"pairdist_csls\t{pairdist_csls}\n"
                    f"param_reduce_spaces\t{spaces_mincount}\n"
                    f"size_X\t{sp.X.shape[0]}\n"
                    f"size_Y\t{sp.Y.shape[0]}\n"
                    f"\n# time_taken:\n{cluster_timer}")







