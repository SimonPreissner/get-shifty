"""
these are helper functions to analyze clusters and evaluate experiment results
"""

import utils
import numpy as np
from typing import List, Tuple
import ast

import pandas
from pandas import DataFrame



def z_scores(sequence, weights:np.ndarray=None) -> np.ndarray:
    """
    Normalize the values in a sequence to the normal distribution by subtracting
    its mean and dividing by its standard deviation.
    :param sequence: array-like object of length m
    :param weights: array-like object of length m to scale the values
    :return: np.ndarray of shape (m,)
    """
    sequence = np.array(sequence)
    if weights is not None:
        assert sequence.shape == weights.shape
        sequence = sequence * weights
    mu = np.mean(sequence)
    sd = np.std(sequence)

    return (sequence - mu )/sd

def add_z_scores(df:DataFrame, label:str, sequence, weights:np.ndarray=None) -> np.ndarray:
    """
    Compute z-scores of a sequence and add it as a row to a DataFrame
    :param df: pandas.DataFrame object or Dict. If a DataFrame, must be the
    same length as 'sequence'
    :param label: name of the sequence to be added to the DataFrame.
    :param sequence: any sequence of numeric values
    :param weights: optionally scale each of the values in the sequence
    :return: sequence of z-scores
    """
    sequence = z_scores(sequence, weights=weights)
    df[label]=sequence
    return sequence

def make_normal_dist(samples:int, iterations:int):
    """
    Creates a numerically approximated normal distribution of a specified length.
    :param samples: length of the distribution to be created
    :param iterations: number of approximation iterations (higher = more exact)
    :return: normal distribution.
    """
    normal = np.sort(np.random.randn(samples))
    for i in range(iterations):
        normal += np.sort(np.random.randn(samples))
    return normal / iterations




def read_results(dir_to_files:str, file_stub:str, exptype:str, with_baseline=True, tuples=False) -> (utils.ConfigReader, DataFrame, DataFrame):
    """
    Read exeriment results from a directory. containing at least 2 .tsv files and one text file.
    :param dir_to_files: directory path.
    :param file_stub: specifies the sub-group of results. The 3 files to be read
     start with this stub. e.g. 'all_technical'
    :param exptype: one of ["dis_tech", "distech", "unsup_mono", "unsup_bi"]
    :param with_baseline: if True, this also loads a clustering_baseline.tsv
    :param tuples: if True, this acknowledges the 'centroid' column as containing tuples
    :return: experiment statistics, DataFrame with pair distances, DataFrame with clustering results
    """
    if exptype in ["dis_tech", "distech", "dis_tech/", "distech/"]:
        dists_file = dir_to_files + file_stub + "_dists.tsv"
    else:
        dists_file = dir_to_files + file_stub + "_pairdists.tsv"

    stats_file = dir_to_files + file_stub + "_clustering_stats"
    clust_file = dir_to_files + file_stub + "_shift_clusters.tsv"

    stats = utils.ConfigReader(stats_file)
    dists = pandas.read_csv(dists_file, header=0, index_col=0, delimiter="\t")
    clust = pandas.read_csv(clust_file, header=0, index_col=0, delimiter="\t")


    dists["src_neighbors"] = [ast.literal_eval(x) for x in dists["src_neighbors"]]
    dists["trg_neighbors"] = [ast.literal_eval(x) for x in dists["trg_neighbors"]]

    if tuples is True:
        clust["centroid"] = [ast.literal_eval(x) for x in clust["centroid"]]
    clust["direction_label"] = [ast.literal_eval(x) for x in clust["direction_label"]]
    clust["cluster_words"] = [ast.literal_eval(x) for x in clust["cluster_words"]]

    if with_baseline is True:
        bl_file = dir_to_files + file_stub + "_clustering_baseline.tsv"
        try:
            baseline = pandas.read_csv(bl_file, header=0, index_col=0, delimiter="\t")
            return stats, dists, clust, baseline
        except FileNotFoundError:
            print(f"WARNING: unable to find baseline (=random) clusters. "
                  f"Maybe deactivate the loading by setting 'with_baseline=False'?")
            return stats, dists, clust
    else:
        return stats, dists, clust

def make_readable(dir_to_files:str, filestub:str, exptype:str, extra="", tuples=True):
    """
    Translate a .tsv file into a .txt file with fewer numbers and better spacing.
    :param dir_to_files: directory path to where the experiment results are.
    :param filestub: first, specifying part of the .tsv files (cf. read_results())
    :param exptype: one of ["dis_tech", "distech", "unsup_mono", "unsup_bi"]
    :param extra: optionally give a 'special' name as part of the resulting file's name
    :param tuples: indicate that the 'centroid' and 'pair' columns contain tuples (not just a string)
    """
    stats, df_dist, df_clust = read_results(dir_to_files, filestub, exptype, with_baseline=False, tuples=tuples)

    out_dists = dir_to_files + filestub + extra + "_dists.txt"
    out_clusters = dir_to_files + filestub + extra + "_shift_clusters.txt"

    if exptype in ["distech", "dis_tech", "distech/", "dis_tech/"]:
        with open(out_dists, "w") as f: # write pair distances
            for i in df_dist.index:
                row = df_dist.loc[i]
                f.write(f"{row['word']:<17} {row['distance']:<6.5f}\n")
                f.write(f"   src neighbors: {' '.join([t[0] for t in row['src_neighbors']])}\n")
                f.write(f"   trg neighbors: {' '.join([t[0] for t in row['trg_neighbors']])}\n")
                f.write("\n")

        with open(out_clusters, "w") as f: # write cluster information
            for i in df_clust.index:
                row = df_clust.loc[i]
                f.write(
                    f"{row['centroid']:<17} dist {row['inner_distance']:<6.5f}   size {row['cluster_size']:>3}   cluster ID {i:>3}\n")
                f.write(f"   directions: {' '.join([t[0] for t in row['direction_label']])}\n")
                f.write(f"   c. members: {' '.join(row['cluster_words'])}\n")
                f.write("\n")
    else:
        with open(out_dists, "w") as f: # write pair distances
            for i in df_dist.index:
                row = df_dist.loc[i]
                f.write(f"{row['src_word'] + ' -- ' + row['trg_word']:<30}   {row['distance']:<6.5f}\n")
                f.write(f"   src neighbors: {' '.join([t[0] for t in row['src_neighbors']])}\n")
                f.write(f"   trg neighbors: {' '.join([t[0] for t in row['trg_neighbors']])}\n")
                f.write("\n")

        with open(out_clusters, "w") as f: # write cluster information
            for i in df_clust.index:
                row = df_clust.loc[i]
                cluster_words = row['cluster_words']
                cluster_words = [w1 if w1 == w2 else w1 + "/" + w2 for w1, w2 in cluster_words]
                centroid_pair = row['centroid']
                f.write(
                    f"{centroid_pair[0] + ' -- ' + centroid_pair[1]:<30}   dist {row['inner_distance']:<6.5f}   size {row['cluster_size']:>3}   cluster ID {i:>3}\n")
                f.write(f"   directions: {' '.join([t[0] for t in row['direction_label']])}\n")
                f.write(f"   c. members: {' '.join(cluster_words)}\n")
                f.write("\n")

def read_annotated_pairs(filepath:str) -> List[Tuple[str,str]]:
    """
    Expects a file with at least 2 columns. These first 2 columns are treated as the pairs.
    """
    pairs = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.rstrip().split()[:2]
            pairs.append((line[0],line[1]))
    return pairs

def sort_out_space_filepaths(yearpair:Tuple[int, int], spacetype:str) -> (str, str):
    """
    Sort out the filepaths to the two spaces of a space pair.
    This function is limited to the two sets of embedding models used in the
    shift experiments (not the other ones evaluated in Chapter 3 of the thesis).
    :param spacetype: one of ["incremental", "noalign"]
    """
    if spacetype == "incremental":
        data_dir = "data/vectors1929-init-tc1-t3/"
    elif spacetype == "individual":
        data_dir = "data/vectors1929-noalign-tc0-t3/"
    else:
        raise ValueError(f"Spacetype '{spacetype}' not known. Use 'incremental' or 'noalign'.")

    spacefile1 = data_dir + "rsc-all-corr-" + str(yearpair[0])[:-1] + ".txt"
    spacefile2 = data_dir + "rsc-all-corr-" + str(yearpair[1])[:-1] + ".txt"

    return spacefile1, spacefile2





def significant_clusters(df:DataFrame, criterion:str, end:str, k:int=0, normalize:bool=False):
    """
    Select unusual clusters, either by z-score or (in the case k>0) select the
    entries with the highest score (defined by the criterion).
    the top/bottom k clusters
    :param df: DataFrame or DataFrame.Series
    :param criterion: DataFrame label by which to decide significance
    :param end: one of ["high", "low"]
    :param k: if set to 0, returns clusters with z-scores outside of +-1.65
    :param normalize: if True, the specified values are normalized to z-scores.
    :return: DataFrame
    """
    if k == 0:
        z_crit = 1.65 # marks one-tailed significance of p<0.05
        if normalize is True:
            _ = add_z_scores(df, criterion+"_zscore", df[criterion])
            criterion = criterion+"_zscore"
        if end == "high":
            return df[["centroid", "direction_label", "cluster_words"]][(df[criterion] > z_crit)]
        elif end == "low":
            return df[["centroid", "direction_label", "cluster_words"]][(df[criterion] < -z_crit)]
        else:
            raise ValueError("Parameter 'end' can only be 'high' or 'low'")
    elif k >= 1:
        sorted_df = df.sort_values(by=[criterion])
        if end == "high":
            return sorted_df[["centroid", "direction_label", "cluster_words"]][-k:]
        elif end == "low":
            return sorted_df[["centroid", "direction_label", "cluster_words"]][:k]
        else:
            raise ValueError("Parameter 'end' can only be 'high' or 'low'")
    else:
        print("ERROR: parameter 'k' must be a positive integer or 0")

def jointhem(values, may_be_pairs=False):
    """
    Re-writes a list of values as a string.
    :param values: Tuple or List[str] or List[Tuple[str,str]] or List[Tuple[str,float]]
    :param may_be_pairs: warns that tuples may actually be word pairs instead
    of word-value tuples.
    :return: string
    """
    if type(values[0]) == tuple: #
        if may_be_pairs: # tuples may be word pairs
            # return tuples if necessary, otherwise just one of the two elements.
            return ', '.join([str(t[0]) if t[0] == t[1] else str(t[0]) + "--" + str(t[1]) for t in values])
        else: # tuples are just string-float pairs, so only return the strings of them
            return ', '.join([str(t[0]) for t in values])
    elif type(values) == list: # no list of tuples, but still a list
        return ', '.join(values)
    elif type(values) == tuple: # this takes care of single tuples
        return str(values[0])+"--"+str(values[1]) if values[0] != values[1] else str(values[0])
    else: # neither a list nor a list of tuples nor a tuple
        return values

def print_clusters(df:DataFrame, title:str="", just_IDs:bool=False, establishment:bool=True, tuples:bool=False):
    """
    Similar to make_readable(), but doesn't write the results to a file.
    :param df: DataFrame containing columns with the names "cluster_words", "direction_label", and "centroid"
    :param title: any description.
    :param just_IDs: print only the corresponding index numbers (instead of the data)
    :param establishment: if True, returns the overlap of cluster members and cluster labels
    :param tuples: if True, this will tread the centroid as a Tuple[str,str]
    """
    print(title, "\n")
    if just_IDs is True:
        print(df.index)
    else:
        for ind, cluster, labels, centroid in zip(df.index, df["cluster_words"], df["direction_label"], df["centroid"]):
            if establishment is True:
                est = len(set(jointhem(labels).split(', ')).intersection(
                          set(jointhem(cluster, may_be_pairs=True).split(', '))))
            else:
                est = "N/A"

            print(f"centroid: {jointhem(centroid, may_be_pairs=tuples)}"
                  f" (ID: {ind})\n"
                  f"{'size:':<12} {len(cluster):<5}  "
                  f"{'establishment: ' + str(est) if establishment is True else ''}\n"
                  f"{'labels:':<12} {jointhem(labels)}\n"
                  f"{'members:':<12} {jointhem(cluster, may_be_pairs=True)}\n")

        print("\n")

def print_by_ID(df:DataFrame, ID:int, establishment:bool=True, tuples:bool=False):
    """
    Given the index of a DataFrame entry, print the corresponding cluster
    :param df: DataFrame containing columns with the names "cluster_words", "direction_label", and "centroid"
    :param ID: some index
    :param establishment: if True, returns the overlap of cluster members and cluster labels
    :return: if True, treats the value of 'centroid' as a tuple (use this for UnsupBi)
    """
    data = df.loc[ID]
    if establishment is True:
        est = len(set(jointhem(data['direction_label']).split(', ')).intersection(
                  set(jointhem(data['cluster_words'], may_be_pairs=True).split(', '))))
    else:
        est = "N/A"

    print(f"centroid: {jointhem(data['centroid'], may_be_pairs=tuples)} (ID: {ID})")
    print(f"{'size:':<12} {len(data['cluster_words']):<5}  "
          f"{'establishment: '+str(est) if establishment is True else ''}")
    print(f"{'labels:':<12} {jointhem(data['direction_label'])}")
    print(f"{'members:':<12} {jointhem(data['cluster_words'], may_be_pairs=True)}\n")

