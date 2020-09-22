#!/usr/bin/python3.6
#author: Simon Preissner

"""
This module provides helper functions.
"""

import os
import sys
import re
import csv
import math
from time import time
from ast import literal_eval
from typing import Dict, List, Tuple

import numpy as np
import pickle
from sklearn.cluster import AffinityPropagation
from sklearn.exceptions import ConvergenceWarning
from tqdm import tqdm
import matplotlib.pyplot as plt
from otalign.src import gw_optim
import ot

from default import RSC_YEARS

#========== INPUT, OUTPUT, TIMING ==========

def loop_input(rtype=str, default=None, msg=""):
    """
    Wrapper function for command-line input that specifies an input type
    and a default value. If the return type is set to 'filepath', it will be
    tested whether the file can be opened (only existing files will pass the input).
    :param rtype: type of the input. supports str, int, float, bool, list, dict, 'filepath'
    :type rtype: type or the string 'filepath
    :param default: value to be returned if the input is empty
    :param msg: message that is printed as prompt
    :type msg: str
    :return: value of the specified type
    """
    while True:
        s = input(msg+f" (default (type:{rtype}): {default}): ")
        if not s:
            return default
        elif rtype == str: # literal_eval() doesn't accept simple strings
            return str(s)

        else: # non-empty input, non-string return type
            try:
                s = literal_eval(s)
            except ValueError:
                print("Input needs to be convertable to", rtype, "-- try again.")
                continue

            if rtype == "filepath":
                try:
                    f = open(s, "r")
                    f.close()
                    return s
                except FileNotFoundError as e:
                    print("File",s,"not found -- try again.")
                    continue

            else: # any input other than a filepath
                if type(s)==rtype:
                    return s
                else:
                    try: # e.g. float required, but "2" parsed to int
                        return rtype(s)
                    except ValueError:
                        print("Input needs to be convertable to", rtype, "-- try again.")
                        continue

class ConfigReader():
    """
    Basic container and management of parameter configurations.
    Read a config file (typically ending with .cfg), use this as container for
    the parameters during runtime, and change/write parameters.

    CONFIGURATION FILE SYNTAX
    - one parameter per line, containing a name and a value
        - name and value are separated by at least one white space or tab
        - names should contain alphanumeric symbols and '_' (no '-', please!)
    - lines starting with '#' are ignored
    - no in-line comments!
    - define values of buiilt-in data types just as in Python
    - TODO strings containing quotation marks are not tested yet. Be careful!
    - config files should have the extension 'cfg' (to indicate their purpose)
    """

    def __init__(self, filepath, param_dict=None):
        self.filepath = filepath
        if not param_dict: # usual case: read config from a file
            self.params = self.read_config()
        else: # runtime case: pass a dictionary and return a ConfigReader object
            self.params = param_dict

    def __repr__(self):
        """
        returns tab-separated key-value pairs (one pair per line)
        """
        return "\n".join([str(k)+"\t"+str(v) for k,v in self.params.items()])

    def __call__(self, *paramnames):
        """
        Returns a single value or a list of values corresponding to the
        provided parameter name(s). Returns the whole config in form of a
        dictionary if no parameter names are specified.
        """
        if not paramnames: # return the whole config
            return self.params
        else: # return specified values
            values = []
            for n in paramnames:
                if n in self.params:
                    values.append(self.params[n])
                else:
                    print(f"WARNING: unable to find parameter '{n}'!")
                    values.append(None)
            return values[0] if len(values) == 1 else values

    def read_config(self):
        """
        Reads the ConfigReader's assigned file (attribute: 'filename') and parses
        the contents into a dictionary.
        - ignores empty lines and lines starting with '#'
        - takes the first continuous string as parameter key (or: parameter name)
        - parses all subsequent strings (splits at whitespaces) as values
        - tries to convert each value to float, int, and bool. Else: string.
        - parses strings that look like Python lists to lists
        :return: dict[str:obj]
        """
        cfg = {}
        with open(self.filepath, "r") as f:
            lines = f.readlines()

        for line in lines:
            line = line.rstrip()
            if not line: # ignore empty lines
                continue
            elif line.startswith('#'): # ignore comment lines
                continue

            words = line.split()
            paramname = words.pop(0)
            if not words: # no value specified
                print(f"WARNING: no value specified for parameter {paramname}.")
                cfg[paramname] = None
            else:
                line = " ".join(words)
                try:
                    cfg[paramname] = literal_eval(line)  # tries to read it as Python code
                except:
                    cfg[paramname] = str(line) # normal strings usually are not Python code

        return cfg



    def set(self, paramname, value):
        self.params.update({paramname:value})

class Timer():
    """
    Take time conveniently. Inizialize a timer like this:
        take_time = Timer()
    Then, take time intervals and give them a name:
        take_time("my_interval_name")
    For loops, use again():
        take_time.again("my_loop_name")
    To take the total time:
        take_time.total()
    For print-outs, convert the Timer object to a string, e.g.:
        print(take_time())
    """
    def __init__(self):
        self.T0 = time()
        self.t0 = time()
        self.times = {}
        self.steps = []
        self.period_name = ""

    def __call__(self, periodname:str) -> float:
        span = time() - self.t0
        self.t0 = time()
        self.steps.append(periodname)
        self.times.update({periodname: span})
        return span

    def __repr__(self, *args):
        steps = [s for s in args if s in self.steps] if args else self.steps
        return "\n".join([str(round(self.times[k], 5)) + "   " + k for k in steps])

    def again(self, periodname):
        """
        Take cumulative time of a recurring activity.
        :param periodname: str -- description of the activity
        :return: float -- time in seconds taken for current iteration of the activity
        """
        span = time() - self.t0
        self.t0 = time()
        if periodname in self.times:
            self.times[periodname] += span
        else:
            self.steps.append(periodname)
            self.times[periodname] = span
        return span

    def total(self):
        span = time() - self.T0
        self.steps.append("total")
        self.times.update({"total": span})
        return span


def load_space(filename: str) -> (np.ndarray, Dict[str, int]):
    """
    Read a space from a text file and return it as a 2D numpy array
    together with a dictionary (mapping from word to index).
    """
    print(f"loading space from {filename} ...")
    with open(filename, "r") as f:
        voc_length, dims = [int(s) for s in f.readline().rstrip().split()]
        voc = []
        space = []
        for vectorline in tqdm(f, desc="lines"):
            items = vectorline.rstrip().split()
            voc.append(items[0])
            space.append(np.array([float(i) for i in items[1:]]))

    voc = {w: i for i, w in enumerate(voc)}
    space = np.stack(space)

    return space, voc

def load_coupling(dest_dir):
    """
    'dest_dir' needs to contain at least 3 files: 'coupling', 'src_voc', and 'trg_voc'.
    returns:
        coupling: np.ndarray
        src_voc, trg_voc: List[str]
    """
    try:
        coupling = np.load(dest_dir+"coupling", allow_pickle=True)
    except:
        coupling = np.loadtxt(dest_dir + "coupling", dtype=float)
    with open(dest_dir + "src_voc", 'r') as f:
        src_words = [l.rstrip() for l in f.readlines()]
    with open(dest_dir + "trg_voc", 'r') as f:
        trg_words = [l.rstrip() for l in f.readlines()]

    return coupling, src_words, trg_words

def dump_coupling(coupling, src_words, trg_words, dest_dir, use_pickle=True, cfg=None):
    """
    Write a coupling (and the source/target words) to a directory.
    Set use_pickle to False if the pickle protocol version makes trouble.
    if possible, specify the config of the GWOT object that optimized the coupling.
    :param coupling: np.ndarray
    :param src_words, trg_words: List[str]
    :param dest_dir: str
    :param cfg: ConfigReader
    """
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    if use_pickle:
        coupling.dump(dest_dir+"coupling", protocol=pickle.HIGHEST_PROTOCOL)
    else:
        np.savetxt(dest_dir + "coupling", coupling, delimiter=" ")

    with open(dest_dir + "src_voc", 'w') as f:
        f.write("\n".join(src_words))
    with open(dest_dir + "trg_voc", 'w') as f:
        f.write("\n".join(trg_words))
    if cfg:
        with open(dest_dir + "config.cfg", 'w') as f:
            f.write(str(cfg))


#========== SIMILARITY MEASURES ==========

def cosines(word:str, embeddings:np.ndarray,
            vocab:list, norm:bool=True) -> np.ndarray:
    """
    Partly taken from Jacob Eisenstein:
    https://github.com/jacobeisenstein/language-change-tutorial/blob/master/naacl-notebooks/DirtyLaundering.ipynb

    Compute the cosine values of a word's vector to (a subset of) other word
    embeddings. Assumes that embeddings have a Euclidian norm of 1.
    :param word: word to which to compute cosine values
    :param embeddings: embedding space (one word per row, one dimension per column)
    :param vocab: the word's index corresponds to its row in the embeddings
    :param norm: bool -- faster if True
    :return: np.ndarray -- cosine values
    """
    emb = embeddings[vocab.index(word),]

    if norm:
        sims = np.dot(emb, embeddings.T) # similarity values of all embeddings with 'query'
    else:
        numerator = np.dot(emb, embeddings.T) # .T needed because it's a 'matrix'
        d_emb = math.sqrt(np.dot(emb, emb)) # first part of the denominator
        denominators = np.array([math.sqrt(np.dot(e,e)) for e in embeddings])
        sims = np.array([numerator/(1e-5 + d_emb*d) for d in denominators]) # 1e-5 avoids ZeroDivisionErrors

    return sims

def pairwise_cos_dist(vecs1:np.array, vecs2:np.array, no_zero_dists=False) -> np.array:
    """
    Taken from Jacob Eisenstein:
    https://github.com/jacobeisenstein/language-change-tutorial/blob/master/naacl-notebooks/DirtyLaundering.ipynb

    Compute the index-pairwise cosine distances of two arrays of vectors. The two arrays
    should hold the same word's embedding at the same index.
    :param vecs1: embedding vectors from one space
    :param vecs2: embedding vectors from another space
    :return: array of cosine distances.
    """
    numerator = (vecs1 * vecs2).sum(1)
    denominator = np.sqrt((vecs1**2).sum(1)) * np.sqrt((vecs2**2).sum(1))
    cos_dist = 1 - numerator / (1e-6 + denominator) # 1e-5 avoids ZeroDivisionErrors

    if no_zero_dists:
        # eliminate words that are zeroed out
        return cos_dist * (vecs1.var(1) > 0) * (vecs2.var(1) > 0)
    else:
        return cos_dist


def cosine_matrix(vecs1:np.array, vecs2:np.array, norm:bool=False) -> np.array:
    """
    Compute the cosines (not the cosine distances!) of vecs1 x vecs2.
    :param vecs1:  (m,d)
    :param vecs2:  (n,d)
    :return:       (m,n)
    """
    if norm:
        C = np.dot(vecs1, vecs2.T)
    else:
        nums = np.dot(vecs1, vecs2.T) # (m,n)
        den_v1 = np.expand_dims(np.array([np.sqrt(np.dot(v,v)) for v in vecs1]), -1) # (m,1)
        den_v2 = np.expand_dims(np.array([np.sqrt(np.dot(v,v)) for v in vecs2]), -1) # (n,1)
        dens = np.dot(den_v1, den_v2.T) # (m,n)
        C = nums/dens # (m,n)

    return C

def neighbors(word:str, embeddings:np.ndarray,
              vocab:list, norm:bool=True, k:int=3) -> list:
    """
    Partly taken from Jacob Eisenstein:
    https://github.com/jacobeisenstein/language-change-tutorial/blob/master/naacl-notebooks/DirtyLaundering.ipynb

    Find the k nearest neighbors of a word by cosine distance.
    :param word: word to which to compute nearest neighbors
    :param embeddings: embedding space (one word per row, one dimension per column)
    :param vocab: the word's index corresponds to its row in the embeddings
    :param k: number of neighbors to be returned
    :param norm: ignore the cosine denominator if embeddings have Euclidian norm of 1
    :return: list[str] nearest neighbors (words)
    """
    sims = cosines(word, embeddings, vocab, norm=norm)
    output = []
    for sim_idx in sims.argsort()[::-1][1:(1+k)]:
        if sims[sim_idx] > 0:
            output.append(vocab[sim_idx])
    return output


def csls(scores, knn=5):
    """
    This is copy-pasted from the code of Alvarez-Melis & Jaakkola (2018).

    Adapted from Conneau et al.
        rt = [1/k *  sum_{zt in knn(xs)} score(xs, zt)
        rs = [1/k *  sum_{zs in knn(xt)} score(zs, xt)
        csls(x_s, x_t) = 2*score(xs, xt) - rt - rs

    """

    def mean_similarity(scores, knn, axis=1):
        nghbs = np.argpartition(scores, -knn,
                                axis=axis)  # for rows #[-k:] # argpartition returns top k not in order but it's efficient (doesnt sort all rows)
        # TODO: There must be a faster way to do this slicing
        if axis == 1:
            nghbs = nghbs[:, -knn:]
            nghbs_score = np.concatenate([row[indices] for row, indices in zip(scores, nghbs)]).reshape(nghbs.shape)
        else:
            nghbs = nghbs[-knn:, :].T
            nghbs_score = np.concatenate([col[indices] for col, indices in zip(scores.T, nghbs)]).reshape(
                nghbs.shape)

        return nghbs_score.mean(axis=1)

    # 1. Compute mean similarity return_scores
    src_ms = mean_similarity(scores, knn, axis=1)
    trg_ms = mean_similarity(scores, knn, axis=0)
    # 2. Compute updated scores
    normalized_scores = ((2 * scores - trg_ms).T - src_ms).T
    return normalized_scores

def shift_directions(from_vecs:np.ndarray, to_vecs:np.ndarray, norm:bool=True) -> np.ndarray:
    """
    Computes the pairwise difference of the two sets of vectors to return
    the direction of shift in the form of vectors
    :param from_vecs: m,d
    :param to_vecs: m,d
    :param norm: True if the embeddings are already normed.
    :return: m,d
    """
    if norm is False:
        from_vecs, to_vecs = scale_embeddings(from_vecs, to_vecs)

    return to_vecs - from_vecs


def affprop_clusters(vecs:np.ndarray, max_iter:int=100, conv_max:int=15, conv_min:int=3) -> (list, list, int):
    """
    Cluster vectors with AffinityPropagation (scikit learn implementation).
    Lower the required number of non-changing iterations if AP doesn't converge.
    Return a list L of labels (one per element in vecs) and a list C of cluster centers.
    Additionally return the convergence criterion (= number of non-changing iterations)
    Each label in L corresponds to the index of that cluster's representative in C.
    Each number in C corresponds to the index of the repreentative in vecs.
    :param vecs: vectors to be clustered
    :param max_iter: maximum number of AP iterations.
    :param conv_max, conv_min: Upper and lower bounds of the convergence criterion.
    :return: list of labels, list of cluster center indices, conv_it parameter
    """

    C = cosine_matrix(vecs, vecs, norm=False) # this will be the input for AP

    conv_it = conv_max
    print(f"Performing Affinity Propagation on {len(vecs)} embeddings...")
    while conv_it >= conv_min:
        print(f"   ... with convergence at {conv_it} unchanged iterations...")
        af = AffinityPropagation(affinity='precomputed',
                                 convergence_iter=conv_it,
                                 max_iter=max_iter,
                                 random_state=None,
                                 verbose=True).fit(C)
        cluster_centers_indices = af.cluster_centers_indices_
        labels = af.labels_

        print(f"     found {len(cluster_centers_indices)} clusters.")
        if len(cluster_centers_indices) == 0:
            conv_it -= 1
        else:
            return labels, cluster_centers_indices, conv_it



#========== SPACE MANIPULATION ==========

def select_subspace(space:np.ndarray, voc:Dict[str,int],
                    words:List[str]) -> (np.ndarray, Dict[str,int]):
    """
    Select vectors of certain words and return these vectors along with a new
    vocabulary (= word-to-row mapping).
    """
    limited_voc = {k:voc[k] for k in set(voc.keys()).intersection(set(words))}  # limit the vocabulary
    space = space[sorted(list(limited_voc.values()))] # select certain word vectors
    # assign indices from 0 to len(words)-1 to the words
    voc = {k: i for i, k in enumerate(sorted(limited_voc, key=limited_voc.get))}

    return space, voc


def joint_frequency_space_pruning(space1:np.ndarray, space2:np.ndarray,
                                  voc1:Dict[str,int], voc2:Dict[str,int],
                                  size:int, start:int=0,
                                  freq1:Dict[str,int]=None, freq2:Dict[str,int]=None,
                                  frequencies_file:str=None,
                                  column1:int=None, column2:int=None) -> (np.ndarray, np.ndarray, Dict[str,int], Dict[str,int]):
    """
    Select sub-spaces based on joint freuency information.
    The frequency information can either be provided via two frequency distributions
    or by passing a .tsv file with frequency information as well as the 2 columns.
    :param space1, space2: embedding vectors
    :param voc1, voc2: dictionaries to access the vectors
    :param size: number of concepts to which to reduce
    :param start: leave out the n most frequent words
    :param freq1, freq2: first way of passing frequency information
    :param frequencies_file, column1, column2: second way of passing freq. info
    :return: two spaces and their respective vocabularies
    """

    shared_voc = set(voc1.keys()).intersection(set(voc2.keys()))

    # either read from a file or use the specified frequency distributions
    if frequencies_file and column1 and column2:
        freq1, freq2 = get_freqdists_from_file(frequencies_file,
                                               column1, column2,
                                               words1 = list(shared_voc),
                                               words2 = list(shared_voc))
    elif freq1 and freq2:
        freq1 = {k:v for k,v in freq1.items() if k in shared_voc}
        freq2 = {k:v for k,v in freq2.items() if k in shared_voc}
    else:
        raise ValueError("Missing information: "
                         "you need to specify either {frequencies_file, column1, column2} "
                         "or {freq1, freq2} when calling this function")

    # shorten to the specified range
    select_these1 = sorted(freq1, key=freq1.get, reverse=True)[start:size]
    select_these2 = sorted(freq2, key=freq2.get, reverse=True)[start:size]

    space1, voc1 = select_subspace(space1, voc1, select_these1)  # returns a space and a voc
    space2, voc2 = select_subspace(space2, voc2, select_these2)

    return space1, space2, voc1, voc2

def get_freqdists_from_file(filepath:str, column1:int, column2:int,
                            words1:List[str]=None, words2:List[str]=None,
                            log_flat_base:float=None)-> (Dict[str,float], Dict[str,float]):
    """
    Read a .tsv file to extract frequency distributions for 2 sets of words.
    :param filepath: path to a tsv file containing with a header and words in the first column.
    :param column1, column2: one column for each set of words
    :param words1, words2: sets of words for which to get the frequency
    :param log_flat_base: apply flattening to counter-act zipfian effects. Defaults to 'e'
    :return: 2 frequency distributions, in order of the passed columns
    """

    def crop_freq_to_wordlist(freq, words):
        # delete words that are in freq but not in words
        for k in set(freq.keys()).difference(set(words)):
            freq.pop(k)
        # include words that are not in freq but in words
        if len(freq) < len(words):
            for k in set(words).difference(set(freq.keys())):
                freq.update({k:0})
        return freq

    freq1 = {}
    freq2 = {}

    with open(filepath, newline='') as f:
        _ = f.readline()  # the header is useless
        tsvin = csv.reader(f, delimiter='\t')
        for row in tqdm(tsvin, desc="reading frequencies"):
            freq1.update({row[0] : float(row[column1])})
            freq2.update({row[0] : float(row[column2])})

    if words1:
        freq1 = crop_freq_to_wordlist(freq1, words1)
    if words2:
        freq2 = crop_freq_to_wordlist(freq2, words2)

    if log_flat_base is not None:
        if log_flat_base == 'e': base = math.e
        elif type(log_flat_base) == bool and log_flat_base is True: base = math.e
        elif float(log_flat_base) > 1 : base = log_flat_base
        else: base = math.e

        for k,v in freq1.items(): freq1[k] = flatten(v, base)
        for k,v in freq2.items(): freq2[k] = flatten(v, base)

    return freq1, freq2

def read_single_freqfile(filename:str, sep:str="\t",
                         exclude_header:bool=False,
                         log_flat_base:float=None) -> Dict[str, float]:
    """
    Read a two-column file (first column: words, second column: frequency) and
    return a dictionary of words to their frequencies. Optionally ignore the
    first line (e.g., if it is a header).
    :param filename: path to the file containing the frequencies
    :param sep: separator token.
    :param exclude_header: ignore the first line
    :return: words and their corresponding frequencies
    """
    freq = {}
    with open(filename, "r") as f:
        if exclude_header: _ = f.readline() # the header is useless
        for line in f:
            w, i = line.rstrip().split(sep)
            freq.update({w:float(i)})

    if log_flat_base is not None:
        base = math.e if log_flat_base == 'e' else log_flat_base if float(log_flat_base) > 1 else math.e
        freq = {k:flatten(v, base) for k,v in freq.items()}

    return freq

def rsc_freqfile_column(year:int) -> int:
    """
    look up the index of a given year's column in the RSC frequencies tsv file.
    RSC years span decade-wise from 1660 to 1920.
    """
    year = int(year)
    if year in RSC_YEARS:
        return RSC_YEARS.index(year)+1 # first column is the word
    else:
        raise LookupError(f"{year} is not in the list of RSC years: {RSC_YEARS}")

def flatten(x, base):
    try:
        return math.log(x, base)
    except ValueError:
        return 0.0


def center_embeddings(X, Y):
    """ Copied from Alvarez-Melis & Jaakkola (2018) """
    X -= X.mean(axis=0)
    Y -= Y.mean(axis=0)
    return X, Y

def scale_embeddings(X, Y):
    """ Copied from Alvarez-Melis & Jaakkola (2018) """
    X /= np.linalg.norm(X, axis=1)[:, None]
    Y /= np.linalg.norm(Y, axis=1)[:, None]
    return X, Y

def whiten_embeddings(X, Y):
    """
    Copied from Alvarez-Melis & Jaakkola (2018)

    PCA whitening. https://stats.stackexchange.com/questions/95806/how-to-whiten-the-data-using-principal-component-analysis
    Uses PCA of covariance matrix Sigma = XX', Sigma = ULU'.
    Whitening matrix given by:
        W = L^(-1/2)U'
    """
    X, Y = center_embeddings(X, Y)
    n, d = X.shape

    Cov_x = np.cov(X.T)
    _, S_x, V_x = np.linalg.svd(Cov_x)
    W_x = (V_x.T / np.sqrt(S_x)).T
    assert np.allclose(W_x @ Cov_x @ W_x.T, np.eye(d))  # W*Sigma*W' = I_d

    Cov_y = np.cov(Y.T)
    _, S_y, V_y = np.linalg.svd(Cov_y)
    W_y = (V_y.T / np.sqrt(S_y)).T
    assert np.allclose(W_y @ Cov_y @ W_y.T, np.eye(d))

    X = X @ W_x.T
    Y = Y @ W_y.T
    assert np.allclose(np.cov(X.T), np.eye(d))  # Cov(hat(x)) = I_d
    assert np.allclose(np.cov(Y.T), np.eye(d))

    return X, Y

def zipf_init(lang, n):
    """ Copied from Alvarez-Melis & Jaakkola (2018) """

    # See (Piantadosi, 2014)
    if lang == 'en':
        alpha, beta = 1.40, 1.88 #Other sources give: 1.13, 2.73
    elif lang == 'fi':
        alpha, beta = 1.17, 0.60
    elif lang == 'fr':
        alpha, beta = 1.71, 2.09
    elif lang == 'de':
        alpha, beta = 1.10, 0.40
    elif lang == 'es':
        alpha, beta = 1.84, 3.81
    else: # Deafult to EN
        alpha, beta = 1.40, 1.88
    p = np.array([1/((i+1)+beta)**(alpha) for i in range(n)])
    return p/p.sum()


def matchbins(scores:Dict[Tuple[str,str],float], magnify=1e3, threshold=0) -> (List[float], List[float]):
    """
    Separate the scores of translation pairs into two lists depending on
    whether the two words have an exact string match or not.
    :param scores: translation pairs with associated (coupling) scores.
    :param magnify: return larger values for better readability.
    :param threshold: only return scores above this value
    :return: two lists of float values; one for matches, one for mismatches.
    """
    matches, mismatches = [], []
    for (s, t), score in scores.items():
        score *= magnify
        if score > threshold:
            if s != t:
                mismatches.append(score)
            else:
                matches.append(score)
    return matches, mismatches

def scored_mutual_nn(scores, src_words:List[str]=None, trg_words:List[str]=None):
    """ Adapted from Alvarez-Melis & Jaakkola (2018) in order to include
    translation score. Finds mutual nearest neighbors among the whole source
    and target words. Returns a mapping of tuples: (tok_src, tok_trg) to their
    (coupling) score, or of indices {(id_src, id_trg):score}, if no source or
    target words are passed.
    """

    # find indices of the best translations for each word
    best_match_for_src = scores.argmax(1)  # (translation: trg -> src)
    best_match_for_trg = scores.argmax(0)  # (translation: src -> trg)

    paired = []
    for i in range(scores.shape[0]):  # for all words in the source space
        m = best_match_for_src[i]
        if best_match_for_trg[m] == i:
            paired.append((i, m))  # pair up indices

    if src_words and trg_words: # in order to return actual strings
        scored_pairs = {(src_words[i], trg_words[j]): scores[i, j] for (i, j) in paired}
    else: # in order to return indices
        scored_pairs = {(i, j): scores[i, j] for (i, j) in paired}

    return scored_pairs  # dict{(str,str):float} or dict{(int,int):float}

def hist_analysis(coupling:np.ndarray, src_words:List[str], trg_words:List[str],
                  save_pairs:str=None, save_plot:str=None):
    """
    Make a string-match analysis of a coupling (investigate how many string-matching
    translation pairs it yields) and save the results as a file and/or as a histogram,
    if specified.
    Default parameters for the histogram: bins=100, scale at 1e-3 (not fixed)
    """
    print("Analyzing a coupling...")

    scored_pairs = scored_mutual_nn(coupling, src_words=src_words, trg_words=trg_words)
    matches, mismatches = matchbins(scored_pairs)

    n_match = len(matches)
    n_mismatch = len(mismatches)
    n_overlap = len(set(src_words).intersection(set(trg_words)))
    retrieved = round(n_match / n_overlap, 4)
    print(f"pairs:{len(scored_pairs)}\tmatches:{n_match}\tmismatches:{n_mismatch}\t"
          f"overlap:{n_overlap}\tretrieved:{retrieved}")

    if save_pairs is not None:
        with open(save_pairs, "w") as f:
            f.write(f"pairs:{len(scored_pairs)}\tmatches:{n_match}\t"
                    f"mismatches:{n_mismatch}\toverlap:{n_overlap}\t"
                    f"retrieved:{retrieved}\n")
            for rank, (x, y) in enumerate(sorted(scored_pairs, key=scored_pairs.get, reverse=True)):
                score = scored_pairs[(x, y)]
                f.write(f"{x:<15} {y:<15}   {round(score * 1000, 5):6.5f}\n")

    if save_plot is not None:
        plt.hist([matches, mismatches], label=["matches", "mismatches"],
                 histtype='step', stacked=False, bins=100)
        plt.xlabel("x 1e-3")
        plt.legend()
        plt.savefig(save_plot)
        plt.close()
