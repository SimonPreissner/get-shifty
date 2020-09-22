"""
This class implements Gromov-Wasserstein Optimal Transport.
Several parts are copied from Alvarez-Melis and Jaakkola (2018) and adapted to the use at hand.
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import ot

from otalign.src import gw_optim

from typing import List, Dict, Tuple, Any
import utils
import default
from default import COUPLING_CONFIG as dccfg


class GWOT():

    def __init__(self, gwot_config:utils.ConfigReader,
                 x_voc:Dict[str,int], y_voc:Dict[str,int],
                 x_freq:Dict[str,float]=None, y_freq:Dict[str,float]=None,
                 size:int=None):
        """
        GWOT objects do not hold spaces; they only hold a coupling, source words,
        target words, and the parameters necessary to optimize the coupling.
        The coupling size can be reduced at any time, but not be increased.
        To increase the coupling size, get the config via compile_config(), then
        change the parameters, and re-initialize a GWOT object.
        """

        print("Initializing coupling instance.")
        # translating the parameters into attributes for programming convenience
        self.pretrained_loc  = gwot_config.params.get("pretrained_loc",  None)
        self.out_absdir      = gwot_config.params.get("out_absdir",      None)
        self.score_type      = gwot_config.params.get("score_type",      dccfg["score_type"])
        self.adjust          = gwot_config.params.get("adjust",          dccfg["adjust"])
        self.metric          = gwot_config.params.get("metric",          dccfg["metric"])
        self.normalize_vecs  = gwot_config.params.get("normalize_vecs",  dccfg["normalize_vecs"])
        self.normalize_dists = gwot_config.params.get("normalize_dists", dccfg["normalize_dists"])
        self.distribs        = gwot_config.params.get("distribs",        dccfg["distribs"])
        self.share_vocs      = gwot_config.params.get("share_vocs",      dccfg["share_vocs"])
        self.max_anchors     = gwot_config.params.get("max_anchors",     dccfg["max_anchors"])
        # set the size as specified in the SpacePair cfg, or in te GWOT cfg, or to the default
        self.size = size if size else gwot_config.params.get("size", dccfg["size"])
        max_coupling_size = min(len(x_voc), len(y_voc))
        if self.size > max_coupling_size:
            print(f"WARNING in GWOT.__init__(): desired coupling size "
                  f"({self.size}) exceeds vocabulary size ({max_coupling_size}). "
                  f"The coupling will be smaller only be optimized for "
                  f"these {max_coupling_size} points.")


        self.opt_config = {k[4:]:v for k,v in gwot_config.params.items() if k.startswith("opt_")}
        self.fit_config = {k[4:]:v for k,v in gwot_config.params.items() if k.startswith("fit_")}

        self.scores = None  # matching scores (potentially the same as self.coupling)
        self.coupling = None  # optimized in fit()
        self.mapping = None  # will be updated after optimization

        # also load coupling if provided
        self.src_words = []
        self.trg_words = []
        self.p = {}
        self.q = {}
        self.sort_out_words_and_probs(x_voc, y_voc,
                                      x_freq, y_freq,
                                      joint=self.share_vocs)

        self.solver = self.init_optimizer()
        self.solver.compute_accuracy = False # this is an artifact to fix a bug in otalign.gw_optim



    def init_optimizer(self):
        print('Initializing Gromov-Wasserstein optimizer')
        return gw_optim.gromov_wass_solver(
            metric =           self.metric,
            normalize_dists =  self.normalize_dists,
            loss_fun =         self.opt_config.get("loss_fun",         dccfg["opt_loss_fun"]),
            entropic =         self.opt_config.get("entropic",         dccfg["opt_entropic"]),
            entreg =           self.opt_config.get("entreg",           dccfg["opt_entreg"]),
            tol =              self.opt_config.get("tol",              dccfg["opt_tol"]),
            round_g =          self.opt_config.get("round_g",          dccfg["opt_round_g"]),
            compute_accuracy = self.opt_config.get("compute_accuracy", dccfg["opt_compute_accuracy"]),
            gpu =              self.opt_config.get("gpu",              dccfg["opt_gpu"])
            )

    def sort_out_words_and_probs(self, x_voc:Dict[str,int], y_voc:Dict[str,int],
                                 x_freq:Dict[str,float], y_freq:Dict[str,float],
                                 joint=False):

        if self.pretrained_loc:
            print(f"loading pre-trained coupling from: \n   {self.pretrained_loc} ...")
            self.coupling, self.src_words, self.trg_words = utils.load_coupling(self.pretrained_loc)
            if self.size != len(self.src_words):
                print(f"   WARNING: mismatch between coupling sizes "
                      f"of GWOT object ({self.size}) "
                      f"and pre-trained coupling ({len(self.src_words)}).")
                self.size = len(self.src_words)

            print(f"   estimating p and q based on the coupling ...")
            src_freq = {w:f for w,f in zip(self.src_words, np.sum(self.coupling, 1))}
            trg_freq = {w:f for w,f in zip(self.trg_words, np.sum(self.coupling, 0))}
            self.p, self.q = self.prob_dists(x_freq=src_freq,
                                             y_freq=trg_freq,
                                             dist_shape='custom') # self.[src|trg]_words is already set

        elif x_freq and y_freq:
            if joint:
                print(f"selecting {self.size} shared words for alignment by frequency ...")
                shared_words = set(x_freq.keys()).intersection(y_freq.keys())
                M = sum(x_freq.values())
                N = sum(y_freq.values())
                # shared words, relative frequencies
                joint_freq = {w:(x_freq[w]/M + y_freq[w]/N) for w in shared_words}
                joint_selection = [k for k in sorted(joint_freq,
                                                     key=joint_freq.get,
                                                     reverse=True)][:self.size]
                src_selection = joint_selection
                trg_selection = joint_selection
            else:
                print(f"selecting {self.size} words for alignment by frequency ...")
                src_selection = sorted(x_freq, key=x_freq.get, reverse=True)[:self.size]
                trg_selection = sorted(y_freq, key=y_freq.get, reverse=True)[:self.size]
            # these lists are sorted by index in the space
            self.src_words = sorted(src_selection, key=x_voc.get)
            self.trg_words = sorted(trg_selection, key=y_voc.get)
            self.p, self.q = self.prob_dists(x_freq=x_freq, y_freq=y_freq,
                                             dist_shape=self.distribs,
                                             size=self.size)

        else:
            print(f"selecting the {self.size} first words of the vocabulary for alignment ...")
            self.src_words = sorted(x_voc, key=x_voc.get)[:self.size] # these need to be sorted by their words' indices in the coupling
            self.trg_words = sorted(y_voc, key=y_voc.get)[:self.size]
            self.p = ot.unif(self.size)
            self.q = ot.unif(self.size)

    def fit(self, X, Y, x_voc, y_voc, size:int=None, maxiter:int=None, print_every:int=None,
            plot_every:int=None, verbose:bool=None, save_plots:str=None, proj_limit:int=None):
        """
        Largely adopted from Alvarez-melis & Jaakkola (2018).
        Wrapper function that carries out all necessary steps to estimate a
        bilingual mapping using Gromov-Wasserstein distance.
        """
        print('Fitting bilingual mapping with Gromov Wasserstein')

        # select from optional argumentss, else from self.config, else the default
        maxiter     = self.fit_config.get("maxiter",      300) if maxiter is None else maxiter
        print_every = self.fit_config.get("print_every", None) if print_every is None else print_every
        plot_every  = self.fit_config.get("plot_every",  None) if plot_every is None else plot_every
        verbose     = self.fit_config.get("verbose",    False) if verbose is None else verbose
        save_plots  = self.fit_config.get("save_plots",  None) if save_plots is None else save_plots

        # 0. Pre-processing: select subspaces and normalize them
        if size:
            if size <= self.size:
                # change the object's coupling size additionally to the vector selection
                if size < self.size:
                    print(f"In fit(): changing the size configurations "
                      f"from {self.size} to {size}.")
                    self.size = size

                src_embs, trg_embs, \
                src_voc, trg_voc, \
                self.src_words, self.trg_words, \
                self.p, self.q = self.tailor_data_to_size(size, X, Y, x_voc, y_voc,
                                                          self.src_words, self.trg_words,
                                                          freq1={k: v for k, v in zip(self.src_words, self.p)},
                                                          freq2={k: v for k, v in zip(self.trg_words, self.q)})
            else:
                raise ValueError(f"coupling size {size} exceeds the maximum "
                                 f"coupling size of {self.size} set upon initialization.")
        else:
            # work with the previously provided size settings and just select vectors
            src_embs, src_voc = utils.select_subspace(X, x_voc, self.src_words)
            trg_embs, trg_voc = utils.select_subspace(Y, y_voc, self.trg_words)

        src_embs, trg_embs = self.normalize_embeddings(src_embs, trg_embs)

        # 1. Solve Gromov Wasserstein problem
        print('Solving optimization problem...')
        G = self.solver.solve(src_embs,trg_embs,
                              self.p, self.q,
                              maxiter=maxiter,
                              plot_every=plot_every, print_every=print_every,
                              verbose=verbose, save_plots=save_plots)
        self.coupling = G

        # 2. From Couplings to Translation Score
        print('Computing translation scores...')
        print(self.score_type, self.adjust)
        self.compute_scores(self.score_type, x_embs=src_embs, y_embs=trg_embs, adjust = self.adjust)

        # 3. Find mutual translation pairs and solve Procrustes' to get the mapping
        self.mapping = self.get_mapping(src_embs, trg_embs, src_voc, trg_voc, max_anchors=proj_limit)

    def get_mapping(self,
                    src_embs: np.ndarray, trg_embs: np.ndarray,
                    src_voc: Dict[str, int], trg_voc: Dict[str, int],
                    max_anchors: int = None) -> np.ndarray:
        """
        Adapted from Alvarez-Melis & Jaakkola (2018).
        Obtain an orthogonal mapping from an optimal coupling.
        Only implements the 'mutual_nn' method (not 'barycentric' or 'all')
        if 'max_anchors' is specified, this takes the highest-scoring translation pairs.
        """

        scored_pairs = self.scored_mutual_nn(self.scores,
                                             sorted(src_voc, key=src_voc.get),  # pass word lists
                                             sorted(trg_voc, key=trg_voc.get))
        pseudo = sorted(scored_pairs, key=scored_pairs.get, reverse=True)  # sort by translation probability

        if max_anchors is not None:
            pseudo = pseudo[:max_anchors]
        else:
            pseudo = pseudo[:self.max_anchors]
        print('Finding orthogonal mapping with {} anchor points'.format(len(pseudo)))

        idx_src = [src_voc[ws] for ws, _ in pseudo]  # translate strings to indices
        idx_trg = [trg_voc[wt] for _, wt in pseudo]
        xs_nn = src_embs[idx_src]  # select vectors for SVD from the space
        xt_nn = trg_embs[idx_trg]
        P = procrustes(xs_nn, xt_nn)

        return P

    def sort_out_mapping(self, X, Y, x_voc, y_voc, max_anchors=None):
        """
        Performs the same action as get_mapping(), but acts with the object's attributes.
        This method presupposes that the following attributes are set:
            - src_words, trg_words
            - src_voc, trg_voc
            - coupling
        :param max_anchors: maximum number of translation pairs along which to align
        """
        for attribute in [self.coupling, self.score_type,
                          self.src_words, self.trg_words]:
            if attribute is None:
                print(f"Warning in sort_out_mapping(): missing a required attribute. "
                      f"Returning empty matrix.")
                return np.array([])

        src_embs, src_voc = utils.select_subspace(X, x_voc, self.src_words)
        trg_embs, trg_voc = utils.select_subspace(Y, y_voc, self.trg_words)

        self.scores = self.compute_scores(self.score_type,
                                          src_embs,
                                          trg_embs,
                                          adjust=self.adjust)

        self.mapping = self.get_mapping(src_embs, trg_embs, src_voc, trg_voc,
                                        max_anchors=max_anchors)
        return self.mapping

    def scored_mutual_nn(self, scores, src_words=None, trg_words=None):
        """
        Adapted from Alvarez-Melis & Jaakkola (2018) to incorporate coupling scores.
        Finds mutual nearest neighbors among the whole source and target words.
        Returns a mapping of tuples: (tok_src, tok_trg) to their confidence score,
        or of indices {(id_src, id_trg):score}, if no source or target words are passed.
        :type src_words: list[str]
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


    def sort_out_scored_mutual_nn(self):
        """
        Uses object attributes to create a set of scored mutual translations.
        This method presupposes that the following attributes are set:
            - src_words
            - trg_words
            - scores
        """
        self.compute_scores(self.score_type)

        for attribute in [self.scores, self.src_words, self.trg_words]:
            if attribute is None:
                print(f"Warning in sort_out_scored_mutual_nn(): "
                      f"missing a required attribute. Returning empty dict.")
                return {}

        return self.scored_mutual_nn(self.scores,
                                     src_words=self.src_words,
                                     trg_words=self.trg_words)

    def compute_scores(self, score_type, x_embs=None, y_embs=None, adjust=None, verbose=False):
        """
        Copied from Alvarez-Melis & Jaakkola (2018).
        """
        if score_type == 'coupling':
            scores = self.coupling
        elif x_embs is not None and y_embs is not None:
            if score_type == 'barycentric':
                ot_emd = ot.da.EMDTransport()
                ot_emd.xs_ = x_embs
                ot_emd.xt_ = y_embs
                ot_emd.coupling_ = self.coupling
                xt_s = ot_emd.inverse_transform(Xt=y_embs)  # Maps target to source space
                scores = -sp.spatial.distance.cdist(x_embs, xt_s, metric=self.metric)  # FIXME: should this be - dist?
            elif score_type == 'distance':
                # For baselines that only use distances without OT
                scores = -sp.spatial.distance.cdist(x_embs, y_embs, metric=self.metric)
            elif score_type == 'projected':
                # Uses projection mapping, computes distance in projected space
                scores = -sp.spatial.distance.cdist(x_embs, y_embs @ self.mapping.T, metric=self.metric)
            else:
                print(f"WARNING in GWOT.compute_scores(): invalid score type."
                      f"Expected one of [coupling, barycentric, distance, projected], "
                      f"but got '{score_type}'. Continuing with score type 'coupling'.")
                scores = self.coupling
        else:
            print(f"WARNING in GWOT.compute_scores(): embeddings required to compute"
                  f"scores of the type '{score_type}'. Continuing with score type 'coupling'.")
            scores = self.coupling

        if adjust == 'csls':
            scores = utils.csls(scores, knn=10)

        if verbose:
            plt.figure()
            plt.imshow(scores, cmap='jet')
            plt.colorbar()
            plt.show()

        self.scores = scores
        return scores

    def tailor_data_to_size(self, size:int,
                            space1:np.ndarray, space2:np.ndarray,
                            voc1:Dict[str,int], voc2:Dict[str,int],
                            words1:List[str], words2:List[str],
                            freq1:Dict[str,float]=None, freq2:Dict[str,float]=None):
        """
        Reduce all relevant data to the specified size.
        This affects the spaces, vocabularies, word lists, and frequency counts.
        """

        if freq1 is not None and freq2 is not None:
            keep_these1 = sorted(freq1, key=freq1.get, reverse=True)[:size]  # restricts to a certain size
            keep_these2 = sorted(freq2, key=freq2.get, reverse=True)[:size]
        else:
            keep_these1 = words1[:size]  # just sort by index
            keep_these2 = words2[:size]

        space1, voc1 = utils.select_subspace(space1, voc1, keep_these1)  # selects vectors and makes new indices
        space2, voc2 = utils.select_subspace(space2, voc2, keep_these2)

        words1 = sorted(voc1, key=voc1.get)
        words2 = sorted(voc2, key=voc2.get)

        if freq1 is not None and freq2 is not None:
            p = np.array([freq1[w] for w in words1])  # select frequencies
            q = np.array([freq2[w] for w in words2])
            p /= sum(p)  # normalize to prob. dist.
            q /= sum(q)

        else:
            p = ot.unif(len(words1))
            q = ot.unif(len(words2))

        return space1, space2, voc1, voc2, words1, words2, p, q

    def normalize_embeddings(self, X, Y):
        """
        Copied from Alvarez-Melis & Jaakkola (2018)
        """
        if self.normalize_vecs:
            print("Normalizing embeddings with: {}".format(self.normalize_vecs))

        if self.normalize_vecs == 'whiten':
            X, Y = utils.center_embeddings(X, Y)
            return utils.whiten_embeddings(X, Y)

        elif self.normalize_vecs == 'mean':
            return utils.center_embeddings(X, Y)

        elif self.normalize_vecs == 'both':
            self.solver.normalized = True
            X, Y = utils.center_embeddings(X, Y)
            return utils.scale_embeddings(X, Y)

        else:
            print('Warning: no normalization')


    def prob_dists(self, x_freq:Dict[str,float]=None, y_freq:Dict[str,float]=None,
                   dist_shape:str=None, size:int=None) -> (np.ndarray, np.ndarray):
        """
        Partly taken from Alvarez-Melis & Jaakkola (2018)
        Compute marginal distributions.
        This method presupposes the following attirbutes to be set:
        src_words, trg_words
        :param dist_shape: one of ['uniform', 'custom', 'zipf']
        """
        dist_shape = dist_shape if dist_shape in default.DIST_SHAPES else self.distribs
        size = size if size else self.size if self.size else min(len(x_freq),len(y_freq))

        if dist_shape == 'uniform':
            p = ot.unif(size)
            q = ot.unif(size)
        elif dist_shape == 'zipf':
            p = utils.zipf_init('en', size)
            q = utils.zipf_init('en', size)
        elif dist_shape == 'custom' and x_freq and y_freq:
            p = np.array([x_freq.get(w,0) for w in self.src_words])
            q = np.array([y_freq.get(w,0) for w in self.trg_words])
            p /= np.sum(p)
            q /= np.sum(q)
        else:
            raise ValueError("Unable to compute p/q: use one of default.DIST_SHAPES"
                             " and provide frequencies for the option 'custom'.")

        return p, q

    def compile_config(self) -> utils.ConfigReader:
        cfg = {"pretrained_loc": self.pretrained_loc ,
               "out_absdir":     self.out_absdir ,
               "score_type":     self.score_type,
               "adjust":         self.adjust ,
               "metric":         self.metric ,
               "normalize_vecs": self.normalize_vecs,
               "normalize_dists":self.normalize_dists,
               "distribs":       self.distribs,
               "share_vocs":     self.share_vocs,
               "size":           self.size
               }
        cfg.update({"opt_"+k:v for k,v in self.opt_config.items()})
        cfg.update({"fit_"+k:v for k,v in self.fit_config.items()})

        return utils.ConfigReader("", param_dict=cfg)








def procrustes(X:np.ndarray, Y:np.ndarray) -> np.ndarray:
    """
    Copied from Alvarez-Melis & Jaakkola (2018).
    Solves the classical orthogonal procrustes problem, i.e.

                min_P ||X - YP'||_F subject to P'P=I,

    where X, Y are n x d and P is d x d. (Note that matrices are given as
    rows of observations, as is common in python ML settings, though not in
    Linear Algebra formulation)
    :param X: embedding space to be projected
    :param Y: embedding space to be projected onto
    :return: np.ndarray -- projection matrix from X onto Y
    """
    # U is a rotation, Sigma is for scaling, Vt is another rotation.
    U, _, Vt = np.linalg.svd(X.T @ Y)  # M is supposedly n * m
    # After transposing, P is m * n, so can right multiply Y
    return U @ Vt


    # the following is taken from Jacob Eisenstein, but it's wrong.
    # U, Sigma, Vt = np.linalg.svd(Y.dot(X.T))
    # return U.dot(Vt) # this is the Omega = the projection matrix P

def score_scores(coupling:np.ndarray, voc1:List[str], voc2:List[str],
                 method:str='entropy', top_n:int=0) -> (Dict[str,float],Dict[str,float]):
    """
    Applies re-scoring to the coupling scores. Some results are presented in the
    Appendix of the Thesis.
    For each word in the coupling (all rows, all columns), compute a score expressing the
    top score's margin of 'victory' over other scores.
    The method 'prominence' relates the best score to the second best score.
    :param method: str -- one of 'entropy', 'prominence', 'mean'
    :param voc1: List[str] -- words of the source space, sorted by index in the space
    :return: Dict[str,float], Dict[str,float] -- words and associated entorpy values
    """
    top_n = 2 if method == 'prominence' else min(coupling.shape) if top_n == 0 else top_n

    if method == 'csls':
        coupling = utils.csls(coupling, knn=top_n if top_n > 0 else 5)

    # select scores to look at (sorting columns doesn't disturb the rows' order)
    scores1 = np.array([sorted(row, reverse=True)[:top_n] for row in coupling])
    scores2 = np.array([sorted(row, reverse=True)[:top_n] for row in coupling.T])

    if method == 'entropy':
        # normalize
        margins1 = np.sum(scores1, axis=1)
        margins2 = np.sum(scores1, axis=1)
        normed_scores1 = [scores / m for scores, m in zip(scores1, margins1)]
        normed_scores2 = [scores / m for scores, m in zip(scores2, margins2)]
        # calculate entropy
        scores1 = [-sum(scores * np.log(scores)) for scores in normed_scores1]
        scores2 = [-sum(scores * np.log(scores)) for scores in normed_scores2]

    elif method == 'prominence':
        scores1 = [pair[0] / pair[1] for pair in scores1]
        scores2 = [pair[0] / pair[1] for pair in scores2]

    elif method == 'mean':
        scores1 = [s[0] / (s.mean()) for s in scores1]
        scores2 = [s[0] / (s.mean()) for s in scores2]

    else:
        print(f"WARNING in score_scores(): invalid re-scoring method '{method}'.")

    return {w: e for w, e in zip(voc1, scores1)}, {w: e for w, e in zip(voc2, scores2)}

