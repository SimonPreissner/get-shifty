# Unsupervised Detection of Semantic Shifts in Diachronic Word Embeddings

This repository contains relevant data and code that were collected and produced in 
the process of writing my Master's Thesis at Saarland University. For the thesis,
just e-mail me via the [address on GitHub](https://github.com/SimonPreissner).




### 1.Directory Overview
- `annotation/` — files related to the annotation of mismatching translation pairs.
- `config/` — configuration files as used by the utils class `ConfigReader`.
- `data/` — Embedding models, frequency counts, and words of interest.
    - e.g. `vectors1929-init-tc1-t3` — RSC vectors, with cross-corpus initialization (needs to be downloaded).
    - `words_of_interest/` — Discourse and Technical terms as well as annotated mismatches (cf. Thesis, Table 14 for references).
    - `MEN_dataset_natural_form_full` — test set for a semantic similarity task (Bruni et al. 2014, 
       [available here](https://staff.fnwi.uva.nl/e.bruni/MEN)).
- `otalign/` — code from Alvarez-Melis & Jaakkola (2018); cloned from [their github](https://github.com/dmelis/otalign). 
   Manual installation needed (cf. their README).
- `outputs` — couplings, projections, and other larger data (empty at first)
- `visuals` — graphs and figures (empty at first)




### 2. Scripts, Classes and Notebooks

#### General
- `utils.py` — convenience functions and classes
- `default.py` — meta information and default configurations
- `eval_utils.py` — functions which are frequently used during evaluation

#### Alignment 
- `GWOT.py` — wrapper class which performs optimization and alignment (cf. Chapter 4.1)
- `SpacePair.py` — combines two spaces and two aligners (cf. Chapter 4.2)
- `demo_spacepair.py` — provides basic operations with the `SpacePair` class
- `optimize_couplings.py` — optimize one or more couplings for one or more space pairs
- `full_RSC_heatmap.ipynb` — for Chapter 4.3.3
- `full_RSC_pairup.py` — for Chapter 4.3.3
- `few_anchors.ipynb` — for Chapter 4.3.6
- `eval_annotations.ipynb` — for Chapter 4.3.7

#### Shift Detection
- `shift_experiments.py` — wrapper for `exp_unsup.py` and `exp_distech.py` which holds the data
- `post_processing.py` — cf. Chapter 5.2.3
- `exp_unsup.py` — for Chapter 5.3
- `exp_distech.py` — for Chapter 5.4
- `eval_TSNE.ipynb` — for Chapters 5.2.3, 5.3.5 and 5.4.2

#### Other
- `eval_MEN.py` — for Chapter 3.2.3
- `eval_MEN.ipynb` — for Chapter 3.2.3
- `rescoring.py` — cf. Figure 29 in the Appendix
- `freq_inspections.ipynb` — take a look at corpus token counts 




### 3. Preparations
0. The code is optimized for Python 3.8; older versions may not work.
1. Install the requirements, runing the following command: `pip install -r requirements.txt`
2. Get the `otalign` module from Alvarez-Melis & Jaakkola (2018) and install it: 
   [https://github.com/dmelis/otalign](https://github.com/dmelis/otalign).
3. Get the embedding spaces [from here](http://corpora.ids-mannheim.de/openlab/diaviz1/description.html) and store them in `data/`.
4. Unzip `data/decFreq1929.zip` (these are the corpus frequencies).
5. In order to work bilingually, optimize one or more large couplings using `optimize_couplings.py`, or e-mail me ([address on GitHub](https://github.com/SimonPreissner)).

#### What to Run and How to Run it

Most scripts read a configuration file (in the `config` directory) and draw their information from there. 
Other inputs are asked for at the beginning of script execution.

- If working bilingually, use `optimize_couplings.py` to optimize a large coupling. The output destination can then be 
  written to a config for `shift_exeriments.py`.
- For shift detection, run `shift_experiments.py`; the required parameters will be asked upon execution.





### 4. Classes

#### 4.1 `SpacePair`
This class is designed to hold two spaces as well as two aligner objects to establish a connection between the spaces. The aligners operate on subsets of the two spaces, but don't hold embeddings themselves.

#### 4.2 `GWOT`
Wrapper class for `otalign.src.gw_optim.gromov_wass_solver`. Objects of this class hold pretty much every kind of information needed to optimize a coupling for a given space, _except_ the spaces themselves. 

The various parameters are ideally specified in a configuration file, because it makes initialization easy: 
```Python3
X, x_voc = utils.load_space("path/to/my/spaces/year1")
Y, y_voc = utils.load_space("path/to/my/spaces/year2")

cfg = utils.ConfigReader("path/to/my/gwot_config.cfg")
gwot = GWOT.GWOT(cfg, x_voc, y_voc)
```
If you want to initialize the aligner with frequency information, you need to provide the corresponding dictionaries to the constructor. 

If you want to change a parameter after initialization, the safest way is to retrieve the object's configuration, change it, and overwrite the object with a new one:
```
new_cfg = gwot.compile_config()
new_cfg.set("my_param", "my_value")
gwot = GWOT.GWOT(new_cfg, x_voc, y_voc)
```

For defaults and other possible/recommended parameter values, have a look at the `default` module.



### 5. Understanding the Code: Naming & Data Conventions

This project deals with semantic **spaces**. 
Vectors within these spaces can be accessed with a **vocabulary**.
The connection between two spaces are **translation pairs** on the vocabulary level and a **mapping** (or: projection matrix) on the vector level.
These two connecting elements are based on **couplings** which are the result of Gromov-Wasserstein Optimal Transport (GWOT).
For optimization, one can pass the **frequencies** of the spaces' vocabulary items in the underlying corpus to the optimizer.

##### 5.1 Prefixes
The two spaces and their alignments are modeled by the class `SpacePair`, which holds two spaces `X` and `Y` as well as the corresponding vocabularies and frequencies. 
Additionally, a `SpacePair` object holds two `GWOT` objects which do the job of optimization. They only optimize for subsets of `X` or `Y`. 

Therefore, each `GWOT` object holds subsets of the two spaces as well as corresponding lexicon information. After optimization, a `GWOT` object also holds a coupling and a mapping.

The distinction between the whole of `X` and `Y` (as handled primarily by `SpacePair`) and subsets of `X` and `Y` (as handled primarily by `GWOT`) is made by prefixes: 
- `x_` and `y_` --> correspondence to the unaltered `X` and `Y`
- `src_` and `trg_` --> correspondence to subspaces of `X` and `Y`

##### 5.2 Variable Names
- `X`, `Y`, `embs` — spaces
- `x`, `y`, `emb` — embedding (= word vector)
- `voc` — mapping from vocabulary to space index
- `words` — list of vocabulary items (usually sorted by space's indices)
- `freq` — provides frequency information from the corpus
- `p`, `q` — probability distribution over items in `words`
- `G`, `coupling` — coupling between two (sub)spaces
- `T`, `pairs` — mapping of word pairs to a 'translation' score
- `P`, `mapping` — projection matrix from one space to the other (usually from `X` to `Y`)


##### 5.3 Data Structures

Here is a sample list of variables with relevant data structures, all to be seen in dependence to a single space.
- `src_space` — `np.ndarray` of shape (m,d) with m word embeddings of d dimensions
- `src_emb` — `np.array` of length d
- `src_voc` — `Dict[str,int]` of length m, mapping a word the index of the row of the corresponding vector in the space (e.g., `spam_emb = src_space[src_voc['spam']]`).
- `src_words` — `List[str]`, usually sorted by appearance in the space
- `src_freq` — `Dict[str,float]` of length m, mapping lexical items to frequency values
- `p` — `np.array` of length m, values corresponding to the order of `src_words`
- `coupling` — `np.ndarray` of shape (m,m), rows contain translation likelihood values for a translation from `src_space` to another space  
- `pairs` — `Dict[(str,str):float]` — mutual best translations (from two spaces), mapped to their translation score
- `mapping` — `np.ndarray` of shape (d,d), used to project `src_space` onto another space





#### Foreign Contributions
Some parts of the code are copied and adapted from either [Alvarez-Melix & Jaakkola (2018)](https://github.com/dmelis/otalign)
or from [Jacob Eisenstein's NAACL tutorial on language change](https://github.com/jacobeisenstein/language-change-tutorial/blob/master/naacl-notebooks/DirtyLaundering.ipynb). I marked these foreign contributions to the best of my knowledge.




