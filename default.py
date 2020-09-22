"""
This file contains meta information and default configurations of the project
"""



RSC_YEARS = [1660, 1670, 1680, 1690,
             1700, 1710, 1720, 1730, 1740, 1750, 1760, 1770, 1780, 1790,
             1800, 1810, 1820, 1830, 1840, 1850, 1860, 1870, 1880, 1890,
             1900, 1910, 1920]

# cf. Chapter 4.4.1 of the thesis
SPACE_PAIR_SELECTION = [(1740,1750), (1750,1760),
                        (1680,1710), (1710,1740), (1740,1770), (1770,1800), (1800,1830), (1830,1860), (1860,1890),
                        (1700,1800), (1800,1900),
                        (1700,1900)]


COUPLING_CONFIG = {                                 # Alternatives
                    # parameters passed to the GWOT object
                    'metric': "cosine",             # 'euclidian',
                    'normalize_vecs': "both",       # 'mean', 'whiten', 'whiten_zca'
                    'normalize_dists': "mean",      # 'max', 'median'
                    'score_type': "coupling",       # #TODO fill in the rest of the options in the comments
                    'adjust': None,                 # 'csls', ...
                    'distribs': "uniform",          # 'custom', 'zipf'
                    'share_vocs':False,             # True
                    'size':1000,                    # 100 is small, 1e4
                    'max_anchors':100,              # used with small couplings (for projection)
                    # parameters to be passed to the optimizer
                    'opt_loss_fun': "square_loss",  # 'kl_loss'
                    'opt_entropic': True,           # False
                    'opt_entreg': 5e-4,             # stay within the range of e-4 (originally: 1e-4)
                    'opt_tol': 1e-9,                # no limits
                    'opt_round_g': False,           # True
                    'opt_compute_accuracy': False,  # True would require a test dict, but that's not implemented!
                    'opt_gpu': False,               # GPU optimization not tested
                    # parameters for calling fit()
                    'fit_maxiter': 300,             # no limits; normally converges within 150 iterations
                    'fit_tol': 1e-9,                # no limits
                    'fit_plot_every': 100000,       # normally 20; 'deactivate' the file spam by choosing a large value
                    'fit_print_every': 1,           # no limits
                    'fit_verbose': True,            # False
                    'fit_save_plots': None          # "/my_dir/my_optimizer_plots"
                  }


DIST_SHAPES = ['uniform', 'zipf', 'custom']

SHIFT_EXPERIMENTS = ["all",
                     "unsup_bi",
                     "unsup_mono",
                     "dis_tech"]