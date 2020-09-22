"""
This script optimizes (and dumps) 1K couplings on all pairings of RSC spaces
and logs string match statistics of all couplings in a single file.
"""

import os
import numpy as np

import utils
import GWOT
import default


config_abspath = utils.loop_input(rtype=str, default="config/full_RSC_pairup_nonshare.cfg",
                                  msg="path and name of the configuration file")


take_time = utils.Timer()
cfg = utils.ConfigReader(config_abspath)

if not os.path.exists(cfg("out_absdir")):
    os.makedirs(cfg("out_absdir"))

# Print the header of the stats file
stats_file_abspath = cfg("out_absdir")+"statistics"
with open(stats_file_abspath, "w") as f:
    f.write("\t".join("year1 year2 distribs size "
                      "pairs matches mismatches "
                      "mu_matches med_matches "
                      "mu_mismatches med_mismatches voc_overlap".split()))

# Pair up each space with all its previous spaces
year_pairs = []
for i_2, year2 in enumerate(default.RSC_YEARS):
    for year1 in default.RSC_YEARS[:i_2]:
        year_pairs.append((year1, year2))

# Go through all of the pairings
run = 1
for year1, year2 in year_pairs:

    src_spacefile = cfg("spaces_absdir") + "rsc-all-corr-" + str(year1)[:-1] + ".txt"
    trg_spacefile = cfg("spaces_absdir") + "rsc-all-corr-" + str(year2)[:-1] + ".txt"

    # 1. Data Input & Initialization
    X, voc_x = utils.load_space(src_spacefile)
    Y, voc_y = utils.load_space(trg_spacefile)
    take_time.again("loading spaces")

    c1 = utils.rsc_freqfile_column(year1)
    c2 = utils.rsc_freqfile_column(year2)
    freq_x, freq_y = utils.get_freqdists_from_file(
        cfg("freq_file_abspath"),
        c1, c2,
        words1=sorted(voc_x, key=voc_x.get),
        words2=sorted(voc_y, key=voc_y.get),
        log_flat_base='e')
    take_time.again("frequency information")


    # Go through the various parameter options
    for dist_param in cfg("distribs_range"):

        print(f"\n\n===== NEW RUN: {run} =====\n")
        run += 1

        cfg.set("distribs", dist_param)

        coupling_out_dir = cfg("out_absdir") + str(dist_param) + "/" +  str(year1) + "_" + str(year2) + "/"
        # make sure that all output directories exist
        if not os.path.exists(coupling_out_dir):
            os.makedirs(coupling_out_dir)
        take_time.again("parameter input")

        gwot = GWOT.GWOT(cfg, voc_x, voc_y, x_freq=freq_x, y_freq=freq_y)
        take_time.again("aligner initialization")


        # 2. Optimization
        print(f"\n\nOPTIMIZING for ({year1},{year2}) with {dist_param} p/q.\n")
        gwot.fit(X,Y, voc_x, voc_y)
        take_time.again("optimization")

        scored_pairs = gwot.scored_mutual_nn(gwot.scores, gwot.src_words, gwot.trg_words)

        matches, mismatches = utils.matchbins(scored_pairs)
        n_pairs = len(scored_pairs)

        n_matches = len(matches)
        mu_matches = np.mean(matches)
        med_matches = np.median(matches)

        n_mismatches = len(mismatches)
        mu_mismatches = np.mean(mismatches)
        med_mismatches = np.median(mismatches)

        voc_overlap = len(set(gwot.src_words).intersection(set(gwot.trg_words)))

        take_time.again("translation pairs")


        # 3. Output
        utils.dump_coupling(gwot.coupling, gwot.src_words, gwot.trg_words, coupling_out_dir)
        take_time.again("output coupling")

        with open(coupling_out_dir+"config.cfg", "w") as f:
            f.write(str(gwot.compile_config()))

        with open(stats_file_abspath, "a") as f:
            f.write("\n"+"\t".join([str(v) for v in [year1, year2,
                                                dist_param, cfg("size"),
                                                n_pairs, n_matches, n_mismatches,
                                                mu_matches, med_matches,
                                                mu_mismatches, med_mismatches,
                                                voc_overlap]]))

take_time.total()
with open(cfg("out_absdir")+"times", "w") as f:
    f.write(str(take_time))