"""
Optimize couplings, get the corresponding translation pairs,
and dump everything to one directory (including times and config).
"""


import utils
import default
import GWOT

import time
import os

import numpy as np
import datetime

import matplotlib.pyplot as plt


take_time = utils.Timer()

config_abspath = utils.loop_input(rtype=str, default="config/optimize_couplings.cfg",
                                  msg="Path to the configuration file")
cfg = utils.ConfigReader(config_abspath)


for path in [cfg("out_absdir"), cfg("visuals_absdir")]:
    if not os.path.exists(path):
        os.makedirs(path)

# ensure that no statistics get overwritten and print the header of the stats file
stats_file_abspath = cfg("out_absdir")+"statistics"
if os.path.exists(stats_file_abspath):
    stats_file_abspath = stats_file_abspath + "_"+str(datetime.datetime.fromtimestamp(time.time()).isoformat())

with open(stats_file_abspath, "w") as f:
    f.write("\t".join("year1 year2 size distribs"
                      "pairs matches mismatches "
                      "mu_matches med_matches "
                      "mu_mismatches med_mismatches "
                      "voc_overlap gw_dist".split()))


if cfg("space_pairs") == None:
    space_pairs = default.SPACE_PAIR_SELECTION
else:
    space_pairs = cfg("space_pairs")

default_coupling_size = cfg("size") # stores the size separately in order to work for multiple iterations

# Go through all of the pairings
run = 1
for year1, year2 in space_pairs:

    src_spacefile = cfg("spaces_absdir") + "rsc-all-corr-" + str(year1)[:-1] + ".txt"
    trg_spacefile = cfg("spaces_absdir") + "rsc-all-corr-" + str(year2)[:-1] + ".txt"

    print(f"\n\n===== NEW RUN: {run} =====")
    print(f"   ({year1}, {year2}, max. size: {cfg('size')})\n")
    run += 1

    out_dir = cfg("out_absdir") + str(year1) + "_" + str(year2) + "/"

    # 1. Data Input & Initialization
    X, voc_x = utils.load_space(src_spacefile)
    Y, voc_y = utils.load_space(trg_spacefile)
    take_time.again("loading spaces")

    # reduce coupling size if the spaces are not large enough
    max_coupling_size = min(X.shape[0], Y.shape[0]) # square couplings -> min()
    if default_coupling_size > max_coupling_size:
        cfg.set("size", max_coupling_size)


    # frequencies are needed at least for selection of vectors
    c1 = utils.rsc_freqfile_column(year1)
    c2 = utils.rsc_freqfile_column(year2)
    freq_x, freq_y = utils.get_freqdists_from_file(
                           cfg("freq_file_abspath"),
                           c1, c2,
                           words1=sorted(voc_x, key=voc_x.get),
                           words2=sorted(voc_y, key=voc_y.get),
                           log_flat_base='e')
    take_time.again("frequency information")

    gwot = GWOT.GWOT(cfg, voc_x, voc_y, x_freq=freq_x, y_freq=freq_y)
    take_time.again("aligner initialization")

    diff_x = set(voc_x.keys()).difference(set(freq_x.keys()))
    diff_y = set(voc_y.keys()).difference(set(freq_y.keys()))

    # 2. Optimization
    print(f"\n\nOPTIMIZING for ({year1},{year2}) on {cfg('size')} vectors with {cfg('distribs')} p/q.\n")
    gwot.fit(X,Y, voc_x, voc_y)
    take_time(f"optimizing_{year1}_{year2}_{cfg('size')}")

    scored_pairs = gwot.scored_mutual_nn(gwot.scores, gwot.src_words, gwot.trg_words)

    matches, mismatches = utils.matchbins(scored_pairs, magnify=1e4)
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
    utils.dump_coupling(gwot.coupling, gwot.src_words, gwot.trg_words,
                        out_dir, cfg=gwot.compile_config())
    take_time.again(f"dumping_couplings")

    with open(out_dir+"pairs", "w") as f:
        f.write("\n".join([w1+"\t"+w2+"\t"+str(scored_pairs[(w1,w2)])
                           for w1,w2 in sorted(scored_pairs, key=scored_pairs.get, reverse=True)]))

    # Create and save a match/mismatch histogram
    plt.hist([matches, mismatches], label=['match', 'mismatch'],
             cumulative=False, histtype="step", bins=100)
    plt.xlabel('x 1e-4')
    plt.savefig(cfg("visuals_absdir")+"hist_"
                + str(year1) + "_" + str(year2) + "_"
                + str(cfg('size'))+ "_" + str(cfg('distribs')) + ".png")
    plt.close()

    # Write statistics to one file
    with open(stats_file_abspath, "a") as f:
        f.write("\n"+"\t".join([str(v) for v in [year1, year2, cfg('size'), cfg('distribs'),
                                            n_pairs, n_matches, n_mismatches,
                                            mu_matches, med_matches,
                                            mu_mismatches, med_mismatches,
                                            voc_overlap, gwot.solver.history[-1][1]]]))

with open(cfg('out_absdir')+"times", "w") as f:
    take_time.total()
    f.write(str(take_time))

print("\ntimes: ")
print(take_time)
print("\ndone.\n")


