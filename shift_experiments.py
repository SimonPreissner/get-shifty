"""
Perform shift detection experiments on a SpacePair object.
For this, a SpacePair config is needed which contains the paths to the data and
possibly previously computed couplings.
"""

import os
import numpy as np

from SpacePair import SpacePair
import default
import utils

import exp_unsup
import exp_distech


#========== PARAMETER INPUT ===================

print("Perform Shift Detection Experiments.")
print("Be sure to have trained a large coupling for the space pair!")
print("Parameter Input:\n")

yearpair_is_valid = False
while not yearpair_is_valid:
    source_year = utils.loop_input(rtype=int, default=None, msg="source year")
    target_year = utils.loop_input(rtype=int, default=None, msg="target year")

    if source_year in default.RSC_YEARS \
            and target_year in default.RSC_YEARS\
            and source_year < target_year:
        yearpair_is_valid = True
    else:
        print(f"Invalid input: both of {source_year} or {target_year} must be in "
              f"{default.RSC_YEARS} and {source_year} < {target_year} must be true.")
        continue

    if (source_year, target_year) not in default.SPACE_PAIR_SELECTION:
        print("Watch out: your pairing is not in the space pair selection;"
              "it might be required to optimize a large coupling (-> long runtime).")

config_relpath = utils.loop_input(rtype="filepath", msg="relative path to the config file (in quotation marks)")



experiments = []
exp_in = ""
print("Possible experiments:")
print(default.SHIFT_EXPERIMENTS)
while exp_in is not None:
    exp_in = utils.loop_input(rtype=str,
                              msg="add an experiment to be performed or finish "
                                  "selection with empty input")
    if exp_in == None:
        if not experiments:
            experiments = ["unsup_mono"]
            break
        else:
            break
    elif exp_in in default.SHIFT_EXPERIMENTS:
        experiments.append(exp_in)
    else:
        print(f"Invalid experiment name '{exp_in}'.")
print(f"Performing the following experiments:\n   {experiments}")


#========== CONFIGURATION SETUP ===================
take_time = utils.Timer()
cfg = utils.ConfigReader(config_relpath)

absdir = cfg("root_abspath")

cfg.set("source_year", source_year)
cfg.set("target_year", target_year)
yearstring = str(source_year) + "_" + str(target_year)

outdir = absdir + "outputs/" + cfg("subproject_dir") + yearstring + "/"
visuals_dir = absdir + "visuals/" + cfg("subproject_dir") + yearstring + "/"
for path in [outdir, visuals_dir]:
    if not os.path.isdir(path):
        os.makedirs(path)

# load spaces
cfg.set("source_space_relpath", absdir + cfg("spaces_reldir") + "rsc-all-corr-" + str(source_year)[:-1] + ".txt")
cfg.set("target_space_relpath", absdir + cfg("spaces_reldir") + "rsc-all-corr-" + str(target_year)[:-1] + ".txt")

# Compose file paths to the large coupling
# this expects a pre-trained coupling in the yearpair folder of outputs/large_couplings/
if os.path.isdir(cfg("large_couplings_reldir") + yearstring):
    cfg.set("translation_coupling_pretrained_reldir", cfg("large_couplings_reldir") + yearstring + "/")
    cfg.set("translation_coupling_save_reldir", cfg("translation_coupling_pretrained_reldir"))
    cfg.set("translation_coupling_config_relpath", cfg("translation_coupling_pretrained_reldir") + "config.cfg")
else:
    print(f"WARNING: Expected a pre-trained coupling for {yearstring} in "
          f"{cfg('large_couplings_reldir') + yearstring}, but couldn't find it."
          f"Continuing with dummy config from 'config/translation_aligner_example.cfg'"
          f"and non-optimized large coupling.")
    cfg.set("translation_coupling_config_relpath", "config/translation_aligner_example.cfg")

# compile file paths to the small coupling
# this expects a pre-trained coupling in the yearpair folder of outputs/small_couplings/
if os.path.isdir(cfg("small_couplings_reldir") + yearstring):
    cfg.set("projection_coupling_pretrained_reldir", cfg("small_couplings_reldir") + yearstring + "/")
    cfg.set("projection_coupling_save_reldir", cfg("projection_coupling_pretrained_reldir"))
    cfg.set("projection_coupling_config_relpath", cfg("projection_coupling_pretrained_reldir") + "config.cfg")
else:
    if cfg.params.get("projection_coupling_config_relpath", None) is None:
        print(f"WARNING: Unable to find a pre-trained small coupling "
              f"with parameter 'projection_coupling_pretrained_reldir' "
              f"or a path to a config for training "
              f"with parameter 'projection_coupling_config_relpath'."
              f"Continuing with default config: 'config/projection_aligner_example.cfg'")
        cfg.set("projection_coupling_save_reldir", cfg("small_couplings_reldir") + yearstring + "/")
        cfg.set("projection_coupling_config_relpath", 'config/projection_aligner_example.cfg')


for path in [cfg("translation_coupling_save_reldir"), cfg("projection_coupling_save_reldir")]:
    if not os.path.isdir(path):
        os.makedirs(path)

# compile file paths to the words of interest
for name,value in cfg.params.items():
    if name.startswith("woi_") and name != "woi_relpath":
        cfg.set(name, cfg("woi_relpath")+value)


print("\nWorking with the following parameters:")
print(cfg)
print("\n")
config_savepath = outdir + "spacepair.cfg"
with open(config_savepath, "w") as f:
    f.write(str(cfg))

take_time("config building")
#========== OBJECT INSTANTIATION ===================


# Make a SpacePair object
sp = SpacePair.from_config(config_savepath, init_all=True)

# optimize the small coupling and get P if not already done
if sp.gwot2.coupling is None:
    sp.gwot2.fit(sp.X, sp.Y, sp.voc_x, sp.voc_y, proj_limit=cfg("projection_matrix_size"))
    sp.P = sp.gwot2.mapping # get Projection matrix
    utils.dump_coupling(sp.gwot2.coupling,
                        sp.gwot2.src_words,
                        sp.gwot2.trg_words,
                        cfg("projection_coupling_save_reldir"),
                        cfg=sp.gwot2.compile_config())

if cfg("use_projection") is False:
    print("not projecting because the config says so...")
    PX = sp.X
else:
    PX = np.array([sp.P.dot(x) for x in sp.X])


take_time("object instantiation")



if "all" in experiments:
    experiments = default.SHIFT_EXPERIMENTS

# UNSUPERVISED BILINGUAL EXPERIMENT
if "unsup_bi" in experiments:
    ename = "unsup_bi"
    print(f"\n\nRunning experiment '{ename}'...\n")
    exp_unsup.run(sp, PX,
                  monolingual=False,
                  ap_source=cfg("ap_on_source_vectors"),
                  min_count=cfg("exp_unsup_min_wordcount"),
                  spaces_mincount=cfg("exp_unsup_spaces_mincount"),
                  dir_k=cfg("exp_unsup_clusterlabels"),
                  dist_nbs=cfg("exp_unsup_neighbors"),
                  pairdist_csls=cfg("exp_unsup_use_csls"),
                  out_dir=outdir + ename + "/")
    print(f"total time taken (seconds): {take_time(f'experiment: {ename}')}")


# UNSUPERVISED MONOLINGUAL EXPERIMENT
if "unsup_mono" in experiments:
    ename = "unsup_mono"
    print(f"\n\nRunning experiment '{ename}'...\n")
    exp_unsup.run(sp, PX,
                  monolingual=True,
                  ap_source=      cfg("ap_on_source_vectors"),
                  min_count=      cfg("exp_unsup_min_wordcount"),
                  spaces_mincount=cfg("exp_unsup_spaces_mincount"),
                  dist_nbs=       cfg("exp_unsup_neighbors"),
                  dir_k=          cfg("exp_unsup_clusterlabels"),
                  pairdist_csls=  cfg("exp_unsup_use_csls"),
                  out_dir=outdir + ename + "/")
    print(f"total time taken (seconds): {take_time(f'experiment: {ename}')}")


# SUPERVISED EXPERIMENT: DISCOURSE WORDS VS. TECHNICAL TERMS
if "dis_tech" in experiments:
    ename = "dis_tech"
    experiment_subdir = outdir + ename + "/"
    print(f"\n\nRunning experiment '{ename}'...\n")

    ap_first = cfg("ap_on_source_vectors")

    distech_wordlists = {"woi_3_clusters_dis":[],  # dis
                         "woi_3_clusters_tech":[], # tech
                         "woi_chemistry":[],       # tech
                         "woi_galaxy":[],          # tech
                         "woi_ing-that":[],        # dis
                         "woi_it-adj":[],          # dis
                         "woi_md-vb-vnn":[],       # both
                         "woi_mismatches":[]}      # both

    if "woi_3_clusters" in cfg.params:
        dis_words, tech_words = exp_distech.read_3clusters(cfg("woi_3_clusters"))
        distech_wordlists["woi_3_clusters_dis"] = dis_words
        distech_wordlists["woi_3_clusters_tech"] = tech_words

        exp_distech.run(sp, PX, dis_words, "3clusters_dis", ap_source=ap_first,
                        out_dir=experiment_subdir, options=cfg)
        exp_distech.run(sp, PX, tech_words, "3clusters_tech",
                        out_dir=experiment_subdir, options=cfg)
        #take_time("woi_3_clusters")

    if "woi_chemistry" in cfg.params:
        woi = exp_distech.read_wordlist(cfg("woi_chemistry"), column=0)
        distech_wordlists["woi_chemistry"] = woi

        exp_distech.run(sp, PX, woi, "chemistry_tech", ap_source=ap_first,
                        out_dir=experiment_subdir, options=cfg)
        #take_time("woi_chemistry")

    if "woi_galaxy" in cfg.params:
        woi = exp_distech.read_wordlist(cfg("woi_galaxy"), column=0)
        distech_wordlists["woi_galaxy"] = woi

        exp_distech.run(sp, PX, woi, "galaxy_tech", ap_source=ap_first,
                        out_dir=experiment_subdir, options=cfg)
        #take_time("woi_galaxy")

    if "woi_ing-that" in cfg.params:
        woi = exp_distech.read_wordlist(cfg("woi_ing-that"), column=1)
        distech_wordlists["woi_ing-that"] = woi

        exp_distech.run(sp, PX, woi, "ing-that_dis", ap_source=ap_first,
                        out_dir=experiment_subdir, options=cfg)
        #take_time("woi_ing-that")

    if "woi_it-adj" in cfg.params:
        woi = exp_distech.read_wordlist(cfg("woi_it-adj"), column=1)
        distech_wordlists["woi_it-adj"] = woi

        exp_distech.run(sp, PX, woi, "it-adj_dis", ap_source=ap_first,
                        out_dir=experiment_subdir, options=cfg)
        #take_time("woi_it-adj")

    if "woi_md-vb-vnn" in cfg.params:
        woi = exp_distech.read_wordlist(cfg("woi_md-vb-vnn"), column=1)
        distech_wordlists["woi_md-vb-vnn"] = woi

        exp_distech.run(sp, PX, woi, "md-vb-vnn_distech", ap_source=ap_first,
                        out_dir=experiment_subdir, options=cfg)
        #take_time("woi_md-vb-vnn")


    # run the experiment on all available discourse/technical terms jointly
    dis_words = list(set(distech_wordlists["woi_3_clusters_dis"]
                         +distech_wordlists["woi_ing-that"]
                         +distech_wordlists["woi_it-adj"]))
    tech_words = list(set(distech_wordlists["woi_3_clusters_tech"]
                          +distech_wordlists["woi_chemistry"]
                          +distech_wordlists["woi_galaxy"]))

    exp_distech.run(sp, PX, dis_words, "all_discourse", ap_source=ap_first,
                    out_dir=experiment_subdir, options=cfg)
    exp_distech.run(sp, PX, tech_words, "all_technical", ap_source=ap_first,
                    out_dir=experiment_subdir, options=cfg)


    print(f"total time taken (seconds): {take_time(f'experiment: {ename}')}")




del sp
take_time.total()
print(f"\nTotal times taken per experiment:\n{take_time}")

print("\ndone.\n")
