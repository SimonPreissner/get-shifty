"""
This script is a demo of how to initialize and use the classes SpacePair and GWOT.
"""

import numpy as np
import utils
from SpacePair import SpacePair
from GWOT import GWOT


config_abs_path = "/home/simon/Desktop/thesis/GWOT/config/space_pair_example.cfg"


# Load a SpacePair from a config
spacepair = SpacePair.from_config(config_abs_path, init_all=True)

translation_dump_dir = spacepair.gwot1.out_absdir
projection_dump_dir  = spacepair.gwot2.out_absdir

# output the two couplings
utils.dump_coupling(spacepair.gwot1.coupling,
                    spacepair.gwot1.src_voc,
                    spacepair.gwot1.trg_voc,
                    translation_dump_dir,
                    use_pickle=True,
                    cfg=spacepair.gwot1.compile_config())

utils.dump_coupling(spacepair.gwot2.coupling,
                    spacepair.gwot2.src_voc,
                    spacepair.gwot2.trg_voc,
                    projection_dump_dir,
                    use_pickle=True,
                    cfg=spacepair.gwot2.compile_config())

# Now these two couplings can be loaded individually
translation_coupling, t_src_voc, t_trg_voc = utils.load_coupling(translation_dump_dir)
projection_coupling, p_src_voc, p_trg_voc  = utils.load_coupling(projection_dump_dir)

print("done.")