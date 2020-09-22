"""
This class implements a pair of embedding spaces and their connection.
It utilizes the code of Alvarez-Melis & Jaakkola (2018) in several places.
The original implementation can be found on https://github.com/dmelis/otalign
I mark all adopted code as clearly as possible.
"""

from typing import List, Dict, Tuple, Any
import math

import numpy as np


from GWOT import GWOT

import utils
import default

class SpacePair():
    """
    Information about notation:
    "source" == "_s" == "_1" == "_x"
    "target" == "_t" == "_2" == "_y"
    Within this class, I use x and y to denote source- and target-related
    variables in order to avoid confusion (e.g., with indices for translation pairs '_t')
    """

    def __init__(self, src_space:np.ndarray,   trg_space:np.ndarray,
                       s_voc:Dict[str,int],    t_voc:Dict[str,int],
                       s_freq:Dict[str,float], t_freq:Dict[str,float],
                       cfg:utils.ConfigReader,
                       gwot1_config:utils.ConfigReader, gwot2_config:utils.ConfigReader,
                       init_all:bool=False):
        """
        Set up a pair of spaces with two aligners: one for translation pairs
        and one to get a projection from X to Y.
        """

        # WATCH OUT: this cfg holds the intended coupling sizes, which might
        # differ from the actual sizes if the couplings are loaded from pre-trained!
        self.cfg = cfg # a ConfigReader object

        self.X = src_space
        self.Y = trg_space

        self.nx, self.dx = self.X.shape
        self.ny, self.dy = self.Y.shape

        self.voc_x = s_voc
        self.voc_y = t_voc

        self.words_x = sorted(self.voc_x, key=self.voc_x.get)
        self.words_y = sorted(self.voc_x, key=self.voc_x.get)

        self.freq_x = s_freq
        self.freq_y = t_freq

        self.gwot1 = None # aligner for translation pairs
        self.T = {} # scored translation pairs

        self.gwot2 = None # aligner for space projection
        self.P = np.array([]) # projection matrix

        if init_all:
            self.init_working_parts(gwot1_config=gwot1_config,
                                    gwot2_config=gwot2_config)


    @classmethod
    def from_config(cls, config_file, init_all=False):
        cfg = utils.ConfigReader(config_file)

        # checks for required parameters ans builds file paths
        cfg = cls._sort_out_filepaths(cfg)

        space_x, voc_x = utils.load_space(cfg("source_space_relpath"))
        space_y, voc_y = utils.load_space(cfg("target_space_relpath"))

        # handles loading from one file or two files
        freq_x, freq_y = cls._sort_out_freq_dists(cfg, voc_x, voc_y)

        # handles input of individual parameters and default parameters
        gwot1_config = cls._sort_out_coupling_config(cfg, purpose='translation')
        gwot2_config = cls._sort_out_coupling_config(cfg, purpose='projection')

        return cls(space_x, space_y, voc_x, voc_y, freq_x, freq_y,
                   cfg, gwot1_config, gwot2_config, init_all=init_all)


    #TODO implement a method to load a pre-trained mapping

    @classmethod
    def _sort_out_filepaths(cls, cfg):
        """
        Compose file paths from the snippets given in the config.
        All file paths need to be below the specified root_abspath and need to
        be initialized from there on.
        :return: utils.ConfigReader object with complete file paths
        """
        #TODO make clear that any inputs below root_abspath have to be specified from
        # root_abspath onwards. Outputs are automatically put into data/outputs/visuals/config
        # and subproject directories
        assert ("root_abspath" in cfg.params), "root_path not in the config!"
        assert ("subproject_dir" in cfg.params), "subproject_dir not in the config!"
        assert ("source_year" in cfg.params), "source_year not in the config!"
        assert ("target_year" in cfg.params), "target_year not in the config!"
        assert ("source_space_relpath" in cfg.params), "source_space_relpath not in the config!"
        assert ("target_space_relpath" in cfg.params), "target_space_relpath not in the config!"

        # these will not be used for input; they only determine where to output to
        cfg.params.update({"data_abspath": cfg("root_abspath") + "data/" + cfg("subproject_dir")})
        cfg.params.update({"visuals_abspath": cfg("root_abspath") + "visuals/" + cfg("subproject_dir")})
        cfg.params.update({"outputs_abspath": cfg("root_abspath") + "outputs/" + cfg("subproject_dir")})

        cfg.params.update({"src_space_abspath": cfg("source_space_relpath")})
        cfg.params.update({"trg_space_abspath": cfg("target_space_relpath")})

        return cfg

    @classmethod
    def _sort_out_freq_dists(cls, cfg, voc_x:Dict[str,int], voc_y:Dict[str,int]):
        """
        Create corpus frequency counts for both spaces.
        :return: Dict[str,float], Dict[str,float] -- frequency distributions
        """
        if "freq_files" in cfg.params:
            if type(cfg("freq_files")) == list:  ff_list = cfg("freq_files")
            elif type(cfg("freq_files")) == str: ff_list = [cfg("freq_files")]  # nest it in a list
            else: ff_list = []
        else:
            ff_list = []

        sep =     cfg.params.get("freq_file_separator", '\t')
        header =  cfg.params.get("freq_file_has_header", False)
        flatten = cfg.params.get("log_flat_base", math.e)

        freq_x, freq_y = {}, {}

        if not ff_list:
            pass
        elif len(ff_list) == 1:
            col1 = utils.rsc_freqfile_column(cfg("source_year"))
            col2 = utils.rsc_freqfile_column(cfg("target_year"))
            freq_x, freq_y = utils.get_freqdists_from_file(ff_list[0],
                                                           col1, col2,
                                                           list(voc_x.keys()),
                                                           list(voc_y.keys()),
                                                           log_flat_base=flatten)
        elif len(ff_list) == 2:
            freq_x = utils.read_single_freqfile(ff_list[0], sep=sep, exclude_header=header, log_flat_base=flatten)
            freq_y = utils.read_single_freqfile(ff_list[1], sep=sep, exclude_header=header, log_flat_base=flatten)
        else:
            raise ValueError(f"Too many filepaths. Expected up to 2, but got {len(ff_list)}.")

        return freq_x, freq_y

    @classmethod
    def _sort_out_coupling_config(cls, cfg:utils.ConfigReader, purpose:str) -> utils.ConfigReader:
        """
        This method ensures that the SpacePair object initializes its couplings
        with complete parameter settings. That is, it fills the gaps in the
        config with default values from default.py
        :param purpose: should be either "translation" or "projection"
        """

        c = {}  # this will be turned into a ConfigReader object

        # Make a config during runtime
        if purpose == "custom":
            # assign the specified values, and default values to non-specified parameters
            for k,default_value in default.COUPLING_CONFIG.items():
                c[k] = cfg.params.get(k, default_value)
            return utils.ConfigReader("custom_config.cfg", param_dict=c)

        # Load a config from a file
        file_for_loading = cfg(purpose + "_coupling_pretrained_reldir")
        c.update({"pretrained_loc": file_for_loading})

        out_reldir = cfg.params.get(purpose + "_coupling_save_reldir",
                                    "outputs/" + purpose + "_coupling_default/")
        c.update({"out_absdir": cfg("root_abspath") + out_reldir})

        # if the spacepair config doesn't give a size, try the coupling config instead
        c.update({"size": cfg.params.get(purpose + "_coupling_size", None)})
        if c["size"] is None:
            if cfg.params.get(purpose+"_coupling_size",None) is None:
                if cfg.params.get(purpose + "_coupling_pretrained_reldir", None) is not None:
                    pretrained_cfg = utils.ConfigReader(cfg.params.get(purpose + "_coupling_config_relpath"))
                    c.update({"size":pretrained_cfg("size")})


        if purpose == "projection":
            if "projection_matrix_size" in cfg.params.keys():
                c.update({"max_anchors":cfg("projection_matrix_size")})
            else:
                c.update({"max_anchors":cfg("projection_coupling_size")})

        params_relpath = cfg(purpose + "_coupling_config_relpath")
        # make a default config because the path parameter is not specified
        if not params_relpath:
            print(f"WARNING: parameter {purpose}_coupling_config_relpath not found."
                  f"Continuing with default parameters for this aligner.")
            c = default.COUPLING_CONFIG
            return utils.ConfigReader(f"{purpose}_coupling_config.cfg", c)
        # load the config from the file specified in the SpacePair config
        else:
            tmp_cfg = utils.ConfigReader(cfg("root_abspath") + params_relpath)
            tmp_cfg.params.update(c)
            return tmp_cfg

    def init_working_parts(self, gwot1_config:utils.ConfigReader=None, gwot2_config:utils.ConfigReader=None):
        """
        Initializes the two GWOT objects of a SpacePair and obtain the set of
        translation pairs as well as the mapping (T and P, respectively).
        If a config is not specified, it uses the default config from default.py.
        """
        print("Initializing Gromov-Wasserstein Aligners ...")

        # initialize default configs if none are specified
        if gwot1_config is None:
            gwot1_config = self._sort_out_coupling_config(utils.ConfigReader("",{}), "custom")
            print(f"   WARNING: no config provided for aligner 1. "
                  f"Continuing with default settings.")
        if gwot2_config is None:
            gwot2_config = self._sort_out_coupling_config(utils.ConfigReader("",{}), "custom")
            print(f"   WARNING: no config provided for aligner 2. "
                  f"Continuing with default settings.")

        # initialize aligners and their optimizers
        self.gwot1 = GWOT(gwot1_config,
                          self.voc_x, self.voc_y,
                          x_freq=self.freq_x, y_freq=self.freq_y,
                          size=self.cfg("translation_coupling_size"))
        self.gwot2 = GWOT(gwot2_config,
                          self.voc_x, self.voc_y,
                          x_freq=self.freq_x, y_freq=self.freq_y,
                          size=self.cfg("projection_coupling_size"))

        print(f"\nTrying to get T and P (translation pairs and projection matrix) ...")
        self.T = self.gwot1.sort_out_scored_mutual_nn()
        self.P = self.gwot2.sort_out_mapping(self.X, self.Y, self.voc_x, self.voc_y)

    def new_aligner(self, gwot_config:utils.ConfigReader) -> GWOT:
        """
        Initialize a new GWOT object. Uses default settings for parameters which
        were left unspecified in the provided configuration. Returns the new
        aligner and the corresponding (potentially completed) configuration.
        :param gwot_config: ConfigReader (can be created from a dictionary --> see utils.py)
        :return: GWOT object
        """
        cfg = self._sort_out_coupling_config(gwot_config, "custom")
        new_gwot = GWOT(cfg, self.voc_x, self.voc_y,
                        x_freq=self.freq_x, y_freq=self.freq_y)

        new_gwot.sort_out_scored_mutual_nn()
        new_gwot.sort_out_mapping(self.X, self.Y, self.voc_x, self.voc_y)

        return new_gwot
