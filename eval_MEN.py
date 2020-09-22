"""
This script evaluates an embedding space with the MEN test set in a word similarity task.
(Bruni et al, 2014: Multimodal Distributional Semantics.)
"""

import numpy as np
from scipy import stats
from math import sqrt
import utils
import default
import sys

MEN_file = "data/MEN_dataset_natural_form_full"


def spearman(x,y):
	# Note: this is scipy's spearman, without tie adjustment
	return stats.spearmanr(x, y)[0]

def readMEN(annotation_file):
	pairs=[]
	humans=[]
	with open(annotation_file,'r') as f:
		for l in f:
			l=l.rstrip('\n')
			items=l.split()
			pairs.append((items[0],items[1]))
			humans.append(float(items[2]))
	return pairs, humans

def cosine_similarity(v1, v2):
	"""
    :param v1: ndarray[float] -- vector 1
    :param v2: ndarray[float] -- vector 2
    :return: float -- cosine similarity of v1 and v2 (between 0 and 1)
    """
	if len(v1) != len(v2):
		return 0.0
	num = np.dot(v1, v2)
	den_a = np.dot(v1, v1)
	den_b = np.dot(v2, v2)

	return num / (sqrt(den_a) * sqrt(den_b))

def compile_similarity_lists(dm_dict, annotation_file):
	pairs, humans=readMEN(annotation_file)
	system_actual=[]
	human_actual=[]
	eval_pairs=[]

	for i in range(len(pairs)):
		human=humans[i]
		a,b=pairs[i]
		if a in dm_dict and b in dm_dict:
			cos=cosine_similarity(dm_dict[a],dm_dict[b])
			system_actual.append(cos)
			human_actual.append(human)
			eval_pairs.append(pairs[i])

	return eval_pairs, human_actual, system_actual

def compute_men_spearman(dm_dict, annotation_file):
	eval_pairs, human_actual, system_actual = compile_similarity_lists(dm_dict, annotation_file)
	count = len(eval_pairs)
	sp = spearman(human_actual,system_actual)
	return sp,count


if __name__ == '__main__':

	spacename = utils.loop_input(rtype=str, default="", msg="name of spaces' directory")
	spacefile_stub = "data/"+spacename+"/"

	year = utils.loop_input(rtype=int, msg="decade of the space to be evaluated (type 0 to evaluate all)")
	spaceyears = default.RSC_YEARS if year == 0 else [year]

	outfile = utils.loop_input(rtype=str, default="outputs/MEN/"+spacename, msg="Where to log the results?")


	results = []

	for year in spaceyears:
		spacefile = spacefile_stub + "rsc-all-corr-"+str(year)[:3]+".txt"
		X, voc_x = utils.load_space(spacefile)
		space_dict = {k:X[voc_x[k]] for k in voc_x.keys()}

		rho, instances = compute_men_spearman(space_dict, MEN_file)
		print(f"evaluation of year {year} on {instances:>3} pairs: {rho:<5.4f}")

		results.append((year, instances, rho))

	with open(outfile, "w") as f:
		f.write("year   instances   rho\n")
		for year, instances, rho in results:
			f.write(f"{year}   {instances:>3}   {rho:<6.5f}\n")











