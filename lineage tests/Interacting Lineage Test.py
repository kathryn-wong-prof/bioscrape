import numpy as np
from bioscrape.lineage import LineageModel
from bioscrape.lineage import LineageVolumeSplitter
from bioscrape.lineage import py_SimulateInteractingCellLineage
from bioscrape.lineage import py_SimulateCellLineage
from bioscrape.lineage import py_SimulateSingleCell
from bioscrape.lineage import LineageCSimInterface
from bioscrape.lineage import py_PropagateCells
from bioscrape.lineage import py_SingleCellLineage
import time as pytime

k1 = 3.333
k2 = 33.33
K = 20.2020
g = .01010101
d = 2.020202

species = ["S", "X"]
rxn1 = [[], ["S"], "massaction", {"k":k1}]
rxn2 = [["S", "X"], ["X"], "massaction", {"k":d}]
rxns = [rxn1, rxn2]
x0 = {"S": 0, "X":100}

#Instantiate Model
print("Instantiating Model")
M = LineageModel(species = species, reactions = rxns, initial_condition_dict = x0)

vsplit = LineageVolumeSplitter(M)

M.create_division_rule("deltaV", {"threshold":1.0}, vsplit)
M.create_volume_event("linear volume", {"growth_rate":g}, 
	"hillnegative", {"k":k2, "s1":"S", "n":2, "K":K})
M.py_initialize()


global_sync_period = 2.0
N = 1
sum_list = []
#lineage = 

lineage = None
cell_states = None
single_cell_states = None

for i in range(1, 16):
	maxtime = i*10
	print("maxtime", maxtime, "i", i)
	timepoints = np.arange(0, maxtime, .05)

	print("Beginning Simulation", i, "for", maxtime)
	#interface = LineageCSimInterface(M)
	
	s = pytime.clock()
	#lineage = py_SimulateCellLineage(timepoints, Model = M, initial_cell_states = N)
	#cell_states = py_PropagateCells(timepoints, Model = M, initial_cell_states = N)
	#single_cell_states = py_SingleCellLineage(timepoints, Model = M)
	lineage_list = py_SimulateInteractingCellLineage(timepoints, global_sync_period, model_list = [M],initial_cell_states = [N], global_species = ["S"], global_volume = 1000, average_dist_threshold = 10.0)
	lineage = lineage_list[0]
	#result = py_SimulateSingleCell(timepoints[10:], Model = M)	

	e = pytime.clock()
	print("Simulation", i, "complete in", e-s, "s")

	if i > 0:
		if lineage is not None:
			print("Building Tree")
			sch_tree = [[]]
			sch_tree_length = 1
			for i in range(lineage.py_size()):
				sch = lineage.py_get_schnitz(i)
				if sch.py_get_parent() == None:
					sch_tree[0].append(sch)
				else:
					for j in range(len(sch_tree)):
						parent = sch.py_get_parent()
						if parent in sch_tree[j]:
							if len(sch_tree)<= j+1:
								sch_tree.append([])
								sch_tree_length += 1
							sch_tree[j+1].append(sch)
			counts = [len(sch_list) for sch_list in sch_tree]
			print("counts", counts)
			sum_list.append((maxtime, sum(counts), e-s))

		if cell_states is not None:
			sum_list.append((maxtime, len(cell_states), e-s))
#raise ValueError()
import pylab as plt
if len(sum_list) > 1:
	plt.figure()
	plt.subplot(121)
	plt.plot([e[0] for e in sum_list], [e[1] for e in sum_list])
	plt.xlabel("max simulation time")
	plt.ylabel("Total Cells Simulated")

	plt.subplot(122)
	plt.plot([e[1] for e in sum_list], [e[2] for e in sum_list])
	plt.xlabel("Total Cells Simulated (Lineage)\nOR\nFinal Cells Returned (Propogate)")
	plt.ylabel("CPU runtime (s)")

if single_cell_states is not None:
	plt.figure()
	plt.subplot(131)
	plt.title("volume")
	plt.plot(single_cell_states["time"], single_cell_states["volume"])
	
	import pandas as pd
	#with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
	#	print("**volume**\n", single_cell_states["volume"])
	#	print("**time**\n", single_cell_states["time"])

	plt.subplot(132)
	plt.title("X")
	plt.plot(single_cell_states["time"], single_cell_states["X"])

	plt.subplot(133)
	plt.title("S")
	plt.plot(single_cell_states["time"], single_cell_states["S"])

if cell_states is not None:
	plt.figure()
	plt.subplot(131)
	plt.title("volume histogram")
	plt.hist(cell_states["volume"])

	plt.subplot(132)
	plt.title("X histogram")
	plt.hist(cell_states["X"])

	plt.subplot(133)
	plt.title("S histogram")
	plt.hist(cell_states["S"])

if lineage is not None:
	color_list = []
	for i in range(sch_tree_length):
		color_list.append((i/sch_tree_length, 0, 1.-i/sch_tree_length))

	import pylab as plt
	plt.figure(figsize = (10, 10))
	print("plotting")

	plt.subplot(411)
	plt.title(r"$\emptyset \leftrightarrow S$    $P(Grow) = k \frac{1}{S^2+400}$")

	plt.plot(range(len(counts)), counts)
	plt.ylabel("Cell Count (total ="+str(sum(counts))+")")
	plt.xlabel("Generation")

	print("sch_tree_length", sch_tree_length)
	plt.subplot(412)
	plt.ylabel("S per Cell")
	for i in range(sch_tree_length):
		for sch in sch_tree[i]:
			df = sch.py_get_dataframe(Model = M)
			plt.plot(df["time"], df["S"], color = color_list[i])


	plt.subplot(413)

	plt.ylabel("X per cell")
	totalX = np.zeros(len(timepoints))


	for i in range(sch_tree_length):
		for sch in sch_tree[i]:
			df = sch.py_get_dataframe(Model = M)
			start_ind = np.where(timepoints >= df["time"][0])
			start_ind = start_ind[0][0]
			end_ind = np.where(timepoints >= df["time"][len(df["time"])-1])[0][0]

			plt.plot(df["time"], df["X"], color = color_list[i])
			plt.plot(df["time"][len(df["time"])-1], df["X"][len(df["time"])-1], "x", color = color_list[i])
			plt.plot(df["time"][0], df["X"][0], "o", color = color_list[i])


			#totalX[start_ind:end_ind+1] += df["X"][:len(df["X"])]

	#plt.plot(timepoints, totalX, "--", color = "black", label = "total X")

	plt.subplot(414)
	plt.ylabel("Volume (of each cell)")
	for i in range(sch_tree_length):
	    for sch in sch_tree[i]:
	        df = sch.py_get_dataframe(Model = M)
	        plt.plot(df["time"], df["volume"], color = color_list[i])

plt.show()
