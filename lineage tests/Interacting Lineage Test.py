import numpy as np
import pylab as plt
from bioscrape.lineage import LineageModel
from bioscrape.lineage import LineageVolumeSplitter
from bioscrape.lineage import py_SimulateInteractingCellLineage
from bioscrape.lineage import py_SimulateSingleCell

k1 = 1.0
k2 = 50
K = 20
g = .01
d = .01

species = ["S", "X"]
rxn1 = [["X"], ["S", "X"], "massaction", {"k":k1}]
rxn2 = [["S", "X"], ["X"], "massaction", {"k":d}]
rxns = [rxn1, rxn2]
x0 = {"S": 0, "X":1}

#Instantiate Model
M = LineageModel(species = species, reactions = rxns, initial_condition_dict = x0)

vsplit = LineageVolumeSplitter(M)

M.create_division_rule("deltaV", {"threshold":4}, vsplit)
M.create_volume_event("linear volume", {"growth_rate":g}, 
	"hillnegative", {"k":k2, "s1":"S", "n":2, "K":K})
M.py_initialize()

timepoints = np.arange(0, 50, .001)
global_sync_period = 5
N = 1
lineage = py_SimulateInteractingCellLineage(timepoints, global_sync_period, models = [M], 
	initial_cell_states = [N], global_species = [], global_volume = 0)



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

color_list = []
for i in range(sch_tree_length):
	color_list.append((i/sch_tree_length, 0, 1.-i/sch_tree_length))


plt.figure(figsize = (10, 10))
"""
plt.subplot(211)
plt.title(r"$\emptyset \leftrightarrow S$    $P(Grow) = k \frac{1}{S^2+400}$")
counts = [len(sch_list) for sch_list in sch_tree]
plt.plot(range(len(counts)), counts)
plt.ylabel("Cell Count (total ="+str(sum(counts))+")")
plt.xlabel("Generation")

plt.subplot(412)
plt.ylabel("S per Cell")
for i in range(sch_tree_length):
    for sch in sch_tree[i]:
        df = sch.py_get_dataframe(Model = M)
        plt.plot(df["time"], df["S"], color = color_list[i])"""


plt.subplot(211)

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


        totalX[start_ind:end_ind+1] += df["X"][:len(df["X"])]

plt.plot(timepoints, totalX, "--", color = "black", label = "total X")

plt.subplot(212)
plt.ylabel("Volume (of each cell)")
for i in range(sch_tree_length):
    for sch in sch_tree[i]:
        df = sch.py_get_dataframe(Model = M)
        plt.plot(df["time"], df["volume"], color = color_list[i])



plt.show()