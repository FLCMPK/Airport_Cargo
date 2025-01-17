import numpy as np
import pandas as pd
import os
import networkx as nx
import random
import matplotlib as mp
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')
import pickle
from collections import defaultdict
from gurobipy import Model, GRB, quicksum

# ===============================Read the data===============================
# Define the path to the pickle files
path_to_B = "/Users/paulkloos/Desktop/ERASMUS/Airport/Repository/Airport_Cargo-1/B.pickle"
path_to_I = "/Users/paulkloos/Desktop/ERASMUS/Airport/Repository/Airport_Cargo-1/I.pickle"

# Read the pickle files
with open(path_to_B, 'rb') as file_B:
    Bpickle = pickle.load(file_B)

with open(path_to_I, 'rb') as file_I:
    Ipickle = pickle.load(file_I)

# Bin types
T = {value[0]: {'L': value[1][0], 'H': value[1][1], '#': value[1][2], 'C': value[1][3], 
                'cutA': 0 if value[1][4] == -1 else value[1][4], 
                'cutB': 0 if value[1][5] == -1 else value[1][5]} 
     for value in Bpickle.values()}

# Bins
B = {bin_counter: {'type': t, 'L': v['L'], 'H': v['H'], 'C': v['C'], 'cutA': v['cutA'], 'cutB': v['cutB']}
     for t, v in T.items() for bin_counter in range(1, v['#'] + 1)}

# Bins by type categorization    
B_type = defaultdict(list)
for k, v in B.items():
    B_type[v['type']].append(k)

# Items - needs to add fragility type and color based on properties
I = {k + 1: {'L': v[0], 'H': v[1], 'rotation': v[2], 'fragility': v[3], 'perishable': v[4], 'radioactive': v[5],
             'color': 'green' if v[4] == 1 else 'yellow' if v[2] == 0 else 'red' if v[5] == 1 else 'blue' if v[3] == 1 else 'gray'}
     for k, v in Ipickle.items()}

# Incompatible items (pairs) list due to item type of perishable and radioactive cannot be placed in the same bin
I_inc = [(i1, i2) for i1, v1 in I.items() if v1['perishable'] == 1 for i2, v2 in I.items() if v2['radioactive'] == 1]

model = Model("2D Bin Packing")

# ===============================Sets===============================
Bins = list(B.keys())
Bins_NC0 = B_type[0]
Bins_C1 = B_type[1]
Items = list(I.keys())

# ===============================Parameters===============================
L_b = {k: v['L'] for k, v in B.items()}
H_b = {k: v['H'] for k, v in B.items()}
C_b = {k: v['C'] for k, v in B.items()}
L_max = max(L_b.values())
H_max = max(H_b.values())
cutA_b = {k: v['cutA'] for k, v in B.items()}
cutB_b = {k: v['cutB'] for k, v in B.items()}
L_i = {k: v['L'] for k, v in I.items()}
H_i = {k: v['H'] for k, v in I.items()}

eps = 1
bigM = 1000
n = len(Items)

# ===============================Decision Variables===============================
z = model.addVars(Bins, vtype=GRB.BINARY, name="z")
p = model.addVars(Items, Bins, vtype=GRB.BINARY, name="p")
x = model.addVars(Items, vtype=GRB.CONTINUOUS, name="x")
y = model.addVars(Items, vtype=GRB.CONTINUOUS, name="y")
l = model.addVars(Items, Items, vtype=GRB.BINARY, name="l")
u = model.addVars(Items, Items, vtype=GRB.BINARY, name="u")
r = model.addVars(Items, vtype=GRB.BINARY, name="r")
g = model.addVars(Items, vtype=GRB.BINARY, name="g")
hij = model.addVars(Items, Items, vtype=GRB.BINARY, name="hij")
vij = model.addVars(Items, Items, vtype=GRB.CONTINUOUS, name="vij")
mij = model.addVars(Items, Items, vtype=GRB.BINARY, name="mij")
o = model.addVars(Items, Items, vtype=GRB.BINARY, name="o")
s = model.addVars(Items, Items, vtype=GRB.BINARY, name="s")
beta1 = model.addVars(Items, Items, vtype=GRB.BINARY, name="beta1")
beta2 = model.addVars(Items, Items, vtype=GRB.BINARY, name="beta2")
e1 = model.addVars(Items, Items, vtype=GRB.BINARY, name="e1")
e2 = model.addVars(Items, Items, vtype=GRB.BINARY, name="e2")
c = model.addVars(Items, vtype=GRB.BINARY, name="c")
f = model.addVars(Items, vtype=GRB.BINARY, name="f")

# ===============================Objective Function===============================
model.setObjective(quicksum(C_b[b] * z[b] for b in Bins), GRB.MINIMIZE)

# ===============================Constraints===============================
# Bin Usage Constraints
for i in Items:
    for b in Bins:
        model.addConstr(p[i, b] <= z[b])

for b in range(len(Bins_NC0) - 1):
    model.addConstr(z[Bins_NC0[b + 1]] <= z[Bins_NC0[b]])

for b in range(len(Bins_C1) - 1):
    model.addConstr(z[Bins_C1[b + 1]] <= z[Bins_C1[b]])

for i in Items:
    model.addConstr(quicksum(p[i, b] for b in Bins) == 1)

for pair in I_inc:
    for b in Bins:
        model.addConstr(p[pair[0], b] + p[pair[1], b] <= 1)

for i in Items:
    model.addConstr(x[i] + L_i[i] * (1 - r[i]) + H_i[i] * r[i] <= quicksum(L_b[b] * p[i, b] for b in Bins))
    model.addConstr(y[i] + L_i[i] * r[i] + H_i[i] * (1 - r[i]) <= quicksum(H_b[b] * p[i, b] for b in Bins))

# Geometric Constraints
for i in Items:
    for j in Items:
        if j != i:
            for b in Bins:
                model.addConstr(l[i, j] + l[j, i] + u[i, j] + u[j, i] >= p[i, b] + p[j, b] - 1)

for i in Items:
    for j in Items:
        if j != i:
            model.addConstr(x[j] >= x[i] + L_i[i] * (1 - r[i]) + H_i[i] * r[i] - L_max * (1 - l[i, j]))
            model.addConstr(x[j] + eps <= x[i] + L_i[i] * (1 - r[i]) + H_i[i] * r[i] + L_max * l[i, j])
            model.addConstr(y[j] >= y[i] + L_i[i] * r[i] + H_i[i] * (1 - r[i]) - H_max * (1 - u[i, j]))
            model.addConstr(y[j] + eps <= y[i] + L_i[i] * r[i] + H_i[i] * (1 - r[i]) + H_max * u[i, j])

for i in Items:
    for b in Bins_C1:
        model.addConstr(y[i] >= cutB_b[b] - ((cutB_b[b] / cutA_b[b]) * x[i]) - ((1 - p[i, b]) * bigM))

for i in Items:
    if I[i]['rotation'] == 0:
        model.addConstr(r[i] == 0)

for i in Items:
    model.addConstr(c[i] + quicksum(beta1[i, j] for j in Items if j != i) + quicksum(beta2[i, j] for j in Items if j != i) + (2 * g[i]) >= 2)
    model.addConstr(y[i] <= H_max * (1 - g[i]))

for i in Items:
    for j in Items:
        if j != i:
            model.addConstr((y[j] + H_i[j] * (1 - r[j]) + L_i[j] * r[j]) - y[i] <= vij[i, j])
            model.addConstr(y[i] - (y[j] + H_i[j] * (1 - r[j]) + L_i[j] * r[j]) <= vij[i, j])
            model.addConstr(vij[i, j] <= (y[j] + H_i[j] * (1 - r[j]) + L_i[j] * r[j]) - y[i] + (2 * H_max * (1 - mij[i, j])))
            model.addConstr(vij[i, j] <= y[i] - (y[j] + H_i[j] * (1 - r[j]) + L_i[j] * r[j]) + (2 * H_max * mij[i, j]))
            model.addConstr(hij[i, j] <= vij[i, j])
            model.addConstr(vij[i, j] <= hij[i, j] * H_max)
            model.addConstr(mij[i, j] >= 1 - hij[i, j])

for i in Items:
    for j in Items:
        if j != i:
            model.addConstr(o[i, j] >= l[i, j])
            model.addConstr(o[i, j] >= l[j, i])
            model.addConstr(o[i, j] <= l[i, j] + l[j, i])

for i in Items:
    for j in Items:
        if j != i:
            model.addConstr(1 - s[i, j] <= hij[i, j] + o[i, j])
            model.addConstr(hij[i, j] + o[i, j] <= 2 * (1 - s[i, j]))

for b in Bins:
    for i in Items:
        for j in Items:
            if j != i:
                model.addConstr(p[i, b] - p[j, b] <= 1 - s[i, j])
                model.addConstr(p[j, b] - p[i, b] <= 1 - s[i, j])

for i in Items:
    for j in Items:
        if j != i:
            model.addConstr(beta1[i, j] <= s[i, j])
            model.addConstr(beta2[i, j] <= s[i, j])

for i in Items:
    for j in Items:
        if j != i:
            model.addConstr(x[j] <= x[i] + L_max * e1[i, j])
            model.addConstr(x[i] + L_i[i] * (1 - r[i]) + H_i[i] * r[i] <= x[j] + L_i[j] * (1 - r[j]) + H_i[j] * r[j] + L_max * e2[i, j])
            model.addConstr(e1[i, j] <= 1 - beta1[i, j])
            model.addConstr(e2[i, j] <= 1 - beta2[i, j])

for i in Items:
    for b in Bins_C1:
        model.addConstr((1 - c[i]) * bigM >= y[i] + (cutB_b[b] / cutA_b[b]) * x[i] - cutB_b[b] - (1 - p[i, b]) * bigM)

for i in Items:
    for b in Bins_NC0:
        model.addConstr(c[i] <= 1 - p[i, b])

for j in Items:
    model.addConstr(f[j] == I[j]['fragility'])

for j in Items:
    model.addConstr(quicksum(s[i, j] for i in Items if i != j) <= n * (1 - f[j]))

# ===============================Solve the problem===============================
model.Params.LogFile = "2DBPP_model.log"
model.Params.timeLimit = 7200
model.optimize()

I_b = defaultdict(list)
for i in Items:
    for b in Bins:
        if p[i, b].x >= 0.99:
            I_b[b].append(i)

XY_pos = {i: {'x': int(x[i].x), 'y': int(y[i].x), 
              'L': int(I[i]['L'] * (1 - r[i].x) + I[i]['H'] * r[i].x), 
              'H': int(I[i]['L'] * r[i].x + I[i]['H'] * (1 - r[i].x))} 
          for i in Items}

# ===============================Print the results===============================
print('Overall cost of used bins:', model.objVal)
for i in Items:
    for b in Bins:
        print(f'Item {i} - Bin {b}: {p[i, b].x}')

for i in Items:
    for j in Items:
        if j != i and (beta1[i, j].x == 1 or beta2[i, j].x == 1):
            print(f'beta1[{i},{j}]: {beta1[i,j].x}')
            print(f'beta2[{i},{j}]: {beta2[i,j].x}')  

for i in Items:
    if c[i].x == 1:
        print(f'c[{i}]: {c[i].x}')

for i in Items:
    if g[i].x == 1:
        print(f'g[{i}]: {g[i].x}')

# ===============================Plotting solution===============================
import matplotlib.patches as patches
from matplotlib.patches import Rectangle

axis_font  = {'fontname':'Arial', 'size':'15'}
random.seed(42)
plt.close('all')

for b in B.keys():
    if I_b[b]:
        polygon = patches.Polygon(((cutA_b[b], 0), (B[b]['L'], 0), (B[b]['L'], B[b]['H']), (0, B[b]['H']), (0, cutB_b[b])), closed=True, fill=None, edgecolor='r')
        fig, ax = plt.subplots()
        ax.add_patch(polygon)
        
        for i in I_b[b]:
            ax.add_patch(Rectangle((XY_pos[i]['x'], XY_pos[i]['y']), XY_pos[i]['L'], XY_pos[i]['H'],
                                   edgecolor='black',
                                   facecolor=I[i]['color'],
                                   fill=True, lw=1))
            plt.text((XY_pos[i]['x'] + XY_pos[i]['x'] + XY_pos[i]['L']) / 2,
                     (XY_pos[i]['y'] + XY_pos[i]['y'] + XY_pos[i]['H']) / 2,
                     str(i), fontsize=15, color='w')
        ax.set_xlim(0, B[b]['L'])
        ax.set_ylim(0, B[b]['H'])
        ax.set_xticks(range(0, B[b]['L'] + 1, 10))
        ax.set_yticks(range(0, B[b]['H'] + 1, 10))
        ax.set_xlabel('Length', **axis_font)
        ax.set_ylabel('Height', **axis_font)
        ax.grid(True)
        fig.savefig(f'bin_{b}.png', format='png', dpi=400, bbox_inches='tight', transparent=True, pad_inches=0.02)
        plt.show()