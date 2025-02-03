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

from gurobipy import Model, GRB, quicksum

# ===============================Read the data===============================
# Define the path to the pickle files
path_to_B = "B.pickle"
path_to_I = "I.pickle"

# Read the pickle files
with open(path_to_B, 'rb') as file_B:
    Bpickle = pickle.load(file_B)

with open(path_to_I, 'rb') as file_I:
    Ipickle = pickle.load(file_I)

# Bin types
# input: type (0 or 1), L (length), H (Height), # (number of bins), C (cost), cutA (coordinate 0,A of the cut), cutB (coordinate B,0 of the cut)
# T = {0:{'L':100,'H':70,'#':2,'C':100,'cutA':0,'cutB':0},
#      1:{'L':90,'H':60,'#':2,'C':80,'cutA':20,'cutB':20}}

T = {}
for key, value in Bpickle.items():
    T[value[0]] = {
        'L': value[1][0],
        'H': value[1][1],
        '#': value[1][2],
        'C': value[1][3],
        'cutA': 0 if value[1][4] == -1 else value[1][4],
        'cutB': 0 if value[1][5] == -1 else value[1][5]
    }

# Bins
# Create dictionary of bins, numbered from 1 to n
bin_counter = 1
B    = {}
for t,v in T.items():
    for j in range(0,v['#']):
        B[bin_counter] = {'type':t,'L':v['L'],'H':v['H'],'C':v['C'],'cutA':v['cutA'],'cutB':v['cutB']}
        bin_counter += 1

# Bins by type categorization    
B_type = {t:[k for k,v in B.items() if v['type']==t] for t in T.keys()}

# Items - needs to add fragility type
# I = {1:{'L':40,'H':20,'rotation':1,'fragility':0,'perishable':1,'radioactive':0},
#     2:{'L':20,'H':30,'rotation':1,'fragility':0,'perishable':1,'radioactive':0}, 
#     3:{'L':20,'H':50,'rotation':0,'fragility':0,'perishable':1,'radioactive':0}, 
#     4:{'L':50,'H':10,'rotation':1,'fragility':0,'perishable':1,'radioactive':0}, 
#     5:{'L':30,'H':30,'rotation':1,'fragility':0,'perishable':1,'radioactive':0},
#     6:{'L':30,'H':20,'rotation':1,'fragility':0,'perishable':0,'radioactive':0},
#     7:{'L':20,'H':30,'rotation':1,'fragility':0,'perishable':1,'radioactive':0},
#     8:{'L':50,'H':20,'rotation':1,'fragility':0,'perishable':1,'radioactive':0},
#     9:{'L':30,'H':10,'rotation':0,'fragility':1,'perishable':0,'radioactive':0},
#     10:{'L':20,'H':10,'rotation':1,'fragility':0,'perishable':0,'radioactive':0},
#     11:{'L':50,'H':10,'rotation':1,'fragility':0,'perishable':1,'radioactive':0},
#     12:{'L':40,'H':30,'rotation':0,'fragility':0,'perishable':0,'radioactive':1},
#     13:{'L':40,'H':20,'rotation':1,'fragility':0,'perishable':0,'radioactive':1},
#     14:{'L':20,'H':10,'rotation':1,'fragility':0,'perishable':0,'radioactive':1},
#     15:{'L':20,'H':30,'rotation':1,'fragility':1,'perishable':0,'radioactive':1},
#     16:{'L':20,'H':60,'rotation':1,'fragility':0,'perishable':0,'radioactive':1}  
#      }

I = {}
for k, v in Ipickle.items():
    I[k+1] = {
        'L': v[0],
        'H': v[1],
        'rotation': v[2],
        'fragility': v[3],
        'perishable': v[4],
        'radioactive': v[5],
        'color': 'green' if v[4] == 1 else 'red' if v[5] == 1 else 'blue' if v[3] == 1 else 'yellow' if v[2] == 0 else 'gray'
    }

# Incompatible items (pairs) list due to item type of perishable and radioactive cannot be placed in the same bin
I_inc = [(i1,i2) for i1,v1 in I.items() if v1['perishable'] == 1 for i2, v2 in I.items() if v2['radioactive'] == 1]

model = Model("2D Bin Packing")

# ===============================Sets===============================
Bins = list(B.keys())

# Separate bins into type 0 and type 1
Bins_NC0 = [b for b in Bins if B[b]['type'] == 0]
Bins_C1 = [b for b in Bins if B[b]['type'] == 1]

Items = list(I.keys())

# ===============================Parameters===============================
L_b = {k: v['L'] for k, v in B.items()}
H_b = {k: v['H'] for k, v in B.items()}
C_b = {k: v['C'] for k, v in B.items()}

L_max = max([v['L'] for k, v in B.items()])
H_max = max([v['H'] for k, v in B.items()])

cutA_b = {k: v['cutA'] for k, v in B.items()}
cutB_b = {k: v['cutB'] for k, v in B.items()}

L_i = {k: v['L'] for k, v in I.items()}
H_i = {k: v['H'] for k, v in I.items()}

eps = 1
bigM = 170

n = len(Items)

# ===============================Decision Variables===============================
z = model.addVars(Bins, vtype=GRB.BINARY, name="z") # 1 if bin b is used
p = model.addVars(Items, Bins, vtype=GRB.BINARY, name="p") # 1 if item i is packed in bin b
x = model.addVars(Items, vtype=GRB.CONTINUOUS, name="x") # x-coordinate of the bottom-left corner of item i
y = model.addVars(Items, vtype=GRB.CONTINUOUS, name="y") # y-coordinate of the bottom-left corner of item i
l = model.addVars(Items, Items, vtype=GRB.BINARY, name="l") # 1 if items i is to the left of item j
u = model.addVars(Items, Items, vtype=GRB.BINARY, name="u") # 1 if items i is under/below item j
r = model.addVars(Items, vtype=GRB.BINARY, name="r") # 1 if item i is rotated

g = model.addVars(Items, vtype=GRB.BINARY, name="g") # 1 if item i is placed on the ground

hij = model.addVars(Items, Items, vtype=GRB.BINARY, name="hij") # 1 if item j has the correct height to support item i
vij = model.addVars(Items, Items, vtype=GRB.CONTINUOUS, name="vij") # absolute vertical distance between item i and j, auxiliary variable for h
mij = model.addVars(Items, Items, vtype=GRB.BINARY, name="mij") # 1 if item j has larger or equal height to support item i, auxiliary variable for h

o = model.addVars(Items, Items, vtype=GRB.BINARY, name="o") # 0 if item i is on top of item j and they are on the same projection in the x-axis

s = model.addVars(Items, Items, vtype=GRB.BINARY, name="s") # 1 if item i is placed directly on top of item j inside the same bin

beta1 = model.addVars(Items, Items, vtype=GRB.BINARY, name="beta1") # 1 if vertex 1 (x[i]) of item i is placed on top of item j
beta2 = model.addVars(Items, Items, vtype=GRB.BINARY, name="beta2") # 1 if vertex 2 (x[i] + L_i[i] * (1 - r[i]) + H_i[i] * r[i]) of item i is placed on top of item j
e1 = model.addVars(Items, Items, vtype=GRB.BINARY, name="e1") # 0 if x[j] <= x[i], auxiliary variable for beta1
e2 = model.addVars(Items, Items, vtype=GRB.BINARY, name="e2") # 0 if x[i] + L_i[i] * (1 - r[i]) + H_i[i] * r[i] <= x[j] + L_i[j] * (1 - r[j]) + H_i[j] * r[j], auxiliary variable for beta2

c = model.addVars(Items, vtype=GRB.BINARY, name="c") # 1 if vertex 1 of item i is placed on the bin cut

f = model.addVars(Items, vtype=GRB.BINARY, name="f") # 1 if item i is fragile

# ===============================Objective Function===============================
model.setObjective(quicksum(C_b[b] * z[b] for b in Bins), GRB.MINIMIZE)

# ===============================Constraints===============================
# ====Bin Usage Constraints====
# constraint to flag the bins that are used - Constraint 2
for i in Items:
    for b in Bins:
        model.addConstr(p[i, b] <= z[b])

# constraint to force bin of the same type to be used sequentially - Constraint 3
for b in range(len(Bins_NC0) - 1): # iterating b over bins type 0 (without cut)
    model.addConstr(z[Bins_NC0[b + 1]] <= z[Bins_NC0[b]]) # if bin b+1 (ordered numbering of bins) is used of the same type, then bin b must also be used before

for b in range(len(Bins_C1) - 1): # iterating b over bins type 1 (with cut)
    model.addConstr(z[Bins_C1[b + 1]] <= z[Bins_C1[b]])

# Each item must be placed in exactly one bin - Constraint 4
for i in Items:
    model.addConstr(quicksum(p[i, b] for b in Bins) == 1)

# Avoid placing incompatible items in the same bin due to different type of items (perishable or radioactive) - Constraint 5
for pair in I_inc: # iterating over incompatible item pairs
    for b in Bins:
        model.addConstr(p[pair[0], b] + p[pair[1], b] <= 1) # if both items are in the same bin, the sum is 2, which is not possible

# right side and upper side of the item i must be contained inside the bin - Constraint 6 - 7
for i in Items:
    model.addConstr(x[i] + L_i[i] * (1 - r[i]) + H_i[i] * r[i] <= quicksum(L_b[b] * p[i, b] for b in Bins))

for i in Items:
    model.addConstr(y[i] + L_i[i] * r[i] + H_i[i] * (1 - r[i]) <= quicksum(H_b[b] * p[i, b] for b in Bins))

# ====Geometric Constraints====
# if pair of items i and j in the same bin, preventing items overlap - Constraint 8
for i in Items:
    for j in Items:
        if j != i:
            for b in Bins:
                model.addConstr(l[i, j] + l[j, i] + u[i, j] + u[j, i] >= p[i, b] + p[j, b] - 1)

# x coordinate comparison of items i and j, mutual positioning of items on x-axis - Constraint 9 - 10
for i in Items:
    for j in Items:
        if j != i:
            model.addConstr(x[j] >= x[i] + L_i[i] * (1 - r[i]) + H_i[i] * r[i] - L_max * (1 - l[i, j])) # if l[i,j] = 1, then x[j] >= x[i] + L_i[i] * (1 - r[i]) + H_i[i] * r[i]
            model.addConstr(x[j] + eps <= x[i] + L_i[i] * (1 - r[i]) + H_i[i] * r[i] + L_max * l[i, j]) # if l[i,j] = 0, then x[j] < x[i] + L_i[i] * (1 - r[i]) + H_i[i] * r[i]

# y coordinate comparison of items i and j, mutual positioning of items on y-axis - Constraint 11 - 12
for i in Items:
    for j in Items:
        if j != i:
            model.addConstr(y[j] >= y[i] + L_i[i] * r[i] + H_i[i] * (1 - r[i]) - H_max * (1 - u[i, j]))
            model.addConstr(y[j] + eps <= y[i] + L_i[i] * r[i] + H_i[i] * (1 - r[i]) + H_max * u[i, j])

# x and y coordinate of items i is not on the left or below the bin cut - Constraint 13
for i in Items:
    for b in Bins_C1: # iterating over bins set that have a cut
        model.addConstr(y[i] >= cutB_b[b] - ((cutB_b[b] / cutA_b[b]) * x[i]) - ((1 - p[i, b]) * bigM))

# If item i is not allowed to be rotated, it must be placed in the same orientation - Constraint 14
for i in Items:
    if I[i]['rotation'] == 0:
        model.addConstr(r[i] == 0)

# ====Specific Constraints====
# Overall item stability constraint, item i must be placed on the ground or on top of another item or on the bin cut (only for bin type 1), this links all the support variables - Constraint 15
for i in Items:
    model.addConstr(c[i] + quicksum(beta1[i, j] for j in Items if j != i) + quicksum(beta2[i, j] for j in Items if j != i) + (2 * g[i]) >= 2)

# y coordinate less than the height of the bin with ground support consideration, also ensuring item i with y[i] = 0 is placed on the ground (g[i] = 0)- Constraint 16
for i in Items:
    model.addConstr(y[i] <= H_max * (1 - g[i]))

# Vertical Compatibility: item i can only be placed on top of item j if item j has the right height to support item i - Constraint 17 - 23
for i in Items:
    for j in Items:
        if j != i:
            model.addConstr((y[j] + H_i[j] * (1 - r[j]) + L_i[j] * r[j]) - y[i] <= vij[i, j])
            model.addConstr(y[i] - (y[j] + H_i[j] * (1 - r[j]) + L_i[j] * r[j]) <= vij[i, j])
            model.addConstr(vij[i, j] <= (y[j] + H_i[j] * (1 - r[j]) + L_i[j] * r[j]) - y[i] + (2 * H_max * (1 - mij[i, j])))
            model.addConstr(vij[i, j] <= y[i] - (y[j] + H_i[j] * (1 - r[j]) + L_i[j] * r[j]) + (2 * H_max * mij[i, j]))
            model.addConstr(hij[i, j] <= vij[i, j])
            model.addConstr(vij[i, j] <= hij[i, j] * H_max)
            model.addConstr(mij[i, j] >= 1 - hij[i, j])     # if h[i,j] = 0, then m[i,j] = 1, maybe redundant, without this constraint, the model still works but m is inconsistent on h[i,j] = 0

# Horizontal Compatibility: item i can only be placed on top of item j if they are on the same projection in the x-axis - Constraint 24 - 26
for i in Items:
    for j in Items:
        if j != i:
            # Ensure o[i, j] is 1 if one of l[i, j] or l[j, i] is 1
            model.addConstr(o[i, j] >= l[i, j])
            model.addConstr(o[i, j] >= l[j, i])
            model.addConstr(o[i, j] <= l[i, j] + l[j, i])

# constraint to ensure s[i,j] related to o[i,j], hij[i,j] and p[i,b] - Constraint 27 - 30
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

# vertex support constraints
# vertex support is less than or equal to box support, linking beta1 and beta2 to s[i,j] - Constraint 31 - 32
for i in Items:
    for j in Items:
        if j != i:
            model.addConstr(beta1[i, j] <= s[i, j])
            model.addConstr(beta2[i, j] <= s[i, j])

# Hotizontal compatibility 2 - Constraint 33 - 36
for i in Items:
    for j in Items:
        if j != i:
            # Ensure e1 and e2 are correctly linked to the positions of items
            model.addConstr(x[j] <= x[i] + L_max * e1[i, j])
            
            model.addConstr(x[i] + L_i[i] * (1 - r[i]) + H_i[i] * r[i] <= x[j] + L_i[j] * (1 - r[j]) + H_i[j] * r[j] + L_max * e2[i, j])
            
            # Ensure beta1 and beta2 are correctly linked to e1 and e2
            model.addConstr(e1[i, j] <= 1 - beta1[i, j])
            model.addConstr(e2[i, j] <= 1 - beta2[i, j])

# cut support constraint - Constraint 37
for i in Items:
    for b in Bins_C1:
        model.addConstr((1 - c[i]) * bigM >= y[i] + (cutB_b[b] / cutA_b[b]) * x[i] - cutB_b[b] - (1 - p[i, b]) * bigM)

# c variables on non-cut bins are 0 - Constraint 38
for i in Items:
    for b in Bins_NC0:
        model.addConstr(c[i] <= 1 - p[i, b])

# fragility constraint - Constraint 39
for j in Items:
    if I[j]['fragility'] == 1:
        model.addConstr(f[j] == 1)
    else:
        model.addConstr(f[j] == 0)

for j in Items:
    model.addConstr(quicksum(s[i, j] for i in Items if i != j) <= n * (1 - f[j]))

# ===============================Solve the problem===============================
model.Params.LogFile = "2DBPP_model.log"
model.Params.timeLimit = 72
# model.Params.presolve = 2  # Aggressive presolve
# model.Params.heuristics = 0.05

model.optimize()
I_b = {b: [] for b in B.keys()}

# ===============================Print the results===============================
print('Overall cost of used bins:', model.objVal)
for i in Items:
    for b in Bins:
        print(f'Item {i} - Bin {b}: {p[i, b].x}')
        if p[i, b].x >= 0.99:
            I_b[b].append(i)

XY_pos = {}
for i in Items:
    XY_pos[i] = {'x': int(x[i].x), 'y': int(y[i].x),
                 'L': int(I[i]['L'] * (1 - (r[i].x)) + I[i]['H'] * r[i].x),
                 'H': int(I[i]['L'] * r[i].x + I[i]['H'] * (1 - (r[i].x)))}

# print variables
# s[i,j] = 1 if item i is placed on top of item j
# for i in Items:
#     for j in Items:
#         if j != i and s[i, j].x == 1:
#             print(f's[{i},{j}]: {s[i,j].x}')

# h[i,j] = 1 if item j has the correct height to support item i
# for i in Items:
#     for j in Items:
#         if j != i:
#             print(f'h[{i},{j}]: {hij[i,j].x}')
#             print(f'v[{i},{j}]: {vij[i,j].x}')
#             print(f'm[{i},{j}]: {mij[i,j].x}')

# for i in Items:
#     for j in Items:
#         if j != i:
#             print(f'l[{i},{j}]: {l[i,j].x}')
#             print(f'l[{j},{i}]: {l[j,i].x}')

# for i in Items:
#     for j in Items:
#         if j != i and o[i, j].x == 0:
#             print(f'o[{i},{j}]: {o[i,j].x}')

# for i in Items:
#     for j in Items:
#         if j != i:
#             print(f'e1[{i},{j}]: {e1[i,j].x}')
#             print(f'e2[{i},{j}]: {e2[i,j].x}')

for i in Items:
    for j in Items:
        if j != i and beta1[i, j].x == 1 or beta2[i, j].x == 1:
            print(f'beta1[{i},{j}]: {beta1[i,j].x}')
            print(f'beta2[{i},{j}]: {beta2[i,j].x}')  

for i in Items:
    if c[i].x == 1:
        print(f'c[{i}]: {c[i].x}')

for i in Items:
    if g[i].x == 1:
        print(f'g[{i}]: {g[i].x}')

# exporting the results to a xls file
# results = {'Item': [], 'x': [], 'y': []}
# for i in Items:
#     results['Item'].append(i)
#     results['x'].append(x[i].x)
#     results['y'].append(y[i].x)

# df = pd.DataFrame(results)
# output_path = '/c:/Users/malyd/OneDrive - Delft University of Technology/1st year/AE4446 Airport and Cargo Operations (202425 Q2)/assignment/results.xlsx'
# df.to_excel(output_path, index=False)


#########################
### Plotting solution ###
#########################
import matplotlib.patches as patches
from matplotlib.patches import Rectangle

axis_font  = {'fontname':'Arial', 'size':'15'}

random.seed(42)

plt.close('all')
for b in B.keys():
    if len(I_b[b]) > 0:
        polygon = patches.Polygon(((cutA_b[b], 0), (B[b]['L'], 0), (B[b]['L'], B[b]['H']), (0, B[b]['H']), (0, cutB_b[b])), closed=True, fill=None, edgecolor='r')
        
        fig, ax = plt.subplots()
        
        ax.add_patch(polygon)
        
        for i in I_b[b]:
            ax.add_patch(Rectangle((XY_pos[i]['x'],
                                    XY_pos[i]['y']),
                                   XY_pos[i]['L'], XY_pos[i]['H'],
                 edgecolor='black',
                 facecolor=I[i]['color'],
                 fill=True,
                 lw=1))
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
        # plt.show()
        fig.savefig('bin_%i.png' % (b), format='png', dpi=400, bbox_inches='tight',
                    transparent=True, pad_inches=0.02)
