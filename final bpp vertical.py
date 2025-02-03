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

from gurobipy import Model, GRB, quicksum

# Constraints that have not yet been implemented in the code:
# Different Bin sub set, Bin Cut specification, fragility, perishable, radioactive constraint, Supporting Vertex has not been implemented in the code

############
### Sets ###
############

# Bin types
# input: type (0 or 1), L (length), H (Height), # (number of bins), C (cost), cutA (coordinate 0,A of the cut), cutB (coordinate B,0 of the cut)
T = {0:{'L':20,'H':14,'#':2,'C':10,'cutA':0,'cutB':0},
     1:{'L':16,'H':10,'#':2,'C':8,'cutA':2,'cutB':3}}

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

# Items - incomp should be changed to different type of items (perishable or radioactive)
I = {1:{'L':4,'H':2,'bin_types':[0,1],'rotation':'Y','incomp':[12,15]},
    2:{'L':2,'H':3,'bin_types':[0,1],'rotation':'Y','incomp':[12,16]}, 
    3:{'L':2,'H':5,'bin_types':[0],'rotation':'N','incomp':[]}, 
    4:{'L':5,'H':1,'bin_types':[0],'rotation':'Y','incomp':[14]},
    5:{'L':3,'H':3,'bin_types':[0,1],'rotation':'Y','incomp':[]},
    6:{'L':3,'H':2,'bin_types':[0,1],'rotation':'Y','incomp':[]},
    7:{'L':2,'H':3,'bin_types':[0,1],'rotation':'Y','incomp':[13]},
    8:{'L':5,'H':2,'bin_types':[0,1],'rotation':'Y','incomp':[14]},
    9:{'L':3,'H':1,'bin_types':[0],'rotation':'N','incomp':[]},
    10:{'L':2,'H':1,'bin_types':[0,1],'rotation':'Y','incomp':[]},
    11:{'L':5,'H':1,'bin_types':[0,1],'rotation':'Y','incomp':[15]},
    12:{'L':4,'H':3,'bin_types':[1],'rotation':'N','incomp':[2]},
    13:{'L':4,'H':2,'bin_types':[0,1],'rotation':'Y','incomp':[3]},
    14:{'L':2,'H':1,'bin_types':[0,1],'rotation':'Y','incomp':[4]},
    15:{'L':2,'H':3,'bin_types':[0,1],'rotation':'Y','incomp':[5]},
    16:{'L':2,'H':6,'bin_types':[0,1],'rotation':'Y','incomp':[2,3]}     
     }

# Items and their bin type categorization, resulting in a list of bin numbers compatible with the item
B_i = {i:[[b for b in B_type[t]] for t in I[i]['bin_types']] for i in I.keys()}
B_i = {i:[x for xs in v for x in xs] for i,v in B_i.items()} # Flattening list of lists

# Incompatible items (pairs) list - should be changed to account for different type of items (perishable or radioactive)
I_inc = [(i1,i2) for i1,v in I.items() for i2 in v['incomp']]

model = Model("2D Bin Packing")

# ===============================Define sets===============================
Bins = list(B.keys())

# Separate bins into type 0 and type 1
Bins_0 = [b for b in Bins if B[b]['type'] == 0]
Bins_1 = [b for b in Bins if B[b]['type'] == 1]

Items = list(I.keys())

# ===============================Define parameters===============================
L_b = {k: v['L'] for k, v in B.items()}
H_b = {k: v['H'] for k, v in B.items()}
C_b = {k: v['C'] for k, v in B.items()}

L_max = max([v['L'] for k, v in B.items()])
H_max = max([v['H'] for k, v in B.items()])

cutA_b = {k: v['cutA'] for k, v in B.items()}
cutB_b = {k: v['cutB'] for k, v in B.items()}

L_i = {k: v['L'] for k, v in I.items()}
H_i = {k: v['H'] for k, v in I.items()}

eps = 0.1

# ===============================Define decision variables===============================
x = model.addVars(Items, vtype=GRB.CONTINUOUS, name="x") # x-coordinate of the bottom-left corner of item i
y = model.addVars(Items, vtype=GRB.CONTINUOUS, name="y") # y-coordinate of the bottom-left corner of item i
r = model.addVars(Items, vtype=GRB.BINARY, name="r") # 1 if item i is rotated
p = model.addVars(Items, Bins, vtype=GRB.BINARY, name="p") # 1 if item i is packed in bin b
l = model.addVars(Items, Items, vtype=GRB.BINARY, name="l") # 1 if items i is to the left of item j
b = model.addVars(Items, Items, vtype=GRB.BINARY, name="b") # 1 if items i is below item j
z = model.addVars(Bins, vtype=GRB.BINARY, name="z") # 1 if bin b is used
g = model.addVars(Items, vtype=GRB.BINARY, name="g") # 1 if item i is placed on the ground
a1 = model.addVars(Items, Items, vtype=GRB.BINARY, name="a1") # 1 if vertex 1 of item i is placed on top of item j
a2 = model.addVars(Items, Items, vtype=GRB.BINARY, name="a2") # 1 if vertex 2 of item i is placed on top of item j
c = model.addVars(Items, vtype=GRB.BINARY, name="c") # 1 if vertex 1 of item i is placed on the bin cut

# ===============================Define objective function===============================
model.setObjective(quicksum(C_b[b] * z[b] for b in Bins), GRB.MINIMIZE)

# ===============================Define constraints===============================
# Each item must be packed in one bin, pair of items i and j in the same bin
for i in Items:
    for j in Items:
        if j != i:
            for box in set(B_i[i]) & set(B_i[j]): # iterating over common bins for the pair
                model.addConstr(l[i, j] + l[j, i] + b[i, j] + b[j, i] >= p[i, box] + p[j, box] - 1)

# constraint to avoid placing incompatible items in the same bin - should be changed to account for different type of items (perishable or radioactive)
for pair in I_inc: # iterating over incompatible item pairs
    for box in set(B_i[pair[0]]) & set(B_i[pair[1]]): # iterating over common bins for the pair
        model.addConstr(p[pair[0], box] + p[pair[1], box] <= 1) # if both items are in the same bin, the sum is 2, which is not possible

# x coordinate comparison of items i and j
for i in Items:
    for j in Items:
        if j != i:
            model.addConstr(x[j] >= x[i] + L_i[i] * (1 - r[i]) + H_i[i] * r[i] - L_max * (1 - l[i, j]))
            model.addConstr(x[j] + eps <= x[i] + L_i[i] * (1 - r[i]) + H_i[i] * r[i] + L_max * l[i, j])

# y coordinate comparison of items i and j
for i in Items:
    for j in Items:
        if j != i:
            model.addConstr(y[j] >= y[i] + L_i[i] * r[i] + H_i[i] * (1 - r[i]) - H_max * (1 - b[i, j]))
            model.addConstr(y[j] + eps <= y[i] + L_i[i] * r[i] + H_i[i] * (1 - r[i]) + H_max * b[i, j])

# y coordinate less than the height of the bin with ground support consideration
for i in Items:
    model.addConstr(y[i] <= H_max * (1 - g[i]))

# right side and upper side of the item i must be inside the bin
for i in Items:
    model.addConstr(x[i] + L_i[i] * (1 - r[i]) + H_i[i] * r[i] <= quicksum(L_b[b] * p[i, b] for b in B_i[i]))

for i in Items:
    model.addConstr(y[i] + L_i[i] * r[i] + H_i[i] * (1 - r[i]) <= quicksum(H_b[b] * p[i, b] for b in B_i[i]))

# If item i is not allowed to be rotated, it must be placed in the same orientation
for i in Items:
    if I[i]['rotation'] == 'N':
        model.addConstr(r[i] == 0)

# Each item must be placed in exactly one bin
for i in Items:
    model.addConstr(quicksum(p[i, b] for b in B_i[i]) == 1)

# constraint to flag the bins that are used
for i in Items:
    for b in B_i[i]:
        model.addConstr(p[i, b] <= z[b])

# constraint to force bin of the same type to be used sequentially
for t, v in B_type.items():
    for b in range(len(v) - 1):
        model.addConstr(z[v[b + 1]] <= z[v[b]]) # if bin b+1 is used of the same type, then bin b must also be used

# item stability constraint, item i must be placed on the ground or on top of another item or on the bin cut (only for bin type 1)
for i in Items:
    model.addConstr(c[i] + quicksum(a1[i, j] for j in Items if j != i) + quicksum(a2[i, j] for j in Items if j != i) + (2 * g[i]) >= 2)

# ===============================Solve the problem===============================
model.optimize()

I_b = {b: [] for b in B.keys()}

# ===============================Print the results===============================
print('Overall cost of used bins:', model.objVal)
for i in Items:
    for b in B_i[i]:
        print(f'Item {i} - Bin {b}: {p[i, b].x}')
        if p[i, b].x >= 0.99:
            I_b[b].append(i)

XY_pos = {}
for i in Items:
    XY_pos[i] = {'x': int(x[i].x), 'y': int(y[i].x),
                 'L': int(I[i]['L'] * (1 - (r[i].x)) + I[i]['H'] * r[i].x),
                 'H': int(I[i]['L'] * r[i].x + I[i]['H'] * (1 - (r[i].x)))}

#########################
### Plotting solution ###
#########################
from matplotlib.patches import Rectangle
axis_font  = {'fontname':'Arial', 'size':'15'}

random.seed(42)

plt.close('all')
for b in B.keys():
    if len(I_b[b])>0:
        fig, ax = plt.subplots()
        for i in I_b[b]:
            ax.add_patch(Rectangle((XY_pos[i]['x'],
                                    XY_pos[i]['y']),
                                   XY_pos[i]['L'],XY_pos[i]['H'],
                 edgecolor = 'green',
                 facecolor =  [random.randint(0,255)/255, 
                               random.randint(0,255)/255, 
                               random.randint(0,255)/255 ],
                 fill=True,
                 lw=1))
            plt.text((XY_pos[i]['x']+XY_pos[i]['x']+XY_pos[i]['L'])/2,
                     (XY_pos[i]['y']+XY_pos[i]['y']+XY_pos[i]['H'])/2,
                     str(i),fontsize=15,color='w')
        ax.set_xlim(0,B[b]['L'])
        ax.set_ylim(0,B[b]['H'])
        ax.set_xticks(range(0,B[b]['L']+1))
        ax.set_yticks(range(0,B[b]['H']+1))
        ax.set_xlabel('Length',**axis_font)
        ax.set_ylabel('Height',**axis_font)
        ax.grid(True)
        # plt.show()
        fig.savefig('bin_%i.png'%(b), format='png', dpi=400, bbox_inches='tight',
                 transparent=True,pad_inches=0.02)