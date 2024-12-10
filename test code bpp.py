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

############
### Sets ###
############

# Bin types
# input: type (1 or 2), L (length), W (width), C (cost), # (number of bins)
T = {1:{'L':10,'W':7,'C':10,'#':2},
     2:{'L':8,'W':5,'C':8,'#':2}}

# Bins
# Create dictionary of bins, numbered from 1 to n
bin_counter = 1
B    = {}
for t,v in T.items():
    for j in range(0,v['#']):
        B[bin_counter] = {'type':t,'L':v['L'],'W':v['W'],'C':v['C']}
        bin_counter += 1

# Bins by type categorization    
B_type = {t:[k for k,v in B.items() if v['type']==t] for t in T.keys()}

# Items
I = {1:{'L':4,'W':2,'bin_types':[1,2],'rotation':'Y','incomp':[12,15]},
    2:{'L':2,'W':3,'bin_types':[1,2],'rotation':'Y','incomp':[12,16]}, 
    3:{'L':2,'W':5,'bin_types':[1],'rotation':'N','incomp':[]}, 
    4:{'L':5,'W':1,'bin_types':[1],'rotation':'Y','incomp':[14]},
    5:{'L':3,'W':3,'bin_types':[1,2],'rotation':'Y','incomp':[]},
    6:{'L':3,'W':2,'bin_types':[1,2],'rotation':'Y','incomp':[]},
    7:{'L':2,'W':3,'bin_types':[1,2],'rotation':'Y','incomp':[13]},
    8:{'L':5,'W':2,'bin_types':[1,2],'rotation':'Y','incomp':[14]},
    9:{'L':3,'W':1,'bin_types':[1],'rotation':'N','incomp':[]},
    10:{'L':2,'W':1,'bin_types':[1,2],'rotation':'Y','incomp':[]},
    11:{'L':5,'W':1,'bin_types':[1,2],'rotation':'Y','incomp':[15]},
    12:{'L':4,'W':3,'bin_types':[2],'rotation':'N','incomp':[2]},
    13:{'L':4,'W':2,'bin_types':[1,2],'rotation':'Y','incomp':[3]},
    14:{'L':2,'W':1,'bin_types':[1,2],'rotation':'Y','incomp':[4]},
    15:{'L':2,'W':3,'bin_types':[1,2],'rotation':'Y','incomp':[5]},
    16:{'L':2,'W':6,'bin_types':[1,2],'rotation':'Y','incomp':[2,3]}     
     }

# Items and their bin type categorization, resulting in a list of bin numbers compatible with the item
B_i = {i:[[b for b in B_type[t]] for t in I[i]['bin_types']] for i in I.keys()}
B_i = {i:[x for xs in v for x in xs] for i,v in B_i.items()} # Flattening list of lists

# Incompatible items (pairs) list
I_inc = [(i1,i2) for i1,v in I.items() for i2 in v['incomp']]

model = Model("2D Bin Packing")

# Define sets
Bins = list(B.keys())
Items = list(I.keys())

# Define parameters
L_b = {k: v['L'] for k, v in B.items()}
W_b = {k: v['W'] for k, v in B.items()}
L_i = {k: v['L'] for k, v in I.items()}
W_i = {k: v['W'] for k, v in I.items()}
C_b = {k: v['C'] for k, v in B.items()}

L_max = max([v['L'] for k, v in B.items()])
W_max = max([v['W'] for k, v in B.items()])

eps = 0.1

# Define decision variables
x = model.addVars(Items, vtype=GRB.CONTINUOUS, name="x") # x-coordinate of the bottom-left corner of item i
y = model.addVars(Items, vtype=GRB.CONTINUOUS, name="y") # y-coordinate of the bottom-left corner of item i
r = model.addVars(Items, vtype=GRB.BINARY, name="r") # 1 if item i is rotated
p = model.addVars(Items, Bins, vtype=GRB.BINARY, name="p") # 1 if item i is packed in bin b
l = model.addVars(Items, Items, vtype=GRB.BINARY, name="l") # 1 if items i is to the left of item j
b = model.addVars(Items, Items, vtype=GRB.BINARY, name="b") # 1 if items i is below item j
z = model.addVars(Bins, vtype=GRB.BINARY, name="z") # 1 if bin b is used

# Define objective function
model.setObjective(quicksum(C_b[b] * z[b] for b in Bins), GRB.MINIMIZE)

# Define constraints
# Each item must be packed in one bin, pair of items i and j in the same bin
for i in Items:
    for j in Items:
        if j != i:
            for box in set(B_i[i]) & set(B_i[j]): # iterating over common bins for the pair
                model.addConstr(l[i, j] + l[j, i] + b[i, j] + b[j, i] >= p[i, box] + p[j, box] - 1)

# constraint to avoid placing incompatible items in the same bin
for pair in I_inc: # iterating over incompatible item pairs
    for box in set(B_i[pair[0]]) & set(B_i[pair[1]]): # iterating over common bins for the pair
        model.addConstr(p[pair[0], box] + p[pair[1], box] <= 1) # if both items are in the same bin, the sum is 2, which is not possible

# x coordinate comparison of items i and j
for i in Items:
    for j in Items:
        if j != i:
            model.addConstr(x[j] >= x[i] + L_i[i] * (1 - r[i]) + W_i[i] * r[i] - L_max * (1 - l[i, j]))
            model.addConstr(x[j] + eps <= x[i] + L_i[i] * (1 - r[i]) + W_i[i] * r[i] + L_max * l[i, j])

# y coordinate comparison of items i and j
for i in Items:
    for j in Items:
        if j != i:
            model.addConstr(y[j] >= y[i] + L_i[i] * r[i] + W_i[i] * (1 - r[i]) - W_max * (1 - b[i, j]))
            model.addConstr(y[j] + eps <= y[i] + L_i[i] * r[i] + W_i[i] * (1 - r[i]) + W_max * b[i, j])

# right side and upper side of the item i must be inside the bin
for i in Items:
    model.addConstr(x[i] + L_i[i] * (1 - r[i]) + W_i[i] * r[i] <= quicksum(L_b[b] * p[i, b] for b in B_i[i]))

for i in Items:
    model.addConstr(y[i] + L_i[i] * r[i] + W_i[i] * (1 - r[i]) <= quicksum(W_b[b] * p[i, b] for b in B_i[i]))

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

for t, v in B_type.items():
    for b in range(len(v) - 1):
        model.addConstr(z[v[b + 1]] <= z[v[b]])

# Solve the problem
model.optimize()

I_b = {b: [] for b in B.keys()}

# Print the results
print('Overall cost of used bins:', model.objVal)
for i in Items:
    for b in B_i[i]:
        print(f'Item {i} - Bin {b}: {p[i, b].x}')
        if p[i, b].x >= 0.99:
            I_b[b].append(i)

XY_pos = {}
for i in Items:
    XY_pos[i] = {'x': int(x[i].x), 'y': int(y[i].x),
                 'L': int(I[i]['L'] * (1 - (r[i].x)) + I[i]['W'] * r[i].x),
                 'W': int(I[i]['L'] * r[i].x + I[i]['W'] * (1 - (r[i].x)))}

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
                                   XY_pos[i]['L'],XY_pos[i]['W'],
                 edgecolor = 'green',
                 facecolor =  [random.randint(0,255)/255, 
                               random.randint(0,255)/255, 
                               random.randint(0,255)/255 ],
                 fill=True,
                 lw=1))
            plt.text((XY_pos[i]['x']+XY_pos[i]['x']+XY_pos[i]['L'])/2,
                     (XY_pos[i]['y']+XY_pos[i]['y']+XY_pos[i]['W'])/2,
                     str(i),fontsize=15,color='w')
        ax.set_xlim(0,B[b]['L'])
        ax.set_ylim(0,B[b]['W'])
        ax.set_xticks(range(0,B[b]['L']+1))
        ax.set_yticks(range(0,B[b]['W']+1))
        ax.set_xlabel('Length',**axis_font)
        ax.set_ylabel('Width',**axis_font)
        ax.grid(True)
        plt.show()
        fig.savefig('bin_%i.png'%(b), format='png', dpi=400, bbox_inches='tight',
                 transparent=True,pad_inches=0.02)