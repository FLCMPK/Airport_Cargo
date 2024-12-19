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
# Fragility, Supporting Vertex has not been implemented in the code

############
### Sets ###
############

# Bin types
# input: type (0 or 1), L (length), H (Height), # (number of bins), C (cost), cutA (coordinate 0,A of the cut), cutB (coordinate B,0 of the cut)
T = {0:{'L':10,'H':7,'#':2,'C':10,'cutA':0,'cutB':0},
     1:{'L':9,'H':6,'#':2,'C':8,'cutA':2,'cutB':2}}

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
I = {1: {'L': 4, 'H': 2, 'rotation': 1, 'perishable': 1, 'radioactive': 0, 'fragility': 1},
     2: {'L': 2, 'H': 3, 'rotation': 1, 'perishable': 1, 'radioactive': 0, 'fragility': 1},
     3: {'L': 2, 'H': 5, 'rotation': 0, 'perishable': 1, 'radioactive': 0, 'fragility': 1},
     4: {'L': 5, 'H': 1, 'rotation': 1, 'perishable': 1, 'radioactive': 0, 'fragility': 1},
     5: {'L': 3, 'H': 3, 'rotation': 1, 'perishable': 1, 'radioactive': 0, 'fragility': 1},
     6: {'L': 3, 'H': 2, 'rotation': 1, 'perishable': 0, 'radioactive': 0, 'fragility': 0},
     7: {'L': 2, 'H': 3, 'rotation': 1, 'perishable': 1, 'radioactive': 0, 'fragility': 0},
     8: {'L': 5, 'H': 2, 'rotation': 1, 'perishable': 1, 'radioactive': 0, 'fragility': 0},
     9: {'L': 3, 'H': 1, 'rotation': 0, 'perishable': 0, 'radioactive': 0, 'fragility': 0},
     10: {'L': 2, 'H': 1, 'rotation': 1, 'perishable': 0, 'radioactive': 0, 'fragility': 0},
     11: {'L': 5, 'H': 1, 'rotation': 1, 'perishable': 1, 'radioactive': 0, 'fragility': 0},
     12: {'L': 4, 'H': 3, 'rotation': 0, 'perishable': 0, 'radioactive': 1, 'fragility': 0},
     13: {'L': 4, 'H': 2, 'rotation': 1, 'perishable': 0, 'radioactive': 1, 'fragility': 0},
     14: {'L': 2, 'H': 1, 'rotation': 1, 'perishable': 0, 'radioactive': 1, 'fragility': 0},
     15: {'L': 2, 'H': 3, 'rotation': 1, 'perishable': 0, 'radioactive': 1, 'fragility': 0},
     16: {'L': 2, 'H': 6, 'rotation': 1, 'perishable': 0, 'radioactive': 1, 'fragility': 0},
     }


# Incompatible items (pairs) list due to item type of perishable and radioactive cannot be placed in the same bin
I_inc = [(i1,i2) for i1,v1 in I.items() if v1['perishable'] == 1 for i2, v2 in I.items() if v2['radioactive'] == 1]

model = Model("2D Bin Packing")

# ===============================Define sets===============================
Bins = list(B.keys())

# Separate bins into type 0 and type 1
Bins_NC0 = [b for b in Bins if B[b]['type'] == 0]
Bins_C1 = [b for b in Bins if B[b]['type'] == 1]

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
M = 10000

# ===============================Define decision variables===============================
x = model.addVars(Items, vtype=GRB.CONTINUOUS, name="x") # x-coordinate of the bottom-left corner of item i
y = model.addVars(Items, vtype=GRB.CONTINUOUS, name="y") # y-coordinate of the bottom-left corner of item i
r = model.addVars(Items, vtype=GRB.BINARY, name="r") # 1 if item i is rotated
p = model.addVars(Items, Bins, vtype=GRB.BINARY, name="p") # 1 if item i is packed in bin b
l = model.addVars(Items, Items, vtype=GRB.BINARY, name="l") # 1 if items i is to the left of item j
u = model.addVars(Items, Items, vtype=GRB.BINARY, name="u") # 1 if items i is under/below item j
z = model.addVars(Bins, vtype=GRB.BINARY, name="z") # 1 if bin b is used
g = model.addVars(Items, vtype=GRB.BINARY, name="g") # 1 if item i is placed on the ground
a1 = model.addVars(Items, Items, vtype=GRB.BINARY, name="a1") # 1 if vertex 1 of item i is placed on top of item j
a2 = model.addVars(Items, Items, vtype=GRB.BINARY, name="a2") # 1 if vertex 2 of item i is placed on top of item j
ol = model.addVars(Items, Items, vtype=GRB.BINARY, name="ol")  # 1 if the vertical projections of i and j overlap
ol1 = model.addVars(Items, Items, vtype=GRB.BINARY, name="ol1")
ol2 = model.addVars(Items, Items, vtype=GRB.BINARY, name="ol2")
vt = model.addVars(Items, Items, vtype=GRB.BINARY, name="vt")  # 1 if bottom of i and top of j share y-coordinate
vt1 = model.addVars(Items, Items, vtype=GRB.BINARY, name="vt1")
vt2 = model.addVars(Items, Items, vtype=GRB.BINARY, name="vt2")
sb = model.addVars(Items, Items, vtype=GRB.BINARY, name="sb") # 1 if i and j are in the same bin, 0 otherwise
st = model.addVars(Items, Items, vtype=GRB.BINARY, name="st")  # 1 if ol, vt and sb are all 1, 0 otherwise
c = model.addVars(Items, vtype=GRB.BINARY, name="c") # 1 if vertex 1 of item i is placed on the bin cut
s = model.addVars(Items, Items, vtype=GRB.BINARY, name="s") # 1 if item i is stacked on top of item j


# ===============================Define objective function===============================
model.setObjective(quicksum(C_b[b] * z[b] for b in Bins), GRB.MINIMIZE)

# ===============================Define constraints===============================
# Each item must be packed in one bin, pair of items i and j in the same bin
for i in Items:
    for j in Items:
        if j != i:
            for b in Bins:
                model.addConstr(l[i, j] + l[j, i] + u[i, j] + u[j, i] >= p[i, b] + p[j, b] - 1)

# constraint to avoid placing incompatible items in the same bin due to different type of items (perishable or radioactive)
for pair in I_inc: # iterating over incompatible item pairs
    for b in Bins:
        model.addConstr(p[pair[0], b] + p[pair[1], b] <= 1) # if both items are in the same bin, the sum is 2, which is not possible

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
            model.addConstr(y[j] >= y[i] + L_i[i] * r[i] + H_i[i] * (1 - r[i]) - H_max * (1 - u[i, j]))
            model.addConstr(y[j] + eps <= y[i] + L_i[i] * r[i] + H_i[i] * (1 - r[i]) + H_max * u[i, j])

# x and y coordinate of items i is not on the left or below the bin cut
for i in Items:
    for b in Bins_C1: # iterating over bins set that have a cut
        model.addConstr(y[i] >= cutB_b[b] - ((cutB_b[b] / cutA_b[b]) * x[i]) - ((1 - p[i, b]) * M))

# right side and upper side of the item i must be inside the bin
for i in Items:
    model.addConstr(x[i] + L_i[i] * (1 - r[i]) + H_i[i] * r[i] <= quicksum(L_b[b] * p[i, b] for b in Bins))

for i in Items:
    model.addConstr(y[i] + L_i[i] * r[i] + H_i[i] * (1 - r[i]) <= quicksum(H_b[b] * p[i, b] for b in Bins))

# If item i is not allowed to be rotated, it must be placed in the same orientation
for i in Items:
    if I[i]['rotation'] == 0:
        model.addConstr(r[i] == 0)

# Each item must be placed in exactly one bin
for i in Items:
    model.addConstr(quicksum(p[i, b] for b in Bins) == 1)

# constraint to flag the bins that are used
for i in Items:
    for b in Bins:
        model.addConstr(p[i, b] <= z[b])

# constraint to force bin of the same type to be used sequentially
for t, v in B_type.items():
    for b in range(len(v) - 1):
        model.addConstr(z[v[b + 1]] <= z[v[b]]) # if bin b+1 is used of the same type, then bin b must also be used

# Support and Vertex Stability Constraints need to be implemented
# y coordinate less than the height of the bin with ground support consideration
for i in Items:
    model.addConstr(y[i] <= H_max * (1 - g[i]))

# item stability constraint, item i must be placed on the ground or on top of another item or on the bin cut (only for bin type 1)
for i in Items:
    model.addConstr(c[i] + quicksum(a1[i, j] for j in Items if j != i) + quicksum(a2[i, j] for j in Items if j != i) + (2 * g[i]) >= 2)

# Stacking constraint: item i can only be placed on top of item j if item j can support item i
for i in Items:
    for j in Items:
        if j != i:
            model.addConstr(y[i] >= y[j] + H_i[j] * (1 - r[j]) + L_i[j] * r[j] - H_max * (1 - s[i, j])) # i should be on top of j, either rotated or not and lower than the height of the bin
            model.addConstr(s[i, j] <= quicksum(p[i, b] * p[j, b] for b in Bins))  # Ensure both items are in the same bin

# Define auxiliary variables for the product of two binary variables
aux1 = model.addVars(Items, Items, Bins, vtype=GRB.BINARY, name="aux1")
aux2 = model.addVars(Items, Items, Bins, vtype=GRB.BINARY, name="aux2")

# Define constraints for the auxiliary variables
for i in Items:
    for j1 in Items:
        for j2 in Items:
            if j1 != i and j2 != i and j1 != j2:
                for b in Bins:
                    model.addConstr(aux1[i, j1, b] <= p[i, b])
                    model.addConstr(aux1[i, j1, b] <= p[j1, b])
                    model.addConstr(aux1[i, j1, b] >= p[i, b] + p[j1, b] - 1)
                    model.addConstr(aux2[i, j2, b] <= p[i, b])
                    model.addConstr(aux2[i, j2, b] <= p[j2, b])
                    model.addConstr(aux2[i, j2, b] >= p[i, b] + p[j2, b] - 1)

# Center of gravity constraint: the center of gravity of the top box must be within the base of the bottom box
for i in Items:
    for j in Items:
        if j != i:
            model.addConstr(
                x[i] + (I[i]['L'] * (1 - r[i]) + I[i]['H'] * r[i]) / 2 >= x[j] - (I[j]['L'] * (1 - r[j]) + I[j]['H'] * r[j]) / 2 - (1 - s[i, j]) * L_max
            )
            model.addConstr(
                x[i] + (I[i]['L'] * (1 - r[i]) + I[i]['H'] * r[i]) / 2 <= x[j] + (I[j]['L'] * (1 - r[j]) + I[j]['H'] * r[j]) / 2 + (1 - s[i, j]) * L_max
            )

# New constraint: if the top box is on top of two boxes, its center of gravity must be on top of at least one of these boxes
for i in Items:
    for j1 in Items:
        for j2 in Items:
            if j1 != i and j2 != i and j1 != j2:
                model.addConstr(
                    s[i, j1] + s[i, j2] <= 1 + quicksum(aux1[i, j1, b] * p[j2, b] for b in Bins)
                )
                model.addConstr(
                    s[i, j1] + s[i, j2] <= 1 + quicksum(aux2[i, j2, b] * p[j1, b] for b in Bins)
                )
                model.addConstr(
                    x[i] + (I[i]['L'] * (1 - r[i]) + I[i]['H'] * r[i]) / 2 >= x[j1] - (I[j1]['L'] * (1 - r[j1]) + I[j1]['H'] * r[j1]) / 2 - (1 - s[i, j1]) * L_max
                )
                model.addConstr(
                    x[i] + (I[i]['L'] * (1 - r[i]) + I[i]['H'] * r[i]) / 2 <= x[j1] + (I[j1]['L'] * (1 - r[j1]) + I[j1]['H'] * r[j1]) / 2 + (1 - s[i, j1]) * L_max
                )
                model.addConstr(
                    x[i] + (I[i]['L'] * (1 - r[i]) + I[i]['H'] * r[i]) / 2 >= x[j2] - (I[j2]['L'] * (1 - r[j2]) + I[j2]['H'] * r[j2]) / 2 - (1 - s[i, j2]) * L_max
                )
                model.addConstr(
                    x[i] + (I[i]['L'] * (1 - r[i]) + I[i]['H'] * r[i]) / 2 <= x[j2] + (I[j2]['L'] * (1 - r[j2]) + I[j2]['H'] * r[j2]) / 2 + (1 - s[i, j2]) * L_max
                )

# Ensure a3[i,j] = 1 if the vertical projections of i and j overlap, and 0 otherwise
M = 10000
eps = 0.001
for i in Items:
    for j in Items:
        if i != j:
            width_j = L_i[j]*(1 - r[j]) + H_i[j]*r[j]
            width_i = L_i[i]*(1 - r[i]) + H_i[i]*r[i]

            # Constraints for overlap (if a = 1)
            model.addConstr(x[i] <= x[j] + width_j - eps + M * (1 - ol[i,j]), "overlap_cond1")
            model.addConstr(x[j] <= x[i] + width_i - eps + M * (1 - ol[i,j]), "overlap_cond2")

            # Constraints for non-overlap (if a = 0)
            model.addConstr(x[i] + width_i <= x[j] + M * ol1[i,j], "non_overlap1")
            model.addConstr(x[j] + width_j <= x[i] + M * ol2[i,j], "non_overlap2")

            model.addConstr(ol1[i,j] >= ol[i,j], "b_if_a_1")
            model.addConstr(ol2[i,j] >= ol[i,j], "c_if_a_1")
            model.addConstr(ol1[i,j] + ol2[i,j] == 1 + ol[i,j], "exactly_one_b_or_c_if_a_0")

# Ensure a4[i,j] = 1 if the bottom of i and the top of j share the same y-coordinate, and 0 otherwise
for i in Items:
    for j in Items:
        if i != j:
            height_j = H_i[j] * (1 - r[j]) + L_i[j] * r[j]
            a = y[i]
            b = y[j] + height_j

            # If x > y, then b = 1, otherwise b = 0
            model.addConstr(a >= b + eps - M * (1 - vt1[i,j]))
            model.addConstr(a <= b + M * vt1[i,j])

            # If x < y, then c = 1, otherwise c = 0
            model.addConstr(b >= a + eps - M * (1 - vt2[i,j]))
            model.addConstr(b <= a + M * vt2[i,j])

            # Ensure either b, c, or d is 1.
            model.addConstr(vt1[i,j] + vt2[i,j] + vt[i,j] == 1)

# Constraints to link sb[i, j] and p[i, b]
for i in Items:
    for j in Items:
        if i != j:
            # Ensure z[i, j] is 1 if i and j share any bin
            model.addConstr(sb[i, j] <= quicksum(p[i, b] * p[j, b] for b in Bins))
            model.addConstrs(sb[i, j] >= p[i, b] + p[j, b] - 1 for b in Bins)

# Ensure that if ol[i,j], vt[i,j] ad sb[i,j] are all 1, st[i,j] is 1 and 0 otherwise.
for i in Items:
    for j in Items:
        if i != j:
            model.addConstr(st[i,j] >= ol[i,j] + vt[i,j] + sb[i,j] - 2)
            model.addConstr(st[i,j] <= ol[i,j])
            model.addConstr(st[i,j] <= vt[i,j])
            model.addConstr(st[i,j] <= sb[i,j])

# Fragility constraint
for i in Items:
    for j in Items:
        if i != j:
            if I[j]['fragility'] == 1:
                model.addConstr(st[i,j] == 0)

# ===============================Solve the problem===============================
model.setParam(GRB.Param.TimeLimit, 20)
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

#########################
### Plotting solution ###
#########################
import matplotlib.patches as patches
from matplotlib.patches import Rectangle

axis_font  = {'fontname':'Arial', 'size':'15'}

random.seed(42)

plt.close('all')
for b in B.keys():
    if len(I_b[b])>0:
        polygon = patches.Polygon(((cutA_b[b], 0), (B[b]['L'], 0), (B[b]['L'], B[b]['H']), (0, B[b]['H']), (0, cutB_b[b])), closed=True, fill=None, edgecolor='r')
        
        fig, ax = plt.subplots()
        
        ax.add_patch(polygon)
        
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