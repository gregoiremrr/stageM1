import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pandas as pd

# Parameters that we change in the model
modele = "von bertalanffy" # "logistique" or "gompertz" or "von bertalanffy"
param_n0 = 0.5
param_r0n0 = 0.01
param_b = 5
param_death_exp = 2

# Parameters for the precision of the solution of the ODE
zz = 1000
dt = 1/zz
T = 7

# All the parameters of the model
b = param_b

MaxDose = 2

death_exp = param_death_exp

# Initial conditions
n0 = param_n0
r0 = param_r0n0 * n0
s0 = n0 - r0

# Control functions
# Adaptative therapy : we activate the treatment only when we are above the treshold
# So it stabilises the number of cancerous cells at the treshold while it is possible
def g(n, r, s, treshold):
    if n >= treshold:
        return 1
    return 0

# We let the treatment active all the long
def mtd(): # MTD
    return 1

# No treatment at all
def no_therapy():
    return 0

# To solve numerically the ODE, we define the functions f where \dot{x} = f(t,x)
def gr(n, r, b, modele):
    if modele == "von bertalanffy":
        return b * (n**(-1/3) - 1) * r
    elif modele == "gompertz":
        return b * math.log(1/n) * r
    elif modele == "logistique":
        return b * r * (1-n)

def gs(n, c, s, b, modele):
    if modele == "von bertalanffy":
        return b * (n**(-1/3) - 1) * (1 - MaxDose*c) * s
    elif modele == "gompertz":
        return b * math.log(1/n) * s * (1 - MaxDose*c)
    elif modele == "logistique":
        return b * s * (1-n) * (1 - MaxDose*c)

# Integrated solver
def dX_adaptative(t, L, b, modele, seuil):
    #print(L, b, modele)
    rpoint = gr(L[0], L[1], b, modele)
    spoint = gs(L[0], g(L[0], L[1], L[2], seuil), L[2], b, modele)  
    npoint = rpoint + spoint
    return npoint, rpoint, spoint

def dX_MTD(t, L, b, modele):
    #print("MTD : n = ", L[0], ", r = ", L[1], ", s =", L[2])
    rpoint = gr(L[0], L[1], b, modele)
    spoint = gs(L[0], mtd(), L[2], b, modele)
    npoint = rpoint + spoint
    return npoint, rpoint, spoint

def dX_NT(t, L, b, modele):
    #print("NT : n = ", L[0], ", r = ", L[1], ", s =", L[2])
    rpoint = gr(L[0], L[1], b, modele)
    spoint = gs(L[0], 0, L[2], b, modele)
    npoint = rpoint + spoint
    return npoint, rpoint, spoint

# Probability of being alive at time t with : \dot{x} -> - deathrate * N^{death_exp} * x
# we compute for all t \ge 0 : P[being alive at time t] = exp[- deathrate * \int_0^t N(s)^{death_exp} ds]
def death_curve2(evN, death_rate, death_exp):
    res = []
    integrale = 0
    for i in range(zz*T):
        integrale += dt * evN[i]**death_exp
        prob = math.exp(- death_rate * integrale)
        res.append(prob)
    return res

# We have P[being alive at time t] = P[x \ge t]
# We compute the life expectancy E[x] = \int_0^{+\infty} P[x \ge t] dt
######## Issue with the T = 7 ########
def esp(L):
    integrale = 0
    for i in L:
        integrale += i*dt
    return integrale

# We use a dicotomy to find the deathrate such that x(1) = 0.5
# ie at year one, half of the patients are still alive
def find_good_param(edo_NT, death_exp):
    alpha0 = 0
    alpha1 = 500000000000
    for i in range(50): # because 2^40 >> 500000000000
        alpha2 = (alpha0+alpha1)/2
        x = death_curve2(edo_NT.y[0], alpha2, death_exp)[zz] - 0.5
        if x > 0:
            alpha0 = alpha2
        elif x < 0:
            alpha1 = alpha2
        else:
            break
    return alpha2

# X-axis
x = np.arange(0,T,dt)

# Solutions of the ODE for the three controls
edo_adaptative = solve_ivp(dX_adaptative, [0, T], [n0, r0, s0], method = "LSODA", t_eval = x, args = (b, modele, n0))
edo_MTD = solve_ivp(dX_MTD, [0, T], [n0, r0, s0], method = "LSODA", t_eval = x, args = (b, modele))
edo_NT = solve_ivp(dX_NT, [0, T], [n0, r0, s0], method = "LSODA", t_eval = x, args = (b, modele))

# We find the good deathrate on the "No treatment control"
good_param = find_good_param(edo_NT, death_exp)

# Death curves for all the controls
d_adapt = death_curve2(edo_adaptative.y[0], good_param, death_exp)
d_MTD = death_curve2(edo_MTD.y[0], good_param, death_exp)
d_NT = death_curve2(edo_NT.y[0], good_param, death_exp)
"""
# We plot two graphs
# The first one is the death curves
fig = plt.figure()
plt.subplot(2, 1, 1)
plt.vlines(x = 1, ymin = 0, ymax = 1, colors = 'purple')
plt.hlines(y = 0.5, xmin = 0, xmax = T, colors = 'purple')
plt.plot(x, d_adapt, "g", label = "Adaptative treatment")
plt.plot(x, d_MTD, "b", label = "Continuous treatment")
plt.plot(x, d_NT, "r", label = "No treatment")
plt.legend()
plt.title("alpha = {},\n E_MTD[X] = {}, E_adapt[X] = {}".format(good_param, esp(d_MTD), esp(d_adapt)))

# The second one is the number of cancerous cells
plt.subplot(2, 1, 2)
plt.plot(x, edo_adaptative.y[0], label = "Adaptative", color = "g")
plt.plot(x, edo_MTD.y[0], label = "MTD", color = "b")
plt.plot(x, edo_NT.y[0], label = "NT", color = "r")
plt.legend()
plt.show()
"""

"""
plt.plot(x, edo_MTD.y[0], label = "Cancerous cells", color = "b")
plt.plot(x, edo_MTD.y[1], label = "Resistant cells", color = "r")
plt.plot(x, edo_MTD.y[2], label = "Sensitive cells", color = "g")
plt.legend()
plt.title("MTD therapy in the {} model\n(a = {}, n_0 = {}, r_0/n_0 = {})".format(modele, b, param_n0, param_r0n0))
plt.show()
"""

"""
plt.plot(x, d_adapt, label = "Adaptative therapy", color = "b")
plt.plot(x, d_MTD, label = "MTD therapy", color = "r")
plt.plot(x, d_NT, label = "No therapy", color = "g")
plt.legend()
plt.title("Death curves in the {} model\n(a = {}, n_0 = {}, r_0/n_0 = {})\nE_adapt = {}, E_MTD = {}, E_NT = {}".format(modele, b, param_n0, param_r0n0, round(esp(d_adapt),3), round(esp(d_MTD),3), round(esp(d_NT),3)))
plt.show()
"""



# Sets of parameters that we want to test
liste_modele = ["logistique", "gompertz", "von bertalanffy"] # ["logistique"] or ["gompertz"] or ["von bertalanffy"] or ["logistique", "gompertz", "von bertalanffy"]
liste_n0 = [0.01, 0.05, 0.1, 0.2, 0.5, 0.7] #6
liste_r0n0 = [1e-4, 1e-3, 1e-2, 1e-1, 0.2] #5
liste_b = [1, 2, 5, 8, 10] #5
liste_beta = [1, 2, 4, 6] #4

# To show the progression during the computations
nb_simulations = len(liste_modele)*len(liste_n0)*len(liste_r0n0)*len(liste_b)*len(liste_beta)
nb = 1

# Creation of a Pandas data frame
df = pd.DataFrame(columns=["model", "n0", "r0/n0", "b", "beta", "alpha", "esp_MTD", "esp_adapt", "diff", "ratio", "1", "2", "3"])

# We go through all the lists to compute the expectancy for each combination of parameters
for v_modele in liste_modele:
    for v_n0 in liste_n0:
        for v_r0n0 in liste_r0n0:
            v_r0 = v_n0 * v_r0n0
            v_s0 = v_n0 - v_r0
            for v_b in liste_b:
                for v_beta in liste_beta:
                    
                    print(v_modele, v_n0, v_r0n0, v_b, v_beta)
                    
                    v_edo_adaptative = solve_ivp(dX_adaptative, [0, T], [v_n0, v_r0, v_s0], method = "LSODA", t_eval = x, args = (v_b, v_modele, v_n0))
                    v_edo_MTD = solve_ivp(dX_MTD, [0, T], [v_n0, v_r0, v_s0], method = "LSODA", t_eval = x, args = (v_b, v_modele))
                    v_edo_NT = solve_ivp(dX_NT, [0, T], [v_n0, v_r0, v_s0], method = "LSODA", t_eval = x, args = (v_b, v_modele))
                    
                    good_alpha = find_good_param(v_edo_NT, v_beta)
                    
                    v_d_adapt = death_curve2(v_edo_adaptative.y[0], good_alpha, v_beta)
                    v_d_MTD = death_curve2(v_edo_MTD.y[0], good_alpha, v_beta)
                    
                    value_at_final_tps_adapt = v_d_adapt][T*zz-1]
                    value_at_final_tps_MTD = v_d_MTD[T*zz-1]

                    esp_MTD = esp(v_d_MTD)
                    esp_adapt = esp(v_d_adapt)
                    diff = esp_adapt-esp_MTD
                    ratio = esp_adapt/esp_MTD
                    
                    print(good_alpha, esp_MTD, esp_adapt, esp_adapt-esp_MTD, esp_adapt/esp_MTD)
                    
                    df.loc[len(df)] = pd.Series([v_modele, v_n0, v_r0n0, v_b, v_beta, good_alpha, esp_MTD, esp_adapt, diff, ratio, value_at_final_tps_adapt, value_at_final_tps_MTD, value_at_final_tps_NT], index=df.columns)
                    
                    print("Progression : ", round(1000*nb/nb_simulations)/10, "%")
                    nb += 1

# This line creates a .csv file in the current directory
df.to_csv('sort_param.csv', index=True)

# Order of the axis : L1 big X-axis / L2 big Y-axis / L3 small X-axis / L4 small Y-axis
L3 = liste_n0 #6
etiqu3 = "n0"
L4 = liste_r0n0 #5 
L4bis = [math.log(_) for _ in L4] # log for the logarithmic scale
etiqu4 = "r0/n0"
L1 = liste_b #5
etiqu1 = "b"
L2 = liste_beta #4
etiqu2 = "beta"

# What we compare : "ratio" or "diff"
comparison = "diff"

# To have colors that match between all the plots
zzz = list(df[comparison])
vmin = min(zzz)
vmax = max(zzz)

# Parameters for the plot (should not be changed!)
l1 = 5
l2 = 2
n1 = len(L1)
n2 = len(L2)
n3 = len(L3)
n4 = len(L4)

for model in liste_modele:
    df2 = df.loc[df["model"] == model]
    # We define the plot with subplots
    fig, axs = plt.subplots(figsize=(15, 8), sharex=True, sharey=True, layout="constrained")
    
    plt.title("{} model with the {} between E_adapt and E_MTD".format(model, comparison))
    
    # Values on the X-axis
    axs.set_xlabel(etiqu1)
    axs.xaxis.set_ticks([_/(2*n1) for _ in range(2*n1) if _ % 2 == 1]) # Locations of the values on the axis
    axs.xaxis.set_ticklabels(L1, rotation = 0, color = 'red', fontsize = 8, style = 'italic', verticalalignment = 'center')

    # Values on the Y-axis
    axs.set_ylabel(etiqu2)
    axs.yaxis.set_ticks([_/(2*n2) for _ in range(2*n2) if _ % 2 == 1]) # Locations of the values on the axis
    axs.yaxis.set_ticklabels(L2, rotation = 0, color = 'red', fontsize = 8, style = 'italic', verticalalignment = 'center')

    # We divide the main graph into subpart
    gs = fig.add_gridspec(l2*n2+2, l1*n1+5)

    # Loop for each subplot
    for i,k in enumerate(L1):
        for j,f in enumerate(L2):
            
            jj = n2 - j - 1 # to go from bottom to top
            
            # We take the values that should be in the subgraph
            data = df2.loc[(df2[etiqu1] == k) & (df2[etiqu2] == f)]
            data3 = list(data[etiqu3])
            data4 = [math.log(_) for _ in list(data[etiqu4])] # For the logarithmic scale
            data_diff = list(data[comparison])
            
            # We create the subplot and add the scatter plot
            fig.add_subplot(gs[(l2*jj+1):(l2*(jj+1)+1), (l1*i+1):(l1*(i+1)+1)])
            im = plt.scatter(data3, data4, c=data_diff, linewidths = 4, vmin=vmin, vmax=vmax)
            
            if j == 0: # To have the names of the subplot X-axis only at the bottom
                plt.xlabel(etiqu3)
            plt.gca().xaxis.set_ticks(L3)
            plt.gca().xaxis.set_ticklabels(L3, rotation = -50, color = 'red', fontsize = 6, style = 'italic', verticalalignment = 'center')
            
            if i == 0: # To have the names of the subplot Y-axis only on the left
                plt.ylabel(etiqu4)
            plt.gca().yaxis.set_ticks(L4bis)
            plt.gca().yaxis.set_ticklabels(L4, rotation = 0, color = 'red', fontsize = 6, style = 'italic', verticalalignment = 'center')

    # We add the color bar
    fig.colorbar(im, ax=axs, shrink=0.6)

    # We plot the whole graph
    plt.show()
""""""