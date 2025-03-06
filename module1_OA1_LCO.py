# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 19:28:35 2025

@author: lorenzcr

#Module1_part2_Task2


P(I/P)=P(P/I)*P(I)/(P(P)) with P(P)=P(P/I)*P(I)+P(P/H)*P(H)

Infection Prevalence = [0.001,0.5,50]
Specificity = [99, 99.9, 99.99, 99.999]
Sensitivity = 99

Infection=P(I)
Sensitivity=P(P/I)
Healthy=P(H)=1-P(I)
FP=P(P/H)


"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def prob_func (Infection,Specificity):
    
    Sensitivity = 0.99
    P=(Sensitivity*Infection)/(Sensitivity*Infection+(1-Specificity)*(1-Infection))
    return P

#Generate a range of values for Specificity and Infection
specificity_values = np.linspace(0.99, 0.99999, 100)
infection_values = np.linspace(0.001, 0.5, 100)

# Create a meshgrid from the range of values
Specificity, Infection = np.meshgrid(specificity_values, infection_values)

# Calculate the probability for each point in the meshgrid
P = prob_func(Infection, Specificity)

# Plot the results using a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Surface plot
surf = ax.plot_surface(Specificity, Infection, P, cmap='viridis')

# Labels and title
ax.set_xlabel('Specificity')
ax.set_ylabel('Infection Rate')
ax.set_zlabel('Probability')
ax.set_title('3D plot of Probability Function')

# Add a color bar which maps values to colors
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()


