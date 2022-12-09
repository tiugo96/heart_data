#####################################################################
# Technische Universitaet Muenchen
# Chair of Structural Analysis
# IFEM - Introduction to Finite Element Methods
# Dr.-Ing. Roland Wuechner
#####################################################################
# Exercise 3 - Main File
#####################################################################
# importing required modules
import exercise3_Kernel as exercise_Kernel # importing the Kernel
import numpy as np
from matplotlib import pyplot as plt

# Assemble Master Stiffness Matrix
K = exercise_Kernel.AssembleMasterStiffOfExampleTruss()
print("Master Stiffness Matrix (K):\n", K, "\n")

# Define Node Force Vector
f = np.array([0, 0, 0, 0, 2, 1])
print("Global Nodal Force Vector (f):\n", f, "\n")

print("... Applying Dirchlet Boundary Conditions ...\n")

# Modify master stiffness matrix and node force vector w.r.t. displacement boundary conditions
Kmod = exercise_Kernel.ModifyMasterStiffForDBC([1,2,4],K)
print("Modified Master Stiffness Matrix (Kmod):\n", Kmod, "\n")

fmod = exercise_Kernel.ModifyNodeForcesForDBC([1,2,4],f)
print("Modified Global Force Vector (fmod):\n", fmod, "\n")

# Compute Eigenvalues and Eigenvectors of Kmod
eigenVal, eigenVec = np.linalg.eig(Kmod)
print("Eigenvalues of Kmod:\n", eigenVal, "\n")
print("Eigenvectors of Kmod:\n", eigenVec, "\n")

kLam  = np.dot(Kmod,eigenVec[:,2])
print("####################### ")
print(eigenVal[2])
print(eigenVec[:,2])
print("####################### ")
print(kLam)
print("####################### ")

# Solve system of equations, evaluate vector of displacements u
print("Determinant of Kmod =\n", np.linalg.det(Kmod), "\n")
Kmod_inv = np.linalg.inv(Kmod)
u = Kmod_inv.dot(fmod)
print("Global Vector of Displacements (u):\n", u, "\n")

# External node forces including reactions
fe = np.dot(K,u)
print("Global Vector of External Forces (fe):\n", fe, "\n")

# Determine internal forces
p = exercise_Kernel.GetIntForcesOfExampleTruss(u)
print("Global Vector of Internal Forces (p):\n", p, "\n")

# Generate a plot of increasingly deformed structure
#exercise_Kernel.plotStructure(displacements,NumberOfPlots,ScalingFactor)
exercise_Kernel.plotStructure(u,6,5)