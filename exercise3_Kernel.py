#####################################################################
# Technische Universitaet Muenchen
# Chair of Structural Analysis
# IFEM - Introduction to Finite Element Methods
# Dr.-Ing. Roland Wuechner
# Contact: Klaus B. Sautter
# klaus.sautter@tum.de
#####################################################################
# Exercise 5 - Kernel File
#####################################################################

# importing required modules
import numpy as np
from matplotlib import pyplot as plt

def FormElemStiff2DTwoNodeBar(xyi,xyj,E,A):
    #----------------------------------------------------------
    # Element Stiffness Function for 2D Two-Node Bar
    #----------------------------------------------------------
    # input:
    # xyi[2] ... x,y coordinates node i
    # xyj[2] ... x,y coordinates node j
    # E ... Young's modulus
    # A ... cross sectional area
    # output:
    # Ke[4,4] ... element stiffness matrix
    #----------------------------------------------------------
    dx = xyj[0] - xyi[0]
    dy = xyj[1] - xyi[1]
    L = np.sqrt(dx**2 + dy**2)
    
    c = dx / L
    cc = c**2
    s = dy / L
    ss = s**2
    cs = c * s
    
    T = np.matrix( ((cc, cs,-cc,-cs),
                     (cs, ss,-cs,-ss),
                     (-cc,-cs, cc, cs),
                     (-cs,-ss, cs, ss)) )
    
    Ke = T * E * A / L
    
    return Ke

def MergeElemIntoMasterStiff(Ke,eft,Kin):
    #----------------------------------------------------------
    # Merge element stiff matix into master matrix
    #----------------------------------------------------------
    # input:
    # Ke[4,4] ... element stiffness matrix
    # eft[4] ... element freedom table
    # Kin[n,n] ... master stiffness matrix, ndof = n
    # output:
    # Kin[n,n] ... modified master stiffness matrix
    #----------------------------------------------------------
    for i in range(0, 4):
        ii = eft[i]-1
        for j in range(i,4):
            jj = eft[j]-1
            
            Kin[ii,jj] = Kin[ii,jj] + Ke[i,j]
            Kin[jj,ii] = Kin[ii,jj]

    return Kin
     
def ModifyMasterStiffForDBC(pdof,K):
    #----------------------------------------------------------
    # Modify master for displ. boundary conditions
    #----------------------------------------------------------
    # input:
    # pdof[npdof] ... npdof prescribed degrees of freedom
    # K[nk,nk] ... master stiffness matrix, ndof = nk
    # output:
    # Kmod[nk,nk] ... modified master stiffness matrix
    #----------------------------------------------------------
    nk = np.shape(K)[0]
    npdof = np.shape(pdof)[0]
    
    # ---------- copy master stiffness matrix -----------------
    Kmod = np.copy(K)
    # ---------- evaluate prescribed degrees of freedom--------
    for k in range(0, npdof):
        i = pdof[k]-1
    # ---------- clear rows and columns -----------------------
        for j in range(0, nk):
            Kmod[i,j] = 0
            Kmod[j,i] = 0
    # ---------- set diagonal to 1 ----------------------------
        Kmod[i,i] = 1

    return Kmod

def ModifyNodeForcesForDBC(pdof,f):
    #----------------------------------------------------------
    # Modify node forces for displ. boundary cond.
    #----------------------------------------------------------
    # input:
    # pdof[npdof] ... npdof prescribed degrees of freedom
    # f ... vector of node forces
    # output:
    # fmod ... modified vector of node forces
    #----------------------------------------------------------
    npdof = np.shape(pdof)[0]
    
    fmod = np.copy(f)

    for k in range(0, npdof):
        i = pdof[k]-1
        fmod[i] = 0
    
    return fmod

def GetIntForce2DTwoNodeBar(xyi,xyj,E,A,eft,u):
    #----------------------------------------------------------
    # Determine internal force of individual element
    #----------------------------------------------------------
    # input:
    # xyi[2] ... x,y coordinates node i
    # xyj[2] ... x,y coordinates node j
    # E ... Young's modulus
    # A ... cross sectional area
    # eft[4] ... element freedom table
    # u ... system displacement vector
    # output:
    # Fi ... internal force
    #----------------------------------------------------------
    dx = xyj[0]-xyi[0]
    dy = xyj[1]-xyi[1]
    L = np.sqrt(dx**2 + dy**2)
    
    c = dx / L
    s = dy / L
    
    ix = eft[0]-1
    iy = eft[1]-1
    jx = eft[2]-1
    jy = eft[3]-1
    
    ubar = np.array([c * u[ix] + s * u[iy], -s * u[ix] + c * u[iy], 
                     c * u[jx] + s * u[jy], -s * u[jx] + c * u[jy]])
    
    eps = (ubar[2] - ubar[0])/L
    
    Fi = E * A * eps
    
    return Fi

def AssembleMasterStiffOfExampleTruss():
    #----------------------------------------------------------
    # Assembling master stiffness matrix
    #----------------------------------------------------------
    K = np.zeros((6,6))

    Ke = FormElemStiff2DTwoNodeBar([0,0],[10,0],100,1)
    K = MergeElemIntoMasterStiff (Ke,[1,2,3,4],K)
    Ke = FormElemStiff2DTwoNodeBar([10,0],[10,10],100,1/2) #1.0E-14
    K = MergeElemIntoMasterStiff (Ke,[3,4,5,6],K)
    Ke = FormElemStiff2DTwoNodeBar([0,0],[10,10],100,2*np.sqrt(2))
    K = MergeElemIntoMasterStiff (Ke,[1,2,5,6],K)
    
    return K

def GetIntForcesOfExampleTruss(u):
    #----------------------------------------------------------
    # Evaluate internal forces
    #----------------------------------------------------------
    # input:
    # u ... displacement vector
    #----------------------------------------------------------
    f = np.zeros(3)
    f[0] = GetIntForce2DTwoNodeBar([0,0] ,[10,0] ,100,1,[1,2,3,4],u)
    f[1] = GetIntForce2DTwoNodeBar([10,0],[10,10],100,1/2,[3,4,5,6],u)
    f[2] = GetIntForce2DTwoNodeBar([0,0] ,[10,10],100,2*np.sqrt(2),[1,2,5,6],u)
    
    return f
    
def plotStructure(u, noPlots, scalFac):
    for i in range(noPlots):
        # [1,2,3,1]
        x1 = 0 + i/noPlots*u[0]*scalFac
        x2 = 10 + i/noPlots*u[2]*scalFac
        x3 = 10 + i/noPlots*u[4]*scalFac
        y1 = 0 + i/noPlots*u[1]*scalFac
        y2 = 0 + i/noPlots*u[3]*scalFac
        y3 = 10 + i/noPlots*u[5]*scalFac
        if i == 0:
            plt.plot([x1,x2,x3,x1], [y1,y2,y3,y1],'bo-',label='Initial Configuration',lw=2)
        elif i == noPlots-1:
            plt.plot([x1,x2,x3,x1], [y1,y2,y3,y1],'ro-',label='Final Configuration',lw=2)
        else:
            plt.plot([x1,x2,x3,x1], [y1,y2,y3,y1],'kx--')

    plt.margins(0.1,0.1) # add 10% margin
    plt.title('Structural Deformation',fontsize=20)
    plt.xlabel('x',fontsize=15)
    plt.ylabel('y',fontsize=15)
    plt.legend(loc='best')
    plt.show()
    
print("\n*****************************\n********** 3. Exercise **********\n*****************************\n")