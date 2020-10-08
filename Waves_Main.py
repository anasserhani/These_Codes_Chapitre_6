### Packages 
import matplotlib.pyplot as plt
import numpy as np
from Waves import Wave_2D
from math import pi


### Class construction
W = Wave_2D()


### Spatial domain
x1_0, x1_ell, x2_0, x2_ell = 0., 2., 0., 1.
W.Set_Rectangular_Domain(x1_0, x1_ell, x2_0, x2_ell)


### Time interval
t_initial, t_final  = 0., 6.
W.Set_Initial_Final_Time(t_initial, t_final)


### Physical parameters
Rho =  'x[0]*x[0] - x[1]*x[1] + 2' 

T11 = 'cos(x[0]*x[1]) + 4' 
T12 = ' x[1]' 
T22 = 'abs(x[0]-x[1])+2' 

W.Set_Physical_Parameters(Rho, T11, T12, T22);


### Dissipations
Z = '0.1'

eps = '4 * x[0] * (xL - x[0]) * x[1] * (yL - x[1])'

kappa_11 = '(x[0]+x[1]+1)/6' 
kappa_12 = 'x[1]/6'
kappa_22 = 'x[0]+ x[1] + 3'

W.Set_Damping(damp=['impedance_mbc', 'fluid', 'viscoelastic'],\
              Z=Z, eps=eps, k11=kappa_11, k12=kappa_12, k22=kappa_22)


### Initial data
Aq_0_1 = '0'
Aq_0_2 = '0'
Ap_0   = '0'
W_0    = '0'
W.Set_Initial_Data(Aq_0_1=Aq_0_1, Aq_0_2=Aq_0_2, Ap_0=Ap_0, W_0=W_0)


### Mixed boundaries
Gamma_1 = 'G1'
Gamma_2 = 'G2'
Gamma_3 = 'G3'
Gamma_4 = 'G4'
W.Set_Mixed_Boundaries(Dir=[Gamma_1], Nor=[Gamma_3], Imp=[Gamma_2, Gamma_4])


### Dirichlet boundary condition
f0_Dir = lambda t : 0 * np.sin( 2 * 2*pi/t_final *t) * 25
g0_Dir = '10'
W.Set_Mixed_BC_Dirichlet(Ub_tm0= f0_Dir, Ub_sp0=g0_Dir,\
                         Ub_tm0_dir = lambda t : 0)


### Normal trace boundary condition
f0_Nor = lambda t:  np.sin( 5 * 2*pi/t_final *t) * 50 
g0_Nor = 'x[1] * sin(pi*(1-x[1]))'
W.Set_Mixed_BC_Normal(Ub_tm0=f0_Nor, Ub_sp0=g0_Nor)

### Check problem definition
assert W.Check_Problem_Definition() == 1,\
         "Problem definition to be checked again !"
         
 
### Mesh
W.Set_Gmsh_Mesh('rectangle.xml', rfn_num=3)
W.Plot_Mesh()


### Finite element families
W.Set_Finite_Element_Spaces(family_q='RT', family_p='P', family_b='P',\
                            rq=0, rp=1, rb=1)
W.Plot_Mesh_with_DOFs_MBC()


### Assembly
W.Assembly_Mixed_BC()
W.Plot_Sparsity()

### Interpolations
W.Project_Initial_Data()
W.Project_Boundary_Control()


### Discrete time setting
dt     = 1.e-4
solver = 'DAE:RK4_Augmented'
W.Set_Time_Setting(dt)


### Solve
Alpha, Hamiltonian = W.Time_Integration(solver)


### Plot Hamiltonian
W.Plot_Hamiltonian(W.tspan, Hamiltonian, linewidth=3)


### Plots energy variables
Aq = W.Get_Strain(Alpha)
Ap = W.Get_Linear_Momentum(Alpha)

W.Quiver(Aq, t=t_final, title="Déformation à l'instant t=t_final", with_mesh=True)

W.Trisurf(Ap, t=t_initial, title="Moment linéaire à l'instant t=t_inital")
W.Trisurf(Ap, t=t_final, title="Moment linéaire à l'instant t=t_final")




