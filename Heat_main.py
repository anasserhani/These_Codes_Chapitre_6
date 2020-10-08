import matplotlib.pyplot as plt
import numpy as np
from Heat import Heat_2D, Lyapunov_2D, Entropy_2D, Energy_2D
from dolfin import *
from math import pi

#%% Choose what to run
lyapunov_run = True
entropy_run = True
energy_run = True

#%% Heat_2D
### Construction
Heat = Heat_2D()

### Rectangular domain
x1_0, x1_ell, x2_0, x2_ell = 0., 2., 0., 1.
Heat.Set_Rectangular_Domain(x1_0, x1_ell, x2_0, x2_ell);

### Time
t_initial, t_final  = 0., 2.
Heat.Set_Initial_Final_Time(t_initial, t_final);

### Physical parameters
Rho    =  'x[0]*x[0] - x[1]*x[1] + 2' 

Lambda11 = '5. + x[0]*x[1]'
Lambda12 = '(x[0]-x[1])*(x[0]-x[1])'
Lambda22 = '3.+x[1]/(x[0]+1)'

CV = '3.'

Heat.Set_Physical_Parameters(rho=Rho, CV=CV,\
                          Lambda11=Lambda11, Lambda12=Lambda12, Lambda22=Lambda22);

### Boundary Control
level, amplitude   = 100, 25
Ub_sp0  = ''' 
        ( abs(x[0]) <= DOLFIN_EPS ? - x[1] * (yL-x[1]) : 0 )            
        + ( abs(xL - x[0]) <= DOLFIN_EPS ? exp(x[1] * (yL-x[1])) -1  : 0 )      
        + ( abs(x[1]) <= DOLFIN_EPS ?  - x[0] * (xL-x[0])/3  : 0 )            
        + ( abs(yL - x[1]) <= DOLFIN_EPS ?  pow(x[0],2) * (xL-x[0])/4  : 0 )     
        '''
Ub_tm0  = lambda t : sin(4 * 2*pi/t_final * t) * 5e3
Heat.Set_Boundary_Control(Ub_tm0=Ub_tm0, Ub_sp0=Ub_sp0, 
                          Ub_tm1=lambda t : 0, Ub_sp1='5000', levinit=level);
                          
                   
#%% Discretization of Heat_2D
### Mesh
Heat.Set_Gmsh_Mesh(xmlfile='rectangle.xml', rfn_num=0);
Heat.Plot_Mesh()

### Finite elements families
Heat.Set_Finite_Element_Spaces(family_scalar='P', family_Vector='RT', family_boundary='P', rs=1, rV=0, rb=1);
Heat.Plot_Mesh_with_DOFs()

### Assembly
Heat.Assembly();


#%% Lyapunov 
if lyapunov_run :
    print('\n')
    print('LYAPUNOV')
    
    ### Inheritance 
    Lya             = Lyapunov_2D()
    Lya.__dict__    = Heat.__dict__
    
    ###  Initial data
    ampl, sX, sY, X0, Y0    = 500, Heat.xL/6, Heat.yL/6, Heat.xL/2, Heat.yL/2 
    Gaussian_init           = Expression(' ampl * exp(- pow( (x[0]-X0)/sX, 2) - pow( (x[1]-Y0)/sY, 2) )', 
                                         degree=2, 
                                         ampl=Constant(ampl), 
                                         sX=Constant(sX), sY=Constant(sY),
                                         X0=Constant(X0), Y0=Constant(Y0) )
    Ginit                   = interpolate(Gaussian_init, Heat.Vs)  
    Lya.Set_Initial_Data(init_by_vector=True, eT0=Ginit.vector()[:]);     
    
    ### Discretization formulation
    Lya.Set_Formulation('div');
    
    ### Projections
    Lya.Project_Initial_Data();
    Lya.Project_Boundary_Control();
    
    ### Time setting
    dt     = 1.e-4
    Lya.Set_Time_Setting(dt);
    
    ### Time-stepping
    eT, eQ, Lyapunov = Lya.Integration_ODE_RK4();
    
    ### Plot Hamiltonian
    Lya.Plot_Hamiltonian(Heat.tspan, Lyapunov, linewidth=3, title='Hamiltonien quadratique')
    
    # Energy variables
    Heat.Contour_Quiver(var_scalar=eT, var_Vect=eQ, with_mesh=True, t=t_initial,\
                    title="Les variables d'énergie de la formulation quadratique à l'instant t=t_initial",\
                    margin=0.05, save=False, figsize=(14,5))
    Heat.Contour_Quiver(var_scalar=eT, var_Vect=eQ, with_mesh=True, t=t_final,\
                    title="Les variables d'énergie de la formulation quadratique à l'instant t=t_final",\
                    margin=0.05, save=False, figsize=(14,5))


#%% Entropy
if entropy_run :
    print('\n')
    print('ENTROPY')
    
    ### Inheritance 
    Ent             = Entropy_2D()
    Ent.__dict__    = Heat.__dict__
    
    ### Initial Data
    ampl, sX, sY, X0, Y0    = 500, Heat.xL/6, Heat.yL/6, Heat.xL/2, Heat.yL/2 
    Gaussian_init           = Expression(' CV * (ampl * exp(- pow( (x[0]-X0)/sX, 2) - pow( (x[1]-Y0)/sY, 2) )+1000)', 
                                         degree=2, 
                                         ampl=Constant(ampl), CV=Constant(Heat.CV(0)),
                                         sX=Constant(sX), sY=Constant(sY),
                                         X0=Constant(X0), Y0=Constant(Y0) )
    Ginit                   = interpolate(Gaussian_init, Heat.Vs)  
    Ent.Set_Initial_Data(init_by_vector=True, As0=Ginit.vector()[:]);
    
    ### Discretization formulation
    Ent.Set_Formulation('grad');
    
    ### Projections
    Ent.Project_Initial_Data();
    Ent.Project_Boundary_Control();
    
    ### Time setting
    dt     = 1.e-4
    Ent.Set_Time_Setting(dt);
    
    ### Time-stepping
    As, fS, es, eS, Entropy = Ent.Integration_DAE()

    ### Post-processing
    Heat.Plot_Hamiltonian(Heat.tspan, Entropy, linewidth=3, title='Hamiltonien entropique')

    Heat.Contour_Quiver(var_scalar=As, var_Vect=eS, with_mesh=True, t=t_initial,\
                    title="Les variables de la formulation entropique à l'instant t=t_initial",\
                    margin=0.05, save=False, figsize=(14,5))
    Heat.Contour_Quiver(var_scalar=As, var_Vect=eS, with_mesh=True, t=t_final,\
                    title="Les variables de la formulation entropique à l'instant t=t_final",\
                    margin=0.05, save=False, figsize=(14,5))



#%% Energy
if energy_run:
    print('\n')
    print('ENERGY')
    ### Inheritance 
    Ene             = Energy_2D()
    Ene.__dict__    = Heat.__dict__
    
    ### Initial data
    ampl, sX, sY, X0, Y0    = 1500, Heat.xL/6, Heat.yL/6, Heat.xL/2, Heat.yL/2 
    Gaussian_init           = Expression(' ampl * exp(- pow( (x[0]-X0)/sX, 2) - pow( (x[1]-Y0)/sY, 2) )+ 5000', 
                                     degree=2, 
                                     ampl=Constant(ampl), CV=Constant(Heat.CV(0)),
                                     sX=Constant(sX), sY=Constant(sY),
                                     X0=Constant(X0), Y0=Constant(Y0) )
    Ginit                   = interpolate(Gaussian_init, Heat.Vs)  
    Ene.Set_Initial_Data(init_by_vector=True, au0=5000*np.ones(Heat.Ns), eu0=Ginit.vector()[:]);
    
#    ### Boundary control
#    Ub_sp0 = '-2 * x[0] + 1'
#    def Ub_tm0(t):
#        if t <= Heat.tfinal/3 :
#            return 0 
#        else :
#            return  750 * np.sin(3 * 2*pi/Heat.tfinal * t)
#
##    Ubtm0 = []
##    for i in range(Heat.Nt+1):
##        Ubtm0.append(Heat.Ub_tm0(Heat.tspan[i]))
##    plt.plot(Heat.tspan, Ubtm0)
#    
#    Ene.Set_Boundary_Control(Ub_tm0=Ub_tm0, Ub_sp0=Ub_sp0, Ub_tm1=lambda t : 0, Ub_sp1='5000');
    
    
    ### Formulation
    Ene.Set_Formulation('div');
    
    ### Projections
    Ene.Project_Initial_Data();
    Ene.Project_Boundary_Control();
    
    ### Time setting
    dt     = 1.e-3
    Heat.Set_Time_Setting(dt);
    
    ### Time-stepping
    au, fU, fsig, eu, eU, esig, Energie = Ene.Integration_DAE();
    
    ### Post-Processing
    Heat.Plot_Hamiltonian(Heat.tspan, Energie, linewidth=3, title='Hamiltonien énergetique')
    Heat.Contour_Quiver(var_scalar=au, var_Vect=eU, with_mesh=True, t=t_initial,\
                    title="Les variables de la formulation énergetique à l'instant t=t_initial",\
                    margin=0.05, save=False, figsize=(14,5))
    Heat.Contour_Quiver(var_scalar=au, var_Vect=eU, with_mesh=True, t=t_final,\
                    title="Les variables de la formulation énergetique à l'instant t=t_final",\
                    margin=0.05, save=False, figsize=(14,5))
    
    
#%%
import sys
sys.exit()

#%%
time_space_Vector = au
fig = plt.figure(figsize=(14,8))
ax = fig.gca(projection='3d')
ax.set_title(r'Internal energy $\alpha_u$')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_xlim(0, Heat.xL)
ax.set_ylim(0, Heat.yL)
ax.set_zlim(np.min(time_space_Vector), np.max(time_space_Vector))
wframe = None
for i in range(0, Heat.Nt+1, 25):
    if wframe:
        ax.collections.remove(wframe)
    wframe = ax.plot_trisurf(Heat.xs, Heat.ys, time_space_Vector[:,i], linewidth=0.2, antialiased=True, cmap=plt.cm.jet)
    ax.set_zlabel('time=' + np.array2string(Heat.tspan[i]) + '/' + np.array2string(Heat.tspan[-1]) ) 
    plt.pause(.001)











                          