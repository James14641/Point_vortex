import sys, os
from jax import grad, jit
from jax import lax
from jax import random
import jax
jax.config.update("jax_enable_x64", True)# double precision has historically been more stable.
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
cwd = os.getcwd()
print(cwd)
sys.path.insert(0,'/Users/jmw/Documents/GitHub/Point_vortex/')
sys.path.insert(0,cwd)
from initial_condition import *
from mesh import *
from Kernel import *
from time_stepper import *
from utility_functions import *
import time

if jax.devices()[0].platform != 'gpu':
    print("warning: jax not on the gpu. ")
else:
    print(f'using device: {jax.devices()[0]}')

def main():
    """
    _
    Example 0: this constructs a deterministic solution to the point vortex system. 
    _
    """
    T = 32 #2**-4 # affect nt. 
    dt = 2 ** -3 # affect nt. # -4 seems ecessive
    nt = int(T/dt)
    print("nt", nt) # 320 time steps 
    dts = jnp.ones(nt) * dt
    print(f"We use $n_t = {nt}$ timesteps on the interval $t \in [0,{T}]$, with $\Delta t = {dt}$")
    xmin = 0.;xmax = 1.;ymin = 0.;ymax = 1. # same as the meshgrid for weather, but different number of points. 
    NM = 512#512#256#64#128; 
    nx, ny, hdx, hdy, xc, yc, xxc, yyc = mesh_creation(NM,NM,xmin=xmin,xmax=xmax,ymin=ymin,ymax=ymax)
    delta =  1.5*hdx**0.75 #jnp.sqrt(h)  #delta = 1 / jnp.sqrt(32)  # sqrt h # krasney1986 a . ; something like that
    initial_vorticity = compact_vm2(xxc,yyc)#w_0(xxc,yyc)#w_0(xxc, yyc)# Beale_Madja_1985(xxc, yyc)#c_tpm(xxc,yyc)
    xarr = xxc.reshape(-1)
    yarr = yyc.reshape(-1)
    carr = (initial_vorticity).reshape(-1)*hdx*hdy
    index_remove=True
    if index_remove == True:
        index_remove = jnp.argwhere(jnp.abs(carr)<1e-6)
        xarr = jnp.delete(xarr, index_remove)
        yarr = jnp.delete(yarr, index_remove)
        carr = jnp.delete(carr, index_remove)
    xyarr = x_and_y_to_xy(xarr, yarr)
    xyarr0 = xyarr
    print(f"Number of points: {xyarr0.shape[0]/2}")

    plt.clf()
    xarr_det,yarr_det = xy_to_x_and_y(xyarr0)
    plt.scatter(xarr_det,yarr_det,c=carr/(hdx*hdy),marker='.',s=4,cmap='jet',edgecolors='none')
    plt.title(f"Vorticity at initial time {0}")
    plt.draw()
    plt.pause(0.001)
    plt.colorbar(label='Vorticity')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.savefig(f'{cwd}/Initial_time_vorticity.png', dpi=600, bbox_inches='tight')
    plt.show()

    print("starting temporal integraion")
    start_time = time.time()
    xy_det = integrate_deterministic_no_save(ssp33_deterministic, xyarr0, carr, dts, delta) # deterministic by setting dbs*0
    end_time = time.time()
    print(f"Time taken for computation: {end_time - start_time} seconds")
    print("finished temporal integraion")
    fig = plt.figure()
    ax = fig.add_subplot()
    xarr_det,yarr_det = xy_to_x_and_y(xy_det)
    plt.clf()
    plt.scatter(xarr_det,yarr_det,c=carr/(hdx*hdy),marker='.',s=4,cmap='jet',edgecolors='none')
    plt.title(f"Vorticity at final time {T}")
    plt.draw()
    plt.pause(0.001)
    plt.colorbar(label='Vorticity')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.savefig(f'{cwd}/Final_time_vorticity.png', dpi=600, bbox_inches='tight')
    plt.show()

if __name__=="__main__":
    main()