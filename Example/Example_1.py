import sys, os
from jax import grad, jit
from jax import lax
from jax import random
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
# here you may need to point to your locally installed function.
sys.path.insert(0,'/Users/jmw/Documents/GitHub/Point_vortex/')
from initial_condition import *
from mesh import *
from Kernel import *
from time_stepper import *
from utility_functions import *

def main():
    """
    _
    Example 1: this constructs a stochastic solution to the point vortex system. 
    _
    """
    T = 32 #2**-4 # affect nt. 
    dt = 2 ** -3 # affect nt. # -4 seems ecessive
    nt = int(T/dt)
    print("nt", nt) # 320 time steps 
    key = jax.random.PRNGKey(2)
    P = 1
    dbs = jnp.sqrt(dt) * jax.random.normal(key, shape=(nt,P))
    dts = jnp.ones(nt) * dt
    print(f"We use $P = {P}$ basis functions. We use $n_t = {nt}$ timesteps on the interval $t \in [0,{T}]$, with $\Delta t = {dt}$")
    print(f"We use a dBs, a (nt,P) sized sample from the normal distribution with Jax random PRNG Key 2")
    theta_true = {'sigma':0.003*jnp.ones(P)}
    string_for_print = theta_true['sigma']
    print(f'we use $\\theta = {string_for_print}$')

    xmin = 0.;xmax = 1.;ymin = 0.;ymax = 1. # same as the meshgrid for weather, but different number of points. 
    NM = 256#64#128; 
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
    print("starting temporal integraion")
    xy_det = integrate(ssp33, xyarr0, carr, dts, dbs*1, theta_true, delta)# determistic by setting dbs*0
    fig = plt.figure()
    ax = fig.add_subplot()
    for i in range(0,nt+1): # nt+1 initial conditions. 
        plt.clf()
        xarr_det,yarr_det = xy_to_x_and_y(xy_det[i])
        plt.scatter(xarr_det,yarr_det,c=carr/(hdx*hdy),marker='.',s=4,cmap='jet',edgecolors='none')
        plt.title(f"Vorticity at time {i*dt}")
        plt.draw()
        plt.pause(0.001)
    plt.show()

if __name__=="__main__":
    main()