import sys, os
import jax.numpy as jnp
from utility_functions import *

def L(x):
    """Laguerre polynomial, poly order, double then plus 2 is the order of convergence of the vortex method"""
    ans = 1 - 3*x + 3*x**2 - x**3/6 # 3rd order double then plus 2 is the order of method
    #ans = 1 - x # 1st order laguere poly, is a quadratic in r, and a 4th order method. 
    #ans = 1/720 * (720 - 4320*x + 5400*x**2 - 2400*x**3 + 450*x**4 - 36*x**5 + 1*x**6) #6, 12, 14.
    return ans

def VelocityUV(xarr, yarr, carr, delta):
    u_vel, v_vel = Velocity_at_Field(xarr, yarr, xarr, yarr, carr, delta)
    return u_vel, v_vel

def Velocity_at_Field(xarrd, yarrd, xarr, yarr, carr, delta):
    """we extend VelocityUV, to evaluate at an arbitrary set of values defined by xarrd,yarrd."""
    x_diff_2 = xarrd[:, jnp.newaxis] - xarr 
    y_diff_2 = yarrd[:, jnp.newaxis] - yarr
    rsq = x_diff_2**2 + y_diff_2**2 
    denominator1 = jnp.where(rsq<1e-12,1e-12,rsq) # prevent division by zero numerically, sensitive
    mol = (1 - L(rsq /delta**2 )*jnp.exp(-rsq /delta**2)) # uses p-th order kernel. specified by L
    u_vel = - ( 1 / (2 * jnp.pi) * (y_diff_2 / denominator1) * mol ) @ carr
    v_vel =   ( 1 / (2 * jnp.pi) * (x_diff_2 / denominator1) * mol ) @ carr
    return u_vel, v_vel

def det_vel(xarrd, yarrd, xarr, yarr, carr, delta):
    """returns deterministic velocity at the points xarrd,yarrd, given points of vorticity (xarr, yarr), carr"""
    u_d,v_d = Velocity_at_Field(xarrd, yarrd, xarr, yarr, carr, delta)
    uv_d = x_and_y_to_xy(u_d,v_d)
    return uv_d