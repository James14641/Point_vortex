import sys, os
import jax.numpy as jnp

def xy_to_x_and_y(xyarr):
    nv = int(len(xyarr)/2)
    xarr = xyarr[0:nv]
    yarr = xyarr[nv:]
    return xarr, yarr

def x_and_y_to_xy(xarr, yarr):
    xyarr = jnp.hstack((xarr,yarr))
    return xyarr

