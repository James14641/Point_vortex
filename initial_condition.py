import sys, os
import jax.numpy as jnp

def w_0(xxc, yyc):
    """_Initial conditions that are specified by Wei_

    Args:
        xxc (_type_): _x component of a meshgrid_
        yyc (_type_): _y component of a meshgrid_

    Returns:
        _type_: _description_
    """
    ic = jnp.sin(8 * jnp.pi * xxc) * jnp.sin(8 * jnp.pi * yyc) + \
         0.4 * jnp.cos(6 * jnp.pi * xxc) * jnp.cos(6 * jnp.pi * yyc) + \
         0.3 * jnp.cos(10 * jnp.pi * xxc) * jnp.cos(4 * jnp.pi * yyc) + \
         0.02 * jnp.sin(2 * jnp.pi * xxc) + 0.02 * jnp.sin(2 * jnp.pi * yyc)
    return ic

def Beale_Madja_1985(xxc, yyc):
    """_Initial conditions that are specified by Beale and Madja in 1985_

    Args:
        xxc (_type_): _x component of a meshgrid_
        yyc (_type_): _y component of a meshgrid_

    Returns:
        _type_: _meshgrid of values_
    """
    r_sqrd = xxc**2 + yyc**2
    ans = (1 - r_sqrd)**3
    ic = jnp.where(r_sqrd <= 1, ans, 0)
    return ic

def compact_vm(xx,yy):
    """_Initial conditions describing positions_

    Args:
        xx (_type_): _description_
        yy (_type_): _description_

    Returns:
        _type_: _description_
    """
    R = 0.125
    r1 = ((xx-0.5-0.125)**2 + (yy-0.5)**2)**0.5
    r2 = ((xx-0.5+0.125)**2 + (yy-0.5)**2)**0.5
    _ic = jnp.where(r1<R,jnp.cos(0.5*np.pi*r1/R )**2+0.5,0) + jnp.where(r2<R,jnp.cos(0.5*np.pi*r2/R)**2+0.5,0)
    return _ic/2

def compact_vm2(xx,yy):
    R = 0.125
    r1 = ((xx-0.5-0.125)**2+(yy-0.5)**2)**0.5
    ans1 = (1 - (r1/R)**2)**3
    r2 = ((xx-0.5+0.125)**2+(yy-0.5)**2)**0.5
    ans2 = (1 - (r2/R)**2)**3
    _ic = jnp.where(r1<R,(ans1+1)/2,0)+ jnp.where(r2<R,(ans2+1)/2,0)
    return _ic