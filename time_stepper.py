import sys, os
import numpy as np
from jax import *
from Kernel import *
from utility_functions import *

def vector_field_basis(xarr,yarr,n):
    """_stochastic basis of noise function, n is the basis number
        one can change the basis_

    Args:
        xarr (_type_): _description_
        yarr (_type_): _description_
        n (_type_): _description_

    Returns:
        _type_: _description_
    """
    # because u = -\nabla^{perp}; \nabla^{perp} = (-\partial_y,\partial_x). defined from stream sin(2 \pi n x)+ sin(2 \pi n y)
    u =  2*np.pi*n/1*jnp.cos(2*np.pi*n/1*yarr) # u = \partial_y (\psi)
    v = -2*np.pi*n/1*jnp.cos(2*np.pi*n/1*xarr) # v = - \partial_x (\psi)
    return u,v

def sigma(xyarr, theta):
    """___
    Given a set of points, 
    Which can be xyarr for the point vortices, 
    Or another set of flattened points,
    sigma works out the stochastic velocity contribution on them,
    by creating a matrix to be multiplied a d-dimensional Brownian motion. 
    ___
    Args:
        xyarr (_type_): _description_
        theta (_type_): _description_
    Returns:
        _type_: _description_
    """
    x,y = xy_to_x_and_y(xyarr)
    P = len(theta['sigma'])
    Matrix = jnp.zeros([len(xyarr),P])
    for p in range(0,P):
        u,v = vector_field_basis(x,y,p+1)
        uv = x_and_y_to_xy(u,v)
        Matrix = Matrix.at[:,p].set(uv*theta['sigma'][p]) # \in R^{2d,P}
    return Matrix  



def V_sigma(xyarr, theta):
    """_Vectorised version of the sigma function_

    Args:
        xyarr (_type_): _flattened xy array_
        theta (_type_): _theta array_

    Returns:
        _type_: _description_
    """
    x, y = xy_to_x_and_y(xyarr)
    P = len(theta['sigma'])
    def update_matrix_column(p):
        u, v = vector_field_basis(x, y, p + 1)
        uv = x_and_y_to_xy(u, v)
        return uv * theta['sigma'][p]
    # Use vmap to vectorize the loop over P
    updated_columns = jax.vmap(update_matrix_column)(jnp.arange(P))
    # Stack the columns to create the final matrix
    updated_matrix = jnp.column_stack(updated_columns)
    return updated_matrix

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
    """ Returns deterministic velocity at the points xarrd,yarrd, 
    given points of vorticity (xarr, yarr), carr

    Args:
        xarrd (_type_): _description_
        yarrd (_type_): _description_
        xarr (_type_): _description_
        yarr (_type_): _description_
        carr (_type_): _description_
        delta (_type_): _description_

    Returns:
        _type_: _description_
    """
    u_d,v_d = Velocity_at_Field(xarrd, yarrd, xarr, yarr, carr, delta)
    uv_d = x_and_y_to_xy(u_d,v_d)
    return uv_d



def CFE(xyarr, carr, dt, delta ):
    """_Forward Euler flow map_

    Args:
        xyarr (_type_): _description_
        carr (_type_): _description_
        dt (_type_): _description_
        delta (_type_): _description_

    Returns:
        _type_: _description_
    """
    xarr, yarr = xy_to_x_and_y(xyarr)
    u_vel, v_vel = VelocityUV(xarr,yarr,carr,delta)
    xarr = xarr + dt*u_vel
    yarr = yarr + dt*v_vel
    xyarr = x_and_y_to_xy(xarr,yarr)
    return xyarr

def euler(xyarr, carr, sigma, dt, dz, theta, delta):
    """_Euler Maruyama scheme_

    Args:
        xyarr (_type_): _description_
        carr (_type_): _vortex strenghts length_
        sigma (_type_): _description_
        dt (_type_): _description_
        dz (_type_): _description_
        theta (_type_): _description_
        delta (_type_): _description_

    Returns:
        _type_: _description_
    """
    xyarr = CFE(xyarr, carr, dt ,delta) + sigma(xyarr, theta) @ dz
    return xyarr

def euler_diffusion(xyarr, carr, sigma, dt, dz, dw, theta, delta, visc):
    "adds nu dW"
    xyarr = euler(xyarr, carr, sigma, dt, dz, theta, delta) + visc*dw
    return xyarr

def ssp33(xyarr, carr, sigma, dt, dz, theta, delta):
    """_ssp33 scheme see shu osher 1988, note that in the stochastic setting this is not ssp._

    Args:
        xyarr (_type_): _description_
        carr (_type_): _description_
        sigma (_type_): _description_
        dt (_type_): _description_
        dz (_type_): _description_
        theta (_type_): _description_
        delta (_type_): _description_

    Returns:
        _type_: _description_
    """
    f1 = euler(xyarr, carr, sigma, dt, dz, theta, delta)
    f1 = 3 / 4 * f1 + 1 / 4 * euler(f1, carr, sigma, dt, dz, theta, delta)
    f1 = 1 / 3 * f1 + 2 / 3 * euler(f1, carr, sigma, dt, dz, theta, delta)
    return f1


def ARK_SSP33_EM(xyarr, carr, sigma, dt, dz, dw, theta, delta, visc):
    """_additive RK scheme that introduces Ito term representing diffusion._

    Args:
        xyarr (_type_): _description_
        carr (_type_): _description_
        sigma (_type_): _description_
        dt (_type_): _description_
        dz (_type_): _description_
        theta (_type_): _description_
        delta (_type_): _description_

    Returns:
        _type_: _description_
    """
    f1 = euler_diffusion(xyarr, carr, sigma, dt, dz, dw, theta, delta, visc)
    f1 = 3 / 4 * f1 + 1 / 4 * euler(f1, carr, sigma, dt, dz, theta, delta)
    f1 = 1 / 3 * f1 + 2 / 3 * euler(f1, carr, sigma, dt, dz, theta, delta)
    return f1

def integrate(step, xy0, carr, dts, dzs, theta, delta):
    """_integration of scheme see https://github.com/jax-ml/jax/blob/main/jax/_src/lax/_

    Args:
        step (_type_): _name of scheme e.g. ssp33_
        xy0 (_type_): _description_
        carr (_type_): _description_
        dts (_type_): _description_
        dzs (_type_): _description_
        theta (_type_): _description_
        delta (_type_): _description_
    """
    def body(xyarr, dt_dz):
        dt, dz = dt_dz
        xyarr = step(xyarr, carr, sigma ,dt, dz, theta, delta)
        return xyarr, xyarr
    _, xys = jax.lax.scan(body, xy0, (dts, dzs))
    return jnp.insert(xys, 0, xy0, axis=0)


def integrate_new(step, xy0, carr, dts, dzs, dws, theta, delta, visc):

    def body(xyarr, dt_dz_dw):
        dt, dz, dw = dt_dz_dw
        xyarr = step(xyarr, carr, sigma ,dt, dz, dw, theta, delta, visc)
        return xyarr, xyarr
    
    _, xys = jax.lax.scan(body, xy0, (dts, dzs, dws))
    return jnp.insert(xys, 0, xy0, axis=0)






