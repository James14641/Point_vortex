
# Point vortex library

- <strong>Autodifferentiable<strong>
- <strong>CPU-GPU tested -jax[cpu] - jax[cuda12]<strong>
- <strong>Works on nvidia cluster and tested locally on different macOS<strong>

![image of final solution](Final_time_vorticity.png)

# References 

This code is a usable subset of the code developed for the paper:

Stochastic fluids with transport noise: Approximating diffusion from data using SVD and ensemble forecast back-propagation
###https://arxiv.org/pdf/2405.00640
@article{woodfield2024stochastic,
  title={Stochastic fluids with transport noise: Approximating diffusion from data using SVD and ensemble forecast back-propagation},
  author={Woodfield, James},
  journal={arXiv preprint arXiv:2405.00640},
  year={2024}
}
If you use this code or find it helpful please cite the above paper, and at least the additional references (Chorin, Madja + Bertozzi, Rüemelin).

It solves inviscid vortex models, with/without stochastic noise. It is differentiable. Available under a MIT Liscence.  

# To run:
Attain a copy of the code and run it e.g. : 
- <strong>-python3 example_0.py<strong>

### The below timesteppers are differentiable to optimise solution trajectories. (Ensemble 4Dvar)
- <strong>-python3 example_1.py<strong>
- <strong>-python3 example_2.py<strong>
noting that the entire solution trajectory is saved to local memory. 

### The below timestepers just give final condition.
- <strong>-python3 example_3.py<strong>
- <strong>-python3 example_4.py<strong>

### Dependencies: 
jax, matplotlib... I use some other librarys also. 

### Example zero solves
$$\boldsymbol{x}(\boldsymbol{X}, t)=\boldsymbol{x}(\boldsymbol{X}, 0)+\int_0^t \boldsymbol{u}(\boldsymbol{x}(\boldsymbol{X}, s), s) d s; \quad \boldsymbol{x}(\boldsymbol{X}, 0)=\boldsymbol{X}
$$
Where the Kernel is specified by the Euler Kernel. 

### Example one solves: 
$$\boldsymbol{x}(\boldsymbol{X}, t)=\boldsymbol{x}(\boldsymbol{X}, 0)+\int_0^t \boldsymbol{u}(\boldsymbol{x}(\boldsymbol{X}, s), s) d s+\sum_{p=1}^P \int_0^t \theta_p \boldsymbol{\xi}_p(\boldsymbol{x}(\boldsymbol{X}, s)) \circ d B^p(s) ; \quad \boldsymbol{x}(\boldsymbol{X}, 0)=\boldsymbol{X}
$$

### Examples two solves: 
$$\boldsymbol{x}(\boldsymbol{X}, t)=\boldsymbol{x}(\boldsymbol{X}, 0)+\int_0^t \boldsymbol{u}(\boldsymbol{x}(\boldsymbol{X}, s), s) d s+\sum_{p=1}^P \int_0^t \theta_p \boldsymbol{\xi}_p(\boldsymbol{x}(\boldsymbol{X}, s)) \circ d B^p(s) +  \int_0^t \nu d W(s) ; \quad \boldsymbol{x}(\boldsymbol{X}, 0)=\boldsymbol{X}
$$

### Example three solves: 
$$\boldsymbol{x}(\boldsymbol{X}, t)=\boldsymbol{x}(\boldsymbol{X}, 0)+\int_0^t \boldsymbol{u}(\boldsymbol{x}(\boldsymbol{X}, s), s) d s+\sum_{p=1}^P \int_0^t \theta_p \boldsymbol{\xi}_p(\boldsymbol{x}(\boldsymbol{X}, s)) \circ d B^p(s) ; \quad \boldsymbol{x}(\boldsymbol{X}, 0)=\boldsymbol{X}
$$

### Examples four solves: 
$$\boldsymbol{x}(\boldsymbol{X}, t)=\boldsymbol{x}(\boldsymbol{X}, 0)+\int_0^t \boldsymbol{u}(\boldsymbol{x}(\boldsymbol{X}, s), s) d s+\sum_{p=1}^P \int_0^t \theta_p \boldsymbol{\xi}_p(\boldsymbol{x}(\boldsymbol{X}, s)) \circ d B^p(s) +  \int_0^t \nu d W(s) ; \quad \boldsymbol{x}(\boldsymbol{X}, 0)=\boldsymbol{X}
$$

### Numerical Approach:

### Code History: 
This code is based upon: lecture notes and example code provided by JOHN METHVEN in a summer school. Then extended for different notions of integration for the paper: Lévy areas, Wong Zakai anomalies in diffusive limits of Deterministic Lagrangian Multi-Time Dynamics. Then extended into jax, for more points using a different temporal integrator. 


# Kernel convergence in the deterministic setting,
[7] = J Thomas Beale and Andrew Majda. High order accurate vortex methods with explicit velocity kernels. Journal of Computational Physics, 58(2):188–208, 1985


This scheme (in the deterministic setting) has been shown to have the property that if $\delta=h^q$ for $q \in(0,1)$, the order of convergence to the solution of the Euler Equation is given by $O\left(h^{(2 p+2) q}\right)$ see [7].


# Stochastic scheme temporal consistency,

To deal with the stochastic Stratonovich term, we discretise in time with the stochastic generalisation of the SSP33 scheme of Shu and Osher, where the forward Euler scheme is replaced with Euler Maruyama scheme in the Shu Osher representation. This time-stepping is weak order 1, strong order 0.5, as can be found by Taylor expanding or as a subcase of the work by Ruemelin[40].


[40] = W Rüemelin. Numerical treatment of stochastic differential equations. SIAM Journal on Numerical Analysis,
19(3):604–613, 1982.

