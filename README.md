
### Point vortex library
- <strong>Autodifferentiable<strong>
- <strong>CPU-GPU tested -jax[cpu] - jax[cuda12]<strong>

# Example one solves: 
$$\boldsymbol{x}(\boldsymbol{X}, t)=\boldsymbol{x}(\boldsymbol{X}, 0)+\int_0^t \boldsymbol{u}(\boldsymbol{x}(\boldsymbol{X}, s), s) d s+\sum_{p=1}^P \int_0^t \theta_p \boldsymbol{\xi}_p(\boldsymbol{x}(\boldsymbol{X}, s)) \circ d B^p(s) ; \quad \boldsymbol{x}(\boldsymbol{X}, 0)=\boldsymbol{X}
$$

# Examples two solves: 
$$\boldsymbol{x}(\boldsymbol{X}, t)=\boldsymbol{x}(\boldsymbol{X}, 0)+\int_0^t \boldsymbol{u}(\boldsymbol{x}(\boldsymbol{X}, s), s) d s+\sum_{p=1}^P \int_0^t \theta_p \boldsymbol{\xi}_p(\boldsymbol{x}(\boldsymbol{X}, s)) \circ d B^p(s) +  \int_0^t \nu d W(s) ; \quad \boldsymbol{x}(\boldsymbol{X}, 0)=\boldsymbol{X}
$$

# references 
This code is a usable subset of the code developed for the paper,

###https://arxiv.org/pdf/2405.00640

@article{woodfield2024stochastic,
  title={Stochastic fluids with transport noise: Approximating diffusion from data using SVD and ensemble forecast back-propagation},
  author={Woodfield, James},
  journal={arXiv preprint arXiv:2405.00640},
  year={2024}
}

