import jax.numpy as jnp
import numpy as np

class AlphaUpdate_ACOPF:

    def __init__(self,alpha,lb,up):
        #Transform each data structure of the net into jax numpy
        self.alpha = alpha
        self.lb = lb
        self.ub = up
        

    def objective(self,x):

        return jnp.linalg.norm(x - self.alpha)

    def eq_constraints(self,x):
        return jnp.array([0])

    def ineq_constraints(self,x):
        return jnp.array([0])
         






