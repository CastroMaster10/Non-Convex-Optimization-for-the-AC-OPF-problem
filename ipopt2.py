from jax import config
import jax.numpy as jnp
from jax import jit,grad,jacfwd,jacrev
from cyipopt import minimize_ipopt


#Enable 64 bit floating point precision
config.update("jax_enable_x64",True)
#Use the CPU instead of GPU and mute all warnings if no GPU/TPU is found
config.update("jax_platform_name",'cpu')


def ipopt(objective,con_eq_jit,con_ineq_jit,x0,bnds):

    """
    Algorithmic Differentiation
    """
    #jit (just-in-time) functions

    obj_jit = jit(objective)
    con_eq_jit = jit(con_eq_jit)
    con_ineq_jit = jit(con_ineq_jit)

    #build the derivatives and jit them

    obj_grad = jit(grad(obj_jit))  # objective gradient
    obj_hess = jit(jacrev(jacfwd(obj_jit))) # objective hessian
    con_eq_jac = jit(jacfwd(con_eq_jit))  # jacobian
    con_ineq_jac = jit(jacfwd(con_ineq_jit))  # jacobian
    con_eq_hess = jacrev(jacfwd(con_eq_jit)) # hessian
    con_eq_hessvp = jit(lambda x, v: con_eq_hess(x) * v[0]) # hessian vector-product
    con_ineq_hess = jacrev(jacfwd(con_ineq_jit))  # hessian
    con_ineq_hessvp = jit(lambda x, v: con_ineq_hess(x) * v[0]) # hessian vector-product

    #constraints
    cons = [
        {
            "type": 'eq',
            'fun': con_eq_jit,
            'jac':con_eq_jac,
            'hess': con_eq_hessvp
        },
        {
            "type": 'ineq',
            'fun': con_ineq_jit,
            'jac':con_ineq_jac,
            'hess': con_ineq_hessvp  
        }
    ]

    #Executing the solver
    
    #Executing the solver
    res = minimize_ipopt(obj_jit,jac=obj_grad,hess=obj_hess,x0=x0,constraints=cons,bounds=bnds,options={
            'disp':5,
            'tol': 1e-6,
            'max_iter': 5000,
            'linear_solver': 'mumps',
            #'hessian_approximation': 'limited-memory',
            'mu_init': 1e-4,
            'constr_viol_tol': 1e-50,
            'obj_scaling_factor': 1e-10,
            'nlp_scaling_method': 'gradient-based',
            'mu_strategy': 'adaptive',
            'acceptable_tol': 1e-1,
            'acceptable_iter': 5 
        })
    
    
    x_r_k = res['x']



    return x_r_k