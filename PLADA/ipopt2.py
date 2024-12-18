from jax import config
import jax.numpy as jnp
from jax import jit,grad,jacfwd,jacrev
from cyipopt import minimize_ipopt
#Enable 64 bit floating point precision
config.update("jax_enable_x64",True)
#Use the CPU instead of GPU and mute all warnings if no GPU/TPU is found
#config.update("jax_platform_name",'cpu')


def ipopt_x(objective,con_eq,con_ineq,x0,bnds):

    """
    Algorithmic Differentiation
    """
    #jit (just-in-time) functions

    obj_jit = jit(objective)
    con_eq_jit = jit(con_eq)
    con_ineq_jit = jit(con_ineq)

    #build the derivatives and jit them

    obj_grad = jit(grad(obj_jit))  # objective gradient
    obj_hess = jit(jacrev(jacfwd(obj_jit))) # objective hessian
    con_eq_jac = jit(jacfwd(con_eq_jit))  # jacobian
    con_ineq_jac = jit(jacfwd(con_ineq_jit))  # jacobian
    con_eq_hess = jacrev(jacfwd(con_eq_jit)) # hessian
    con_ineq_hess = jacrev(jacfwd(con_ineq_jit))  # hessian
    #con_eq_hessvp = jit(lambda x, v: con_eq_hess(x) * v[0]) # hessian vector-product
    #con_ineq_hessvp = jit(lambda x, v: con_ineq_hess(x) * v[0]) # hessian vector-product
    
    def con_eq_hessvp(x, v):
        H = con_eq_hess(x)  # H has shape (m, n, n)
        # Compute the weighted sum of the Hessians
        Hv = jnp.tensordot(v, H, axes=1)  # Sum over m constraints
        return Hv  # Returns an array of shape (n, n)
            

    # JIT compile the function
    con_eq_hessvp = jit(con_eq_hessvp)


    def con_ineq_hessvp(x, v):
        H = con_ineq_hess(x)  # H has shape (m, n, n)
        # Compute the weighted sum of the Hessians
        Hv = jnp.tensordot(v, H, axes=1)  # Sum over m constraints
        return Hv  # Returns an array of shape (n, n)
    # JIT compile the function
    con_ineq_hessvp = jit(con_ineq_hessvp)
    
            


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
    if bnds:
        res = minimize_ipopt(obj_jit,jac=obj_grad,hess=obj_hess,x0=x0,constraints=cons,bounds=bnds,options={
            'disp': True,
            'hessian_approximation': 'exact',
            'constr_viol_tol': 1e-6,
            'obj_scaling_factor': 1e8,
            'mu_strategy': 'adaptive',
            'max_iter': 3000,
        })
    
    
    x_r_k = res['x']



    return res


def ipopt_u(objective,con_eq,u0,bnds):

    config.update("jax_enable_x64",True)

    obj_jit = jit(objective)
    obj_grad = jit(grad(obj_jit))  # objective gradient
    obj_hess = jit(jacrev(jacfwd(obj_jit))) # objective hessian

    '''    
    con_eq_jit = jit(con_eq)
    con_eq_jac = jit(jacfwd(con_eq_jit))  # jacobian
    con_eq_hess = jacrev(jacfwd(con_eq_jit)) # hessian

    def con_eq_hessvp(x, v):
        H = con_eq_hess(x)  # H has shape (m, n, n)
        # Compute the weighted sum of the Hessians
        Hv = jnp.tensordot(v, H, axes=1)  # Sum over m constraints
        return Hv  # Returns an array of shape (n, n)
    
        #constraints
    cons = [
            {
                "type": 'eq',
                'fun': con_eq_jit,
                'jac':con_eq_jac,
                'hess': con_eq_hessvp
            }
    ]
   
    '''


    res = minimize_ipopt(obj_jit,jac=obj_grad,hess=obj_hess,x0=u0,options={
            "tol": 1e-8,
            "check_derivatives_for_naninf": "yes",                           # Tighten convergence tolerance for better accuracy
            "hessian_approximation": 'limited-memory',
            "max_iter": 500,                      # Increase max iterations to allow more time for convergence
            "mu_strategy": "adaptive",             # Use adaptive barrier parameter strategy for stability

    })
    return res
