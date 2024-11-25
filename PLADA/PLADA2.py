import jax.numpy as jnp
import numpy as np
from jax import jit,grad,jacfwd,jacrev


def PLADA(regions,x_r_arr0,bnds_xr_arr,xbar0,max_iter):

    k = 0
    
    x_r_arr = x_r_arr0
    #xj_rl = get_xj_rl(x_r_arr,regions)
    bnds_arr = bnds_xr_arr
    xbar = xbar0

    while k < max_iter:
        
            #Local update for each agent r
            for idx,x_r in enumerate(x_r_arr):
                
                region = idx + 1
                boundBuses_idx = regions[region][1].union(regions[region][2])


                bnds_r = bnds_arr[idx]
                xbar_r_dict = {k:v for k,v in xbar0.items() if k in boundBuses_idx}
                xbar_r = jnp.array(list(xbar_r_dict.values()))

                local_update =  LocalUpdate_ACOPF(net,regions,region,G,B,S,xbar_r,rho,y_r,z_r)
                #Implement Interior Point Method Solver to solve Local Problem from region R
                res_local = ipopt(local_update.objective,local_update.eq_constraints,local_update.ineq_constraints,x_r,bnds_r)
                x_r_new = res_local['x']
                obj_arr.append(res_local['fun'])
                solver_messages.append(res_local['message'])
                eq_cons_violation.append(local_update.eq_constraints(x_r_new))
                ineq_cons_violation.append(local_update.ineq_constraints(x_r_new)[jnp.where(local_update.ineq_constraints(x_r_new) >  1e-4)])


                #Add new values from x_r
                x_r_new_arr.append(x_r_new)


        print("Iteration {k}")
        k += 1
    

    return 0
