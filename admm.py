
from updates.LocalUpdate_ACOPF import LocalUpdate_ACOPF
from updates.GlobalUpdate_ACOPF import GlobalUpdate_ACOPF
from ipopt2 import ipopt
import jax.numpy as jnp


def ADMM(net,regions,x_1,x_2,alpha,xbar,G,B,S,bnds1,bnds2,rho=5000, max_iter = 10,tol = 1e-4):

    residual_list = []

    for _ in range(max_iter):
        #Local Date
        local_update_r1 = LocalUpdate_ACOPF(net,regions,1,G,B,S,xbar,rho,alpha,0) 
        local_update_r2 = LocalUpdate_ACOPF(net,regions,2,G,B,S,xbar,rho,alpha,2)
        x_1,x_2 = ipopt(local_update_r1.objective,local_update_r1.eq_constraints,local_update_r1.ineq_constraints,x_1,bnds1),ipopt(local_update_r2.objective,local_update_r2.eq_constraints,local_update_r2.ineq_constraints,x_2,bnds2)
        #Global Update
        x_r_list = [x_1,x_2]
        global_update = GlobalUpdate_ACOPF(net,regions,rho,x_r_list,alpha)
        xbar = global_update.objective()

        #update dual variable
        region = 0
        idx_bar_r = 0
        alpha_k = []
        residual = 0
        for idx,x_r in enumerate(x_r_list):
            region = idx +  1
            x_int = jnp.array(list(regions[region][0]))
            x_bound = jnp.array(list(regions[region][1]))
            #alpha
            alpha_e_r = alpha[:len(alpha) // 2][idx_bar_r: idx_bar_r + len(x_bound)]
            alpha_j_r = alpha[len(alpha) // 2:][idx_bar_r:idx_bar_r + len(x_bound)]
            alpha_r = jnp.concatenate([alpha_e_r,alpha_j_r])
            #print(alpha_e_r)
            #xbar
            xbar_e_r = xbar[:len(xbar) // 2][idx_bar_r: idx_bar_r + len(x_bound)]
            xbar_j_r = xbar[len(xbar) // 2:][idx_bar_r:idx_bar_r + len(x_bound)]
            xbar_r = jnp.concatenate([xbar_e_r,xbar_j_r])
            #print(xbar_e_r)

            #Ax_jr
            Ax_jr = x_r[4 * len(x_int):]
            #new alpha
            alpha_r += rho * (Ax_jr - xbar_r)

            alpha_k.append(alpha_r)

            #Residual
            residual += jnp.linalg.norm(Ax_jr - xbar_r)

            idx_bar_r += x_bound
                


        alpha_k = jnp.concatenate([alpha_k[0],alpha_k[1]])
        print(residual)
        residual_list.append(residual ** 2)

    

    return x_1,x_2,xbar,residual_list
