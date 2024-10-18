
from updates.LocalUpdate_ACOPF import LocalUpdate_ACOPF
from updates.GlobalUpdate_ACOPF import GlobalUpdate_ACOPF
from ipopt2 import ipopt
import jax.numpy as jnp


def ADMM_ACOPF(net,regions,G,B,S,idx_buses_arr,alpha_arr,x_r_arr,xbar,rho,bnds_arr,max_iter):


    for _ in range(max_iter):
        #Local update
        xnew_r_arr = []
        objective_f_before = 0
        objective_f_new = 0
        for idx,x_r in enumerate(x_r_arr):

            region = idx + 1
            idx_buses_before_regioni,idx_buses_after_regioni = idx_buses_arr[region]
            alpha_r = alpha_arr[idx]
            bnds_r = bnds_arr[idx]  

            local_update_i = LocalUpdate_ACOPF(net,regions,region,G,B,S,xbar,rho,alpha_r,idx_buses_before_regioni,idx_buses_after_regioni)
            objective_f_before += local_update_i.objective(x_r)
            x_r_k = ipopt(local_update_i.objective,local_update_i.eq_constraints,local_update_i.ineq_constraints,x_r,bnds_r)
            objective_f_new += local_update_i.objective(x_r_k)
            xnew_r_arr.append(x_r_k)

        print("\nObjective function before local update: ",objective_f_before)
        print("Objective function after local update: ",objective_f_new)
        print("\n---------------------------------------------------------")
        #Global update
        global_update = GlobalUpdate_ACOPF(net,regions,rho,xnew_r_arr,alpha_arr,idx_buses_arr)
        new_xbar = global_update.update_xbar(xbar)
        alpha_new_arr = []

        #Update Dual variable
        for idx,x_r in enumerate(xnew_r_arr):
            region = idx + 1
            idx_buses_before_regioni,idx_buses_after_regioni = idx_buses_arr[region]

            x_int = jnp.array(list(regions[region][0]))
            x_bound = jnp.array(list(regions[region][1]))

            X_bound = x_r[len(x_int) * 4:].reshape((2,-1))

            
            X_bound_v = X_bound.reshape(-1)
            X_bound_v_e = X_bound_v[:len(x_bound)]
            X_bound_v_f = X_bound_v[len(x_bound):]

            Ar_xr_e = jnp.concatenate([jnp.zeros(idx_buses_before_regioni),X_bound_v_e,jnp.zeros(idx_buses_after_regioni)])
            Ar_xr_f = jnp.concatenate([jnp.zeros(idx_buses_before_regioni),X_bound_v_f,jnp.zeros(idx_buses_after_regioni)])

            Ar_xr_k = jnp.concatenate([Ar_xr_e,Ar_xr_f])
            alpha_new_arr.append(alpha_arr[idx] + rho * (Ar_xr_k -  new_xbar))
        
        
        x_r_arr = xnew_r_arr
        xbar = new_xbar
        alpha_arr = jnp.array(alpha_new_arr)


    



    return  xnew_r_arr, new_xbar, alpha_arr