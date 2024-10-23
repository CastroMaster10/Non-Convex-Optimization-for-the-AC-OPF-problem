
from updates.LocalUpdate_ACOPF import LocalUpdate_ACOPF
from updates.GlobalUpdate_ACOPF import GlobalUpdate_ACOPF
from ipopt2 import ipopt
import jax.numpy as jnp


def ADMM_ACOPF(net,regions,G,B,S,idx_buses_arr,alpha,x_r_arr,xbar,rho,bnds_arr,max_iter):

    
    infi_arr = []
    gcost_arr = []


    for _ in range(max_iter):
        #Local update
        xnew_r_arr = []
        #objective_f_new = 0
        eq_cons_violation = []
        ineq_cons_violation = []

        for idx,x_r in enumerate(x_r_arr):

            region = idx + 1
            idx_buses_before_regioni,idx_buses_after_regioni = idx_buses_arr[region]
            bnds_r = bnds_arr[idx]  

            local_update_i = LocalUpdate_ACOPF(net,regions,region,G,B,S,xbar,rho,alpha,idx_buses_before_regioni,idx_buses_after_regioni)
            x_r_k = ipopt(local_update_i.objective,local_update_i.eq_constraints,local_update_i.ineq_constraints,x_r,bnds_r)
            #objective_f_new += local_update_i.objective(x_r_k)
            eq_cons_violation.append(jnp.sum(local_update_i.eq_constraints(x_r_k)))
            ineq_cons_violation.append((jnp.where(local_update_i.ineq_constraints(x_r_k) >= 0)[0].size,local_update_i.ineq_constraints(x_r_k).size))
            xnew_r_arr.append(x_r_k)

   
        #Global update
        global_update_i = GlobalUpdate_ACOPF(net,regions,rho,xnew_r_arr,alpha,idx_buses_arr)
        #new_xbar = global_update.update_xbar(xbar)
        #new_xbar = ipopt(global_update_i.objective,global_update_i.eq_constraints,global_update_i.ineq_constraints,xbar,[])
        new_xbar = global_update_i.global_update()
        alpha_new = alpha
        Ar_xr_arr = []
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
            Ar_xr_arr.append(Ar_xr_k)
        
        Ar_xr_arr_new = jnp.sum(jnp.array(Ar_xr_arr),axis=0)
        alpha_new +=  rho * (Ar_xr_arr_new -  new_xbar)

        #Generation cost
        generation_cost = 0
        for idx,x_r in enumerate(xnew_r_arr):
            region = idx + 1
            #generation_cost_i = generation_cost(x_r,net,regions,region)
            #generation_cost += generation_cost_i
            x_int = jnp.array(list(regions[region][0])) #Interior buses
            x_bound = jnp.array(list(regions[region][1])) #Boundary buses
            
                            
            pg = x_r[:len(x_int)]
            gencost_data_r = net['gencost'][x_int, :][:,4:]

            a = gencost_data_r[:,0]
            b = gencost_data_r[:,1]
            c = gencost_data_r[:,2]
                    
            #c_r(x)
            total_c = 0
            for i in range(len(x_int)):
                total_c += a[i] *  pg[i] ** 2 + b[i] * pg[i] + c[i]
            
            generation_cost += total_c


        print(f'N. Iteration: {_}')
        print('\nConstraints violation for each region')
        for idx in range(len(eq_cons_violation)):
            print(f'Region {idx + 1}\n \t-Equality constraints violation: {eq_cons_violation[idx]} \n \t-Inequality constraints violation: {ineq_cons_violation[idx][0]} / {ineq_cons_violation[idx][1]}')
        print("\nInfeasibility ||Ax + Bx||: ",jnp.linalg.norm(Ar_xr_arr_new - new_xbar))
        print(f'\nGeneretation cost: {generation_cost}')
        print('\n----------------------------------------------------------------------')
        
        x_r_arr = xnew_r_arr
        xbar = new_xbar
        alpha = alpha_new

        #Add values
        infi_arr.append(jnp.linalg.norm(Ar_xr_arr_new - new_xbar))
        gcost_arr.append(generation_cost)



    return  {
        "x_r":xnew_r_arr, 
        "xbar":new_xbar, 
        "alpha":alpha,
        "infeasibility_arr": infi_arr,
        "generation_cost": gcost_arr
        }



