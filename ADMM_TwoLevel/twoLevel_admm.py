
import jax.numpy as jnp
import numpy as np
from .ipopt2 import ipopt
from .LocalUpdate_twoLevel_ACOPF import LocalUpdate_ACOPF



def get_xj_arr(x_r_arr,regions,idx_buses_arr):
    Ar_xr_arr = []

    Ar_xr_arr_int_e = []
    Ar_xr_arr_int_f = []
    
    for idx,x_r in enumerate(x_r_arr):
        region  = idx + 1
        idx_buses_before_i =  idx_buses_arr[region][0]
        idx_buses_after_i =  idx_buses_arr[region][1]

        x_int = jnp.array(list(regions[region][0]))
        x_bound = jnp.array(list(regions[region][1]))
        
        X_int = x_r[:len(x_int) * 4].reshape((4,-1))
        X_bound = x_r[len(x_int) * 4:].reshape((2,-1))


        #Reshape and select
        X_int_e =  X_int[2, :] 
        X_int_f =  X_int[3, :]

        Ar_xr_arr_int_e.append(X_int_e)
        Ar_xr_arr_int_f.append(X_int_f)


        
        #Calculate A_rx_r to perform operations when updating dual variables and slack variables
        X_bound_v = X_bound.reshape(-1)
        X_bound_v_e = X_bound_v[:len(x_bound)]
        X_bound_v_f = X_bound_v[len(x_bound):]
        Ar_xr_e = jnp.concatenate([jnp.zeros(idx_buses_before_i),X_bound_v_e,jnp.zeros(idx_buses_after_i)])
        Ar_xr_f = jnp.concatenate([jnp.zeros(idx_buses_before_i),X_bound_v_f,jnp.zeros(idx_buses_after_i)])
        Ar_xr = jnp.concatenate([Ar_xr_e,Ar_xr_f])
        Ar_xr_arr.append(Ar_xr)
    

    Ar_xr_arr_int_e = jnp.concatenate(Ar_xr_arr_int_e,axis=0)
    Ar_xr_arr_int_f = jnp.concatenate(Ar_xr_arr_int_f,axis=0)

    int_values_e = []
    int_values_f = []
 
    for idx,x_r in enumerate(x_r_arr):
        region = idx + 1
        x_bound = jnp.array(list(regions[region][1]))

        int_values_e.append(Ar_xr_arr_int_e[x_bound])
        int_values_f.append(Ar_xr_arr_int_f[x_bound])

    

    int_values_e = jnp.concatenate(int_values_e,axis=0)
    int_values_f = jnp.concatenate(int_values_f,axis=0)

        




    return jnp.sum(jnp.array(Ar_xr_arr),axis=0),jnp.concatenate([int_values_e,int_values_f])


def objective(x_r_arr,net,regions):
    "Local Objective function"
    total_cost = 0
    for idx,x_r in enumerate(x_r_arr):
        region = idx + 1
        x_int = jnp.array(list(regions[region][0])) #Interior buses
        x_bound = jnp.array(list(regions[region][1])) #Boundary buses
        
                        
        pg = x_r[:len(x_int)]
        gencost_data_r = net['gencost'][x_int, :][:,4:]

        a = gencost_data_r[:,0]
        b = gencost_data_r[:,1]
        c = gencost_data_r[:,2]
                
        #c_r(x)
        total_c_i = 0
        for i in range(len(x_int)):
            total_c_i += a[i] *  pg[i] ** 2 + b[i] * pg[i] + c[i]

            
        total_cost += total_c_i
    
    return total_cost


def TwoLevel_ADMM_ACOPF(net,regions,G,B,S,idx_buses_arr,d,beta,alpha,x_r_arr0,bnds_xr_arr,xbar0,xbar_bnds,c,lmbda):
    

    """
    Outer Loop
    """

    #Initialize values for Outer Loop
    k = 1
    theta = 0.8
    tol_outer = 1e-9

    x_r_arr = x_r_arr0
    xj_arr,int_values = get_xj_arr(x_r_arr,regions,idx_buses_arr)

    bnds_arr =bnds_xr_arr

    y = jnp.zeros(d) #Dual variable
    z = -alpha / beta #Slack Variable

    xbar = xbar0
    gcost_arr = [objective(x_r_arr,net,regions)]
    infeasibility_arr = [jnp.linalg.norm(xj_arr - xbar)]
    z_arr = [jnp.linalg.norm(z)]
    print("Initial values and constraint violations for x_r0")
    
    for idx,x_r in enumerate(x_r_arr0):
        rho = 2 * beta
        region = idx + 1
        idx_buses_before_i =  idx_buses_arr[region][0]
        idx_buses_after_i =  idx_buses_arr[region][1]
        bnds_r = bnds_arr[idx]
                

        local_update =  LocalUpdate_ACOPF(net,regions,region,G,B,S,xbar,rho,y,idx_buses_before_i,idx_buses_after_i,z)

        ineq_viol = local_update.eq_constraints(x_r)[jnp.where(local_update.ineq_constraints(x_r) > 1e-4)[0]]

        print(f'Region {region}\n \t-Equality constraints violation: {jnp.sum(jnp.abs(local_update.eq_constraints(x_r)))}')
        print(f'Inequality constraints violation: {jnp.sum(jnp.abs(ineq_viol))}')
    
    print(f'\n Total Generation cost: {objective(x_r_arr0,net,regions)}')

    while jnp.linalg.norm(xj_arr - xbar) >=  (jnp.sqrt(d) * 1e-5):

        """
        Inner Loop
        """
        #Initialize values for Inner Loop
        if k > 20:
            break

        rho = 2 * beta
        z = (-alpha -  y) / beta
        #y = -alpha - beta * z



        t = 1
        print("\n||Ax_r + Bx + z || = ",jnp.linalg.norm(xj_arr - xbar + z))
        print("Threshold: ",(jnp.sqrt(d) / 2500 * k))
        print("|| z ||: ",(jnp.linalg.norm(z)))
        while (float(jnp.linalg.norm(xj_arr - xbar + z)) >= float(jnp.sqrt(d) * 2 / 2500 * k)):
            if t >= 50:
                break

            x_r_new_arr = []

            #Keep track of local constraints violations
            eq_cons_violation = []
            ineq_cons_violation = []

            #Local update for each agent r
            for idx,x_r in enumerate(x_r_arr):
                region = idx + 1
                idx_buses_before_i =  idx_buses_arr[region][0]
                idx_buses_after_i =  idx_buses_arr[region][1]
                bnds_r = bnds_arr[idx]
                

                local_update =  LocalUpdate_ACOPF(net,regions,region,G,B,S,xbar,rho,y,idx_buses_before_i,idx_buses_after_i,z)
                #Implement Interior Point Method Solver to solve Local Problem from region R
                res_local = ipopt(local_update.objective,local_update.eq_constraints,local_update.ineq_constraints,x_r,bnds_r)
                x_r_new = res_local['x']
                eq_cons_violation.append(local_update.eq_constraints(x_r_new))
                ineq_cons_violation.append(local_update.ineq_constraints(x_r_new)[jnp.where(local_update.ineq_constraints(x_r_new) >  1e-4)])


                #Add new values from x_r
                x_r_new_arr.append(x_r_new)
            

            xr_j_new,int_values = get_xj_arr(x_r_new_arr,regions,idx_buses_arr)

            int_values_e = int_values[:d//2]
            int_values_f = int_values[d//2:]
            #Global update
            #global_update = GlobalUpdate_ACOPF(net,regions,rho,y,idx_buses_arr,z,x_r_new_arr)
            #xbar_new = ipopt(global_update.global_update,global_update.eq_constraints,global_update.ineq_constraints,xbar,xbar_bnds)['x']
            xbar_new_proj_e =  (xr_j_new[:d//2] + int_values_e +  2 * z[:d//2]) / 2 + 2 * y[:d//2] / (2 * rho) 
            xbar_new_proj_f =   (xr_j_new[d//2:] + int_values_f +   2 * z[d//2:]) / 2 + 2 * y[d//2:] / (2 * rho)
            xbar_new_e = jnp.clip(xbar_new_proj_e,0.8140638470649719,1.0599999999)
            xbar_new_f = jnp.clip(xbar_new_proj_f,-0.4699999988079071,0.52999)

            xbar_new = jnp.concatenate([xbar_new_e,xbar_new_f])
            

            #Slack Variable update
            z_new = (-alpha - y - rho * (xr_j_new - xbar_new)) / (beta + rho)
            #z_new = z + xr_j_new - xbar_new 
            #z_new = (-alpha -y -    rho * (xr_j_new - xbar_new)) / (beta + rho) 

            #Dual variable update
            y_new = y +  rho * (xr_j_new - xbar_new + z_new)

            #adapt rho
            if jnp.linalg.norm(xr_j_new - xbar_new + z_new) > theta * jnp.linalg.norm(xj_arr - xbar + z):
                rho = lmbda * rho
            


            x_r_arr = x_r_new_arr
            xj_arr = xr_j_new
            xbar = xbar_new
            y = y_new

            #Check if slack variable hasn't change          
            if jnp.linalg.norm(z_new - z) <= 1e-8:
                print("\nZ is not changing...") 
                z = z_new
                break
            else:
                z = z_new

    
            

            
            print(f'\nN. Iteration for Inner Loop: {t}')
            #print(f'\nTotal Generation cost: {objective(x_r_arr,net,regions)}')
            #print('\nConstraints violation for each region')
            #for idx in range(len(eq_cons_violation)):
            #    print(f'Region {idx + 1}\n \t-Equality constraints violation: {jnp.sum(jnp.abs(eq_cons_violation[idx]))}')
            #    print(f'\t-Ineq constraint violation: {jnp.sum(ineq_cons_violation[idx])}')
            
            #print(f'\nint values: {int_values}')
            print(f'\n|| z ||: {jnp.linalg.norm(z)}')
            print(f'\n|| Axr + Bx ||: {jnp.linalg.norm(xj_arr - xbar)}')
            #print(f'\nGlobal infeasibility with z || Axr + Bx + + z||: {jnp.linalg.norm(xj_arr - xbar + z)}')
           #print("*******************************************************************")
            t += 1

        

        
        print(f'\n\nN. Iteration for Outer Loop: {k}')
        print(f'Number of iterationns for inner loop: {t}')
        print('\nConstraints violation for each region')
        for idx in range(len(eq_cons_violation)):
            print(f'Region {idx + 1}\n \t-Equality constraints violation: {jnp.sum(jnp.abs(eq_cons_violation[idx]))}')
            print(f'\t-Ineq constraint violation: {jnp.sum(ineq_cons_violation[idx])}')
        print("\n Global Infeasibility without z ||Ax^k + Bx^k||: ",jnp.linalg.norm(xr_j_new - xbar_new))
        print(f'\nGeneretation cost: {objective(x_r_new_arr,net,regions)}')
        print("|| z ||Norm of Slack variable: ",jnp.linalg.norm(z))
        #Update outer dual variables
        print(f'\nUpdating Alpha...')
        alpha = np.clip(alpha + beta * z,-1e12,1e12)
        #y = jnp.zeros(d)


        if jnp.linalg.norm(z) <= theta * jnp.linalg.norm(z_arr[-1]):
            beta = c * beta
            print("Updating beta = ",beta)
        
        z_arr.append(jnp.linalg.norm(z))
        print('\n----------------------------------------------------------------------')
        
        

        infeasibility_arr.append(jnp.linalg.norm(xj_arr - xbar))
        gcost_arr.append(objective(x_r_arr,net,regions))
        

        k += 1


        
    return {
        "x_r": x_r_arr,
        "xbar": xbar,
        "z": z,
        "infeasibility_arr": infeasibility_arr,
        "generation_cost_arr": gcost_arr
        
    }









