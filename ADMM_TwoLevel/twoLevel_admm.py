
import jax.numpy as jnp
import numpy as np
from .ipopt2 import ipopt
from .LocalUpdate_twoLevel_ACOPF import LocalUpdate_ACOPF
import time






def TwoLevel_ADMM_ACOPF(net,regions,G,B,S,d,beta,alpha,x_r_arr0,bnds_xr_arr,xbar0,y_arr0,z_arr0,c,lmbda,max_iter_outer, max_iter_inner):
    
    start_time = time.time()


    #Initialize values for Outer Loop
    k = 1
    theta = 0.95
    tol_outer = 1e-9

    x_r_arr = x_r_arr0
    xj_rl = get_xj_rl(x_r_arr,regions)
    bnds_arr = bnds_xr_arr

    xbar = xbar0

    y_arr = y_arr0 #Dual variable
    z_arr = z_arr0 #Slack variable

    gcost_arr = [objective(x_r_arr,net,regions)]
    infeasibility_arr = [jnp.linalg.norm(Ax_xbar_difference(xj_rl,xbar))]
    z_values = [jnp.linalg.norm(jnp.array(list(z_arr.values())))]
    iteration_times = []
    """    
    print("Initial values and constraint violations for x_r0")
    for idx,x_r in enumerate(x_r_arr0):
        rho = 2 * beta
        region = idx + 1
        bnds_r = bnds_arr[idx]


        local_update =  LocalUpdate_ACOPF(net,regions,region,G,B,S,xbar,rho,y,z)

        ineq_viol = local_update.ineq_constraints(x_r)[jnp.where(local_update.ineq_constraints(x_r) > 1e-4)[0]]

        print(f'Region {region}\n \t-Equality constraints violation: {jnp.sum(jnp.abs(local_update.eq_constraints(x_r)))}')
        print(f'Inequality constraints violation: {jnp.sum(jnp.abs(ineq_viol))}')
    """
    
    print(f'\n Total Generation cost: {objective(x_r_arr,net,regions)}')
    print(f'\n|| Axr + Bx ||: {jnp.linalg.norm(Ax_xbar_difference(xj_rl,xbar))}')
    
    end_time = time.time()
    iteration_time = end_time - start_time
    iteration_times.append(iteration_time)


    while jnp.linalg.norm(Ax_xbar_difference(xj_rl,xbar)) >=  (jnp.sqrt(d) * 1e-5):
        start_time = time.time()

        """
        Outer Loop
        """

        if k > max_iter_outer:
            break

        rho = 2 * beta
        #y = -alpha - beta * z

        y_values = jnp.array(list(y_arr.values()))
        z_new_values = (-alpha - y_values) / beta #z variable = 0

        z_arr = {key: new_value for key, new_value in zip(z_arr.keys(), z_new_values)}

        print("\n||Ax_r + Bx + z || = ",jnp.linalg.norm(Ax_xbar_z_difference(xj_rl,xbar,z_arr)))
        print("Threshold: ",(jnp.sqrt(d) / 2500 * k))
        print(f'|| z ||: {jnp.linalg.norm(z_new_values)}')
        t = 1
        while (float(jnp.linalg.norm(Ax_xbar_z_difference(xj_rl,xbar,z_arr))) >= float(jnp.sqrt(d) / 2500 * k)):

            """
            Inner Loop
            """

            if t > max_iter_inner:
                break
  

            x_r_new_arr = []

            #Keep track of local constraints violations
            eq_cons_violation = []
            ineq_cons_violation = []
            solver_messages =  []
            obj_arr = []


            #Local update for each agent r
            for idx,x_r in enumerate(x_r_arr):
                
                region = idx + 1
                boundBuses_idx = regions[region][1].union(regions[region][2])

                y_r_dict = {key:value for key, value in y_arr.items() if key.endswith(f'_{region}')}
                z_r_dict = {key:value for key, value in z_arr.items() if key.endswith(f'_{region}')}

                y_r = jnp.array(list(y_r_dict.values()))
                z_r = jnp.array(list(z_r_dict.values()))


                bnds_r = bnds_arr[idx]
                xbar_r_dict = {k:v for k,v in xbar.items() if k in boundBuses_idx}
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


            xj_rl_new = get_xj_rl(x_r_new_arr,regions)
            xbar_new = xbar_update(xj_rl_new,xbar,z_arr,y_arr,rho)            

            #Slack Variable update
            Ax_Bx_diff_new = Ax_xbar_difference(xj_rl_new,xbar_new)
            z_new_values = (-alpha - jnp.array(list(y_arr.values())) - rho * (Ax_Bx_diff_new)) / (beta + rho)
            z_arr_new = {key: new_value for key, new_value in zip(z_arr.keys(), z_new_values)}

            Ax_Bx_z_diff_new = Ax_xbar_z_difference(xj_rl_new,xbar_new,z_arr_new)
            #Dual variable update
            y_new_values = jnp.array(list(y_arr.values())) +  rho * (Ax_Bx_z_diff_new)
            y_arr_new = {key: new_value for key, new_value in zip(y_arr.keys(), y_new_values)}

            #update rho
                        
            if jnp.linalg.norm(Ax_xbar_z_difference(xj_rl_new,xbar_new,z_arr_new)) > theta * jnp.linalg.norm(Ax_xbar_z_difference(xj_rl,xbar,z_arr)):
                rho = lmbda * rho
                print(f'Updating rho={rho}')
             
            

            


            x_r_arr = x_r_new_arr
            xj_rl = xj_rl_new
            xbar = xbar_new
            y_arr = y_arr_new

            if jnp.linalg.norm(jnp.array(list(z_arr_new.values())) - jnp.array(list(z_arr.values()))) <= 1e-8:
                print("\nZ is not changing...") 
                z_arr = z_arr_new
                break
            else:
                z_arr = z_arr_new

            
            print(f'\nN. Iteration for Inner Loop: {t}')
            #print(f'Local constraint violation: {sum(objective_p)}')
            #print(f'\nTotal Generation cost: {objective(x_r_arr,net,regions)}')
            #print('\nConstraints violation for each region')
            for idx in range(len(eq_cons_violation)):
            #    print(f'Region {idx + 1}\n \t-Equality constraints violation: {jnp.sum(jnp.abs(eq_cons_violation[idx]))}')
            #    print(f'\nRegion {idx+1}\n x_r: {x_r_arr[idx]}')
            #    print(f'\t-Ineq constraint violation: {jnp.sum(ineq_cons_violation[idx])}')
            #    print(f'\n\Region {idx+ 1}\n\t-Message : {solver_messages[idx]}')
                print(f'\t-Objective Function (with penalty) : {obj_arr[idx]}')
     
            print(f'\n|| Axr + Bx + z||: {jnp.linalg.norm(Ax_xbar_z_difference(xj_rl,xbar,z_arr))}')
            #print(f'\n|| Axr + Bx ||: {jnp.linalg.norm(Ax_xbar_difference(xj_rl,xbar))}')

            #print(f'\n || z ||: {jnp.linalg.norm(jnp.array(list(z_arr.values())))}')
            #print(f'\n || y ||: {jnp.linalg.norm(jnp.array(list(y_arr.values())))}')
            #print(f'\nSolver messages: {solver_messages}')
            #print(f'\nObjective function (with penalty): {sum(objective_i)}')
            print(f'\n|| xj_rl ||: {np.linalg.norm(list(xj_rl.values()))}')
            print(f'\nxbar: {np.linalg.norm(list(xbar.values()))}')
            #print(f'\n z: {jnp.array(list(z_arr.values()))}')
            #print(f'\nGlobal infeasibility with z || Axr + Bx + + z||: {jnp.linalg.norm(xj_arr - xbar + z)}')
           #print("*******************************************************************")
            #Check if slack variable hasn't change
            t += 1


        

        
        print(f'\n\nN. Iteration for Outer Loop: {k}')
        print(f'Number of iterationns for inner loop: {t}')
        print('\nConstraints violation for each region')
        for idx in range(len(eq_cons_violation)):
            print(f'Region {idx + 1}\n \t-Equality constraints violation: {jnp.sum(jnp.abs(eq_cons_violation[idx]))}')
            print(f'\t-Ineq constraint violation: {jnp.sum(ineq_cons_violation[idx])}')
        print("\n Global Infeasibility without z ||Ax^k + Bx^k||: ",jnp.linalg.norm(Ax_xbar_difference(xj_rl,xbar)))
        print(f'\nGeneretation cost: {objective(x_r_arr,net,regions)}')
        print("|| z ||Norm of Slack variable: ",jnp.linalg.norm(jnp.array(list(z_arr.values()))))
        #Update outer dual variables
        print(f'\nUpdating Alpha...')
        alpha = jnp.clip(alpha + beta * jnp.array(list(z_arr.values())),-1e12,1e12)

        
        if theta * jnp.linalg.norm(jnp.array(list(z_arr.values()))) >=  z_values[-1]:
            beta = jnp.clip(c * beta,1000,1e24)
            print("Updating beta = ",beta)
        

        
        z_values.append(jnp.linalg.norm(jnp.array(list(z_arr.values()))))
        print('\n----------------------------------------------------------------------')
        
        
        infeasibility_arr.append(jnp.linalg.norm(Ax_xbar_difference(xj_rl,xbar)))
        gcost_arr.append(objective(x_r_arr,net,regions))

        

        k += 1

        end_time = time.time()
        iteration_time = end_time - start_time
        iteration_times.append(iteration_time)

        #if  jnp.abs(infeasibility_arr[-1] - infeasibility_arr[-2]) <= 1e-8:
        #    print("\nNo changes")
        #    break

        
    return {
        "x_r": x_r_arr,
        "xbar": xbar,
        "infeasibility_arr": infeasibility_arr,
        "generation_cost_arr": gcost_arr,
        "iteration_times": iteration_times
        
    }


def get_xj_rl(x_r_arr,regions):

    """
    Get the local copies of variables in each region
    """

    xj_r_global = {}

    for idx,x_r in enumerate(x_r_arr):
        region = idx + 1

        x_int = jnp.array(list(regions[region][0]))
        xj_int = jnp.array(list(regions[region][1]))
        x_bound = jnp.array(list(regions[region][2]))
        
        xj_int_idx = jnp.where(jnp.isin(x_int, xj_int))[0]


        X_int = x_r[:len(x_int) * 4].reshape((4,-1))
        X_bound = x_r[len(x_int) * 4:].reshape((2,-1))


        X_bound_v = X_bound.reshape(-1)
        X_bound_v_e = X_bound_v[:len(X_bound_v) // 2]
        X_bound_v_f = X_bound_v[len(X_bound_v) // 2:]
    
        bound_xj = jnp.sqrt(X_bound_v_e ** 2 +  X_bound_v_f ** 2)
        #Arx^r -  xbar
        int_xj = jnp.sqrt(X_int[2,xj_int_idx] ** 2 +  X_int[3,xj_int_idx] ** 2)

        buses_idx = jnp.concatenate([xj_int,x_bound],dtype=jnp.int32)
        xj_r_unsorted = jnp.concatenate([int_xj,bound_xj])
    
        sorted_indices = jnp.argsort(buses_idx)
        buses_idx_sorted = buses_idx[sorted_indices]
        xj_r = xj_r_unsorted[sorted_indices]
        
        for k,v in zip(buses_idx_sorted,xj_r):
            xj_r_global[f'{k}_{region}'] = float(v)   
    
        

        
    
    return xj_r_global


def Ax_xbar_difference(xj_rl, xbar):

    differences = {}
    differences = {key: 0 for key in xj_rl.keys()}
    for bus,global_copy in xbar.items():
        local_copies_xj = {key:value for key,value in xj_rl.items() if key.startswith(str(bus))}
        for local_bus,local_c in local_copies_xj.items():
            differences[local_bus] = local_c - global_copy

    
    return jnp.array(list(differences.values()))


def Ax_xbar_z_difference(xj_rl, xbar,z_arr):

    differences = {}
    differences = {key: 0 for key in xj_rl.keys()}
    for bus,global_copy in xbar.items():
        local_copies_xj = {key:value for key,value in xj_rl.items() if key.startswith(str(bus))}
        for local_bus,local_c in local_copies_xj.items():
            differences[local_bus] = local_c - global_copy + z_arr.get(local_bus)

    
    return jnp.array(list(differences.values()))



def xbar_update(xj_rl, xbar, z_arr, y_arr, rho):
    new_xbar = {}
    for bus in xbar.keys():
        local_copies = {key: value for key, value in xj_rl.items() if key.startswith(str(bus))}
        numerator = 0
        for local_bus, local_c in local_copies.items():
            numerator += rho * (local_c + z_arr.get(local_bus)) + y_arr.get(local_bus)
        new_xbar[bus] = float(jnp.clip(numerator / (len(local_copies) * rho), 0.94, 1.05999))
    return new_xbar



def objective(x_r_arr,net,regions):

    #Objective function
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






