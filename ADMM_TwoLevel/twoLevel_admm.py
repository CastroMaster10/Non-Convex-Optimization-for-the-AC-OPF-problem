
import jax.numpy as jnp
import numpy as np
from .ipopt2 import ipopt
from .LocalUpdate_twoLevel_ACOPF import LocalUpdate_ACOPF
from .GlobalUpdate_twoLevel_ACOPF import GlobalUpdate_ACOPF


def generate_x0_bnds(net,regions):

    x_r = {}
    bnds = {}


    gen_data = jnp.array(net["gen"])
    bus_data = jnp.array(net["bus"])

    for region,buses in regions.items():

        bnds_i = []
        x_int = jnp.array(list(buses[0]))
        x_bou = jnp.array(list(buses[1]))

        pg_min_int = gen_data[x_int,:][:, 9]
        qg_min_int = gen_data[x_int,:][:, 4]
        pg_max_int = gen_data[x_int,:][:, 8]
        qg_max_int = gen_data[x_int,:][:, 3]

        #v_min_int = bus_data[x_int,:][:,12]
        v_min_int = np.zeros(len(x_int)) 
        v_max_int = bus_data[x_int,:][:,11]  


        X_int = np.zeros((4,len(x_int)))
        Bnds_int_min = np.zeros((4,len(x_int)))
        Bnds_int_max = np.zeros((4,len(x_int)))


        for j in range(len(x_int)):
            #iterates through each column
            X_int[0,j] = 0
            X_int[1,j] = 0
            X_int[2,j] = 1
            X_int[3,j] =  0

            Bnds_int_min[0,j] =  pg_min_int[j]
            Bnds_int_min[1,j] =  qg_min_int[j]
            Bnds_int_min[2,j] =  v_min_int[j]
            Bnds_int_min[3,j] =  v_min_int[j]

            Bnds_int_max[0,j] =  pg_max_int[j]
            Bnds_int_max[1,j] =  qg_max_int[j]
            Bnds_int_max[2,j] =  v_max_int[j]
            Bnds_int_max[3,j] =  v_max_int[j]

        bnds_int_min = Bnds_int_min.reshape(-1)
        bnds_int_max = Bnds_int_max.reshape(-1)
        for k in range(len(bnds_int_min)):
            bnds_i.append((float(bnds_int_min[k]),float(bnds_int_max[k])))


        #v_min_bou = bus_data[x_int,:][:,12]
        v_min_bou = np.zeros(len(x_bou))  
        v_max_bou = bus_data[x_int,:][:,11]  

        X_bound = np.zeros((2,len(x_bou)))

        Bnds_bou_min = np.zeros((2,len(x_bou)))
        Bnds_bou_max = np.zeros((2,len(x_bou)))

 
        for j in range(len(x_bou)):

            X_bound[0,j] = 1
            X_bound[1,j] = 0

            Bnds_bou_min[0,j] =  v_min_bou[j]
            Bnds_bou_min[1,j] =  v_min_bou[j]

            Bnds_bou_max[0,j] =  v_max_bou[j]
            Bnds_bou_max[1,j] =  v_max_bou[j]
        
        bnds_bou_min = Bnds_bou_min.reshape(-1)
        bnds_bou_max = Bnds_bou_max.reshape(-1)
        
        for k in range(len(bnds_bou_min)):
            bnds_i.append((float(bnds_bou_min[k]),float(bnds_bou_max[k])))
        
        x_r[region] = jnp.concatenate([jnp.array(X_int).reshape(-1),jnp.array(X_bound).reshape(-1)])
        bnds[region] = bnds_i
        

    
    return x_r, bnds

def get_xj_arr(x_r_arr,regions,idx_buses_arr):
    Ar_xr_arr = []
    for idx,x_r in enumerate(x_r_arr):
        region  = idx + 1
        idx_buses_before_i =  idx_buses_arr[region][0]
        idx_buses_after_i =  idx_buses_arr[region][1]

        x_int = jnp.array(list(regions[region][0]))
        x_bound = jnp.array(list(regions[region][1]))
        X_bound = x_r[len(x_int) * 4:].reshape((2,-1))
        
        #Calculate A_rx_r to perform operations when updating dual variables and slack variables
        X_bound_v = X_bound.reshape(-1)
        X_bound_v_e = X_bound_v[:len(x_bound)]
        X_bound_v_f = X_bound_v[len(x_bound):]
        Ar_xr_e = jnp.concatenate([jnp.zeros(idx_buses_before_i),X_bound_v_e,jnp.zeros(idx_buses_after_i)])
        Ar_xr_f = jnp.concatenate([jnp.zeros(idx_buses_before_i),X_bound_v_f,jnp.zeros(idx_buses_after_i)])
        Ar_xr = jnp.concatenate([Ar_xr_e,Ar_xr_f])
        Ar_xr_arr.append(Ar_xr)
    

    
    return jnp.sum(jnp.array(Ar_xr_arr),axis=0)


def TwoLevel_ADMM_ACOPF(net,regions,G,B,S,idx_buses_arr,d,beta,alpha):
    

    """
    Outer Loop
    """

    #Initialize values for Outer Loop
    c =  6
    k = 1
    lmbda = 6
    theta = 0.8
    nk =  jnp.array([1/i for i in range(1,100)])

    initial_data = generate_x0_bnds(net,regions)
    x_r_arr = list(initial_data[0].values())
    xj_arr = get_xj_arr(x_r_arr,regions,idx_buses_arr)

    bnds_arr = list(initial_data[1].values())

    y = jnp.zeros(d) #Dual variable
    z = jnp.zeros(d)  #Slack Variable

    global_update = GlobalUpdate_ACOPF(net,regions,beta * 2,y,idx_buses_arr,z,x_r_arr)
    xbar_bnds = [(0,1.1)] * d
    xbar = jnp.zeros(d)



    #xbar = ipopt(global_update.global_update,global_update.eq_constraints,global_update.ineq_constraints, jnp.array(np.random.random(d)),xbar_bnds)
    gcost_arr = []
    infeasibility_arr = []
    global_z = [0]
    while jnp.linalg.norm(xj_arr - xbar) >= jnp.sqrt(d) * 1e-2:

        if k > 10:
            break
        """
        Inner Loop
        """
        #Initialize values for Inner Loop

        y = jnp.zeros(d)
        z = (-alpha -  y) / beta
        if global_z[-1] <= theta * jnp.linalg.norm(z):
            beta = c * beta
            print("Updating beta = ",beta)
        rho = 2 * beta
        
        #initial_data = generate_x0_bnds(net,regions)
        #x_r_arr = x_r_arr0
        #xj_arr = xj_arr0
        #xbar = jnp.zeros(d)
        #x_r_arr =  [x_r + 1e-3 * np.random.random(len(x_r)) for x_r in x_r_arr0]
        #xj_arr = get_xj_arr(x_r_arr,regions,idx_buses_arr)




        t = 1
        print("\n||Ax_r + Bx + z || = ",jnp.linalg.norm(xj_arr - xbar + z))
        print("Threshold: ",(jnp.sqrt(d) / 2500 * k))
        while (float(jnp.linalg.norm(xj_arr - xbar + z)) >= float(0.05)):
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
                ineq_cons_violation.append(local_update.ineq_constraints(x_r_new))


                #Add new values from x_r
                x_r_new_arr.append(x_r_new)
            

            xr_j_new = get_xj_arr(x_r_new_arr,regions,idx_buses_arr)

            #Global update
            global_update = GlobalUpdate_ACOPF(net,regions,rho,y,idx_buses_arr,z,x_r_new_arr)
            xbar_new = ipopt(global_update.global_update,global_update.eq_constraints,global_update.ineq_constraints,xbar,xbar_bnds)['x']

            
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

            global_z.append(jnp.linalg.norm(z))
            

            
            print(f'\nN. Iteration for Inner Loop: {t}')
            print(f'\n|| Axr + Bx + + z||: {jnp.linalg.norm(xj_arr - xbar + z)}')
            """            
            print('\n\t-Constraints violation for each region')
            for idx in range(len(eq_cons_violation)):
                print(f'\t-Region {idx + 1}\n \t* Equality constraints violation: {jnp.sum(eq_cons_violation[idx])} \n \t* Inequality constraints violation: {jnp.sum(ineq_cons_violation[idx])}')
            print("\n\tInfeasibility ||Ax^k + Bx^k + z|| = ",jnp.linalg.norm(xr_j_new - xbar_new + z))
            print(f'\n\tGeneretation cost: {objective(x_r_new_arr,net,regions)}')
            print("\n\t|| z ||Norm of Slack variable: ",jnp.linalg.norm(z))
            print("\n******************************************************************")
            """
            t += 1

        

        
        print(f'\n\nN. Iteration for Outer Loop: {k}')
        print(f'Number of iterationns for inner loop: {t}')
        print('\nConstraints violation for each region')
        for idx in range(len(eq_cons_violation)):
            print(f'Region {idx + 1}\n \t-Equality constraints violation: {jnp.sum(eq_cons_violation[idx])} \n \t-Inequality constraints violation: {jnp.sum(ineq_cons_violation[idx])}')
        print("\nInfeasibility ||Ax^k + Bx^k||: ",jnp.linalg.norm(xr_j_new - xbar_new))
        print(f'\nGeneretation cost: {objective(x_r_new_arr,net,regions)}')
        print("|| z ||Norm of Slack variable: ",jnp.linalg.norm(z))
        print('\n----------------------------------------------------------------------')
        

        infeasibility_arr.append(jnp.linalg.norm(xj_arr - xbar))
        gcost_arr.append(objective(x_r_arr,net,regions))
    
        
        #Update outer dual variables
        #if jnp.linalg.norm(z) <= nk[k - 1]:
        print(f'\nUpdating Alpha...')
        alpha = np.clip(alpha + beta * z,-1e12,1e12)    


        k += 1


        
    return {
        "x_r": x_r_arr,
        "xbar": xbar,
        "z": z,
        "infeasibility_arr": infeasibility_arr,
        "generation_cost_arr": gcost_arr
        
    }


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






