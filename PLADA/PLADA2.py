from .LocalUpdate_PLADA_ACOPF import LocalUpdate_ACOPF
import time
import torch




def ADMM_ACOPF(net,regions,G,B,S,d,rho,x_r_arr0,bnds_xr_arr0,xbar0,y_arr0,max_iter):

    #Initialize values for Outer Loop
    t = 1
    tol = 1e-5

    x_r_arr = x_r_arr0
    xj_rl = get_xj_rl(x_r_arr,regions)
    bnds_arr = bnds_xr_arr0

    xbar = xbar0

    y_arr = y_arr0 #Dual variables for consensus between local copies and global variables
    # Initialize constraint dual variables on GPU
    lambda_eq_arr = []
    mu_ineq_arr = []

    gcost_arr = [objective(net,regions,x_r_arr)]
    infeasibility_arr = [torch.norm(Ax_xbar_difference(xj_rl,xbar))]

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
    
    print(f'\n Total Generation cost: {objective(net,regions,x_r_arr)}')
    print(f'\n|| Axr + Bx ||: {torch.norm(Ax_xbar_difference(xj_rl,xbar))}')
    
 



    while float(torch.norm(Ax_xbar_difference(xj_rl,xbar))) >= float(tol):
  
        """
        Inner Loop
        """
        if t > max_iter:
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
            y_r = torch.tensor(list(y_r_dict.values()),dtype=torch.float32)

            bnds_r = bnds_arr[idx]
            xbar_r_dict = {k:v for k,v in xbar.items() if k in boundBuses_idx}
            xbar_r = torch.tensor(list(xbar_r_dict.values()),dtype=torch.float32)

            local_update =  LocalUpdate_ACOPF(net,regions,region,G,B,S,xbar_r,rho,y_r)
            if t == 1:
                lambda_eq_arr.append(torch.zeros(len(local_update.eq_constraints(x_r)), device=x_r.device, dtype=torch.float64))
                mu_ineq_arr.append(torch.zeros(len(local_update.ineq_constraints(x_r)), device=x_r.device, dtype=torch.float64))

            lambda_eq = lambda_eq_arr[idx]
            mu_ineq = mu_ineq_arr[idx]

            #Implement Interior Point Method Solver to solve Local Problem from region R
            #res_local = ipopt(local_update.objective,local_update.eq_constraints,local_update.ineq_constraints,x_r,bnds_r)
            x_r_new = gradient_descent(local_update.objective,local_update.eq_constraints,local_update.ineq_constraints,x_r,bnds_r,torch.tensor(10),lambda_eq,mu_ineq)
            obj_arr.append(local_update.objective(x_r_new))
            #solver_messages.append(res_local['message'])
            eq_cons_violation.append(local_update.eq_constraints(x_r_new))
            ineq_cons_violation.append(local_update.ineq_constraints(x_r_new)[torch.where(local_update.ineq_constraints(x_r_new) >  1e-4)])


            #Add new values from x_r
            x_r_new_arr.append(x_r_new)


        xj_rl_new = get_xj_rl(x_r_new_arr,regions)
        xbar_new = xbar_update(xj_rl_new,xbar,y_arr,rho)            

        #Slack Variable update
        Ax_Bx_diff_new = Ax_xbar_difference(xj_rl_new,xbar_new)

        #Dual variable update
        y_new_values = torch.tensor(list(y_arr.values())) +  rho * (Ax_Bx_diff_new)
        y_arr_new = {key: new_value for key, new_value in zip(y_arr.keys(), y_new_values)}

        #update rho
                        
        #if jnp.linalg.norm(Ax_xbar_z_difference(xj_rl_new,xbar_new,z_arr_new)) > theta * jnp.linalg.norm(Ax_xbar_z_difference(xj_rl,xbar,z_arr)):
        #    rho = lmbda * rho
        #    print(f'Updating rho={rho}')
             
            

            


        x_r_arr = x_r_new_arr
        xj_rl = xj_rl_new
        xbar = xbar_new
        y_arr = y_arr_new






        
        
        print(f'\n\nN. Loop: {t}')
        print('\nConstraints violation for each region')
        for idx in range(len(eq_cons_violation)):
            print(f'Region {idx + 1}\n \t-Equality constraints violation: {torch.sum(torch.abs(eq_cons_violation[idx]))}')
            print(f'\t-Ineq constraint violation: {torch.sum(ineq_cons_violation[idx])}')
            print(f'\t-Local objective function (with penalty): {obj_arr[idx]}')
        print("\n Infeasibility ||Ax^k + Bx^k||: ",float(torch.norm(Ax_xbar_difference(xj_rl,xbar))))
        print(f'\nxbar: {xbar}')
        print(f'\nxj_rl: {xj_rl}')
        print(f'\nGeneretation cost: {objective(net,regions,x_r_arr)}')


        print('\n----------------------------------------------------------------------')
        
        
        infeasibility_arr.append(torch.norm(Ax_xbar_difference(xj_rl,xbar)))
        gcost_arr.append(objective(net,regions,x_r_arr))

        

        t += 1

        #if  jnp.abs(infeasibility_arr[-1] - infeasibility_arr[-2]) <= 1e-8:
        #    print("\nNo changes")
        #    break

        
    return {
        "x_r": x_r_arr,
        "xbar": xbar,
        "infeasibility_arr": infeasibility_arr,
        "generation_cost_arr": gcost_arr,
        
    }




def gradient_descent(objective, eq_constraints, ineq_constraints, x, bnds, rho2, lambda_eq, mu_ineq):
    
    x = x.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([x], lr=0.1)

    variable_lower_bounds = torch.tensor([b[0] for b in bnds], dtype=x.dtype, device=x.device)
    variable_upper_bounds = torch.tensor([b[1] for b in bnds], dtype=x.dtype, device=x.device)

    scaler = torch.cuda.amp.GradScaler()

    max_iterations = 100

    for iteration in range(max_iterations):
        optimizer.zero_grad()
        L_aug = augmented_lagrangian(x, objective, eq_constraints, ineq_constraints, lambda_eq, mu_ineq, rho2)

        # Check for NaN in L_aug
        if torch.isnan(L_aug):
            print(f"NaN detected in L_aug at iteration {iteration}")
            break

        L_aug.backward()

        # Check for NaN in gradients
        if torch.isnan(x.grad).any():
            print(f"NaN detected in gradients at iteration {iteration}")
            break

        # Optionally clip gradients
        torch.nn.utils.clip_grad_norm_(x, max_norm=1.0)

        optimizer.step()

        # Enforce variable bounds
        with torch.no_grad():
            x.clamp_(min=variable_lower_bounds, max=variable_upper_bounds)

        # Detach x to prevent computation graph growth
        x.detach_().requires_grad_(True)

        with torch.no_grad():
            c_eq = eq_constraints(x)
            c_ineq = ineq_constraints(x)

            # Update dual variables based on residuals
            lambda_eq += rho2 * c_eq
            mu_ineq += rho2 * torch.clamp(c_ineq, min=0)

            if torch.isnan(c_ineq).any():
                break

        # Monitor constraint violations
        constraint_norm = torch.norm(c_eq) + torch.norm(torch.clamp(-c_ineq, min=0))

        # Check for convergence
        if constraint_norm < 1e-4:
            break

    return x




def augmented_lagrangian(x, objective,eq_constraints,ineq_constraints,lambda_eq, mu_ineq, rho):

    f = objective(x)
    c_eq = eq_constraints(x)
    c_ineq = ineq_constraints(x)
    
    # Lagrangian terms
    lagrangian = f
    lagrangian += torch.dot(lambda_eq, c_eq)
    lagrangian += torch.dot(mu_ineq, c_ineq)
    
    # Penalty terms for equality constraints
    penalty_eq = (rho / 2) * torch.sum(c_eq ** 2)
    
    # Penalty terms for inequality constraints
    penalty_ineq = (rho / 2) * torch.sum(torch.clamp(c_ineq, min=0) ** 2)

    
    # Total augmented Lagrangian
    L_aug = lagrangian + penalty_eq + penalty_ineq
    #L_aug = lagrangian

    return L_aug



def get_xj_rl(x_r_arr,regions):

    """
    Get the local copies of variables in each region
    """

    xj_r_global = {}

    for idx,x_r in enumerate(x_r_arr):
        region = idx + 1
        x_int = torch.tensor(torch.sort(torch.tensor(list(regions[region][0])))[0],dtype=torch.int64)
        xj_int = torch.tensor(torch.sort(torch.tensor(list(regions[region][1])))[0],dtype=torch.int64)
        x_bound = torch.tensor(torch.sort(torch.tensor(list(regions[region][2])))[0],dtype=torch.int64)
        
        xj_int_idx = torch.tensor(torch.where(torch.isin(x_int, xj_int))[0],dtype=torch.int64)


        X_int = x_r[:len(x_int) * 4].reshape((4,-1))
        X_bound = x_r[len(x_int) * 4:].reshape((2,-1))


        X_bound_v = X_bound.reshape(-1)
        X_bound_v_e = X_bound_v[:len(X_bound_v) // 2]
        X_bound_v_f = X_bound_v[len(X_bound_v) // 2:]
    
        bound_xj = torch.sqrt(X_bound_v_e ** 2 +  X_bound_v_f ** 2)
        #Arx^r -  xbar
        int_xj = torch.sqrt(X_int[2,xj_int_idx] ** 2 +  X_int[3,xj_int_idx] ** 2)

        buses_idx = torch.cat([xj_int.long(),x_bound.long()],dim=0)
        xj_r_unsorted = torch.cat([int_xj,bound_xj],dim=0)
    
        sorted_indices = torch.argsort(buses_idx)
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

    
    return torch.tensor(list(differences.values()))



def xbar_update(xj_rl, xbar,y_arr, rho):
    new_xbar = {}
    for bus in xbar.keys():
        local_copies = {key: value for key, value in xj_rl.items() if key.startswith(str(bus))}
        numerator = 0
        for local_bus, local_c in local_copies.items():
            numerator += rho * local_c  + y_arr.get(local_bus)
        new_xbar[bus] = float(torch.clip(torch.tensor(numerator / (len(local_copies) * rho)), 0.94, 1.05999))
    return new_xbar



def objective(net,regions,x_r_arr):
    "Local Objective function"
    total_c = 0
    for idx,x_r in enumerate(x_r_arr):
        region = idx + 1
        x_int = torch.tensor(torch.sort(torch.tensor(list(regions[region][0])))[0],dtype=torch.int64)
        n_int = len(x_int)
        X_int = x_r[:n_int * 4].reshape((4, n_int))
        X_bound = x_r[n_int * 4:].reshape((2, -1))        

        pg = X_int[0, :]  # Active power outputs at internal buses
        gencost_data = net['gencost']

        # Cost coefficients for all generators
        a_all = gencost_data[:, 4]
        b_all = gencost_data[:, 5]
        c_all = gencost_data[:, 6]

        # Generator bus numbers (as integers)
        gen_bus_nums_all = gencost_data[:, 0].long()

        # Internal bus numbers (as integers)
        x_int_bus_nums = x_int.long()

        # Boolean mask: True where generator buses are in the region
        is_in_region = torch.isin(gen_bus_nums_all, x_int_bus_nums)

        # Indices of generators within the region
        region_gen_indices = torch.nonzero(is_in_region).squeeze()

        # Handle the case when no generators are in the region
        if region_gen_indices.numel() == 0:
            total_c += torch.tensor(0.0, dtype=x_r.dtype, device=x_r.device)
        else:
            # Cost coefficients for generators in the region
            a_region = a_all[region_gen_indices]
            b_region = b_all[region_gen_indices]
            c_region = c_all[region_gen_indices]
            gen_bus_nums_region = gen_bus_nums_all[region_gen_indices]

            # Map bus numbers to indices in pg
            bus_num_to_pg_idx = {int(bus_num.item()): idx for idx, bus_num in enumerate(x_int_bus_nums)}

            # Indices in pg corresponding to generator buses in the region
            pg_indices = []
            for bus_num in gen_bus_nums_region:
                pg_idx = bus_num_to_pg_idx.get(int(bus_num.item()))
                if pg_idx is None:
                    raise ValueError(f"Generator bus index {bus_num.item()} not found in internal buses.")
                pg_indices.append(pg_idx)
            pg_indices = torch.tensor(pg_indices, dtype=torch.long, device=x_r.device)

            # Active power outputs for generators in the region
            pg_gen = pg[pg_indices]

            # Compute total cost for generators in the region
            total_c += torch.sum(a_region * pg_gen ** 2 + b_region * pg_gen + c_region)

                

    return total_c






