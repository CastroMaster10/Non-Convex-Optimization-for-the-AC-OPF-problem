import collections
import torch


class LocalUpdate_ACOPF:

    def __init__(self,net,regions,region,G,B,S,xbar_r,rho,y):

   #Transform each data structure of the net into jax numpy
        for key,matrix in net.items():
            net[key] = torch.tensor(matrix)
        self.net = net
        self.regions = regions
        self.region = region
        self.G =  G
        self.B = B
        self.S = S
        self.xbar_r = xbar_r
        self.rho = rho
        self.y = y
        self.x_int = torch.tensor(torch.sort(torch.tensor(list(self.regions[self.region][0])))[0],dtype=torch.int64)
        self.xj_int = torch.tensor(torch.sort(torch.tensor(list(self.regions[self.region][1])))[0],dtype=torch.int64)
        self.x_bound = torch.tensor(torch.sort(torch.tensor(list(self.regions[self.region][2])))[0],dtype=torch.int64)
        
        self.xj_int_idx = torch.tensor(torch.where(torch.isin(self.x_int, self.xj_int))[0],dtype=torch.int64)
    
    def objective(self,x):
        "Local Objective function"
        #X_int = x.reshape((4,-1))[:,x_int]
        #X_bound = x.reshape((4,-1))[:,x_bound][2:,:]
        n_int = len(self.x_int)
        X_int = x[:n_int * 4].reshape((4, n_int))
        X_bound = x[n_int * 4:].reshape((2, -1))        

        pg = X_int[0, :]  # Active power outputs at internal buses
        gencost_data = self.net['gencost']

        # Cost coefficients for all generators
        a_all = gencost_data[:, 4]
        b_all = gencost_data[:, 5]
        c_all = gencost_data[:, 6]

        # Generator bus numbers (as integers)
        gen_bus_nums_all = gencost_data[:, 0].long()

        # Internal bus numbers (as integers)
        x_int_bus_nums = self.x_int.long()

        # Boolean mask: True where generator buses are in the region
        is_in_region = torch.isin(gen_bus_nums_all, x_int_bus_nums)

        # Indices of generators within the region
        region_gen_indices = torch.nonzero(is_in_region).squeeze()

        # Handle the case when no generators are in the region
        if region_gen_indices.numel() == 0:
            total_c = torch.tensor(0.0, dtype=x.dtype, device=x.device)
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
            pg_indices = torch.tensor(pg_indices, dtype=torch.long, device=x.device)

            # Active power outputs for generators in the region
            pg_gen = pg[pg_indices]

            # Compute total cost for generators in the region
            total_c = torch.sum(a_region * pg_gen ** 2 + b_region * pg_gen + c_region)

                
        X_bound_v = X_bound.reshape(-1)
        X_bound_v_e = X_bound_v[:len(X_bound_v) // 2]
        X_bound_v_f = X_bound_v[len(X_bound_v) // 2:]


        #Ar_xr_e = jnp.concatenate([jnp.zeros(self.idx_buses_before),X_bound_v_e,jnp.zeros(self.idx_buses_after)])
        #Ar_xr_f = jnp.concatenate([jnp.zeros(self.idx_buses_before),X_bound_v_f,jnp.zeros(self.idx_buses_after)])

        #Ar_xr = jnp.concatenate([Ar_xr_e,Ar_xr_f])
    
        bound_xj = torch.sqrt(X_bound_v_e ** 2 +  X_bound_v_f ** 2)
        #Arx^r -  xbar
        int_xj = torch.sqrt(X_int[2,self.xj_int_idx] ** 2 +  X_int[3,self.xj_int_idx] ** 2)



        buses_idx = torch.cat([self.xj_int.long(),self.x_bound.long()],dim=0)
        xj_r_unsorted = torch.cat([int_xj,bound_xj],dim=0)
        

 
   
        sorted_indices = torch.argsort(buses_idx)
        buses_idx_sorted = buses_idx[sorted_indices]
        xj_r = xj_r_unsorted[sorted_indices]
        xj_r = xj_r.to(dtype=x.dtype, device=x.device)

  
        # Ensure self.y and self.xbar_r have the correct dtype and device
        self.y = self.y.to(dtype=x.dtype, device=x.device)
        self.xbar_r = self.xbar_r.to(dtype=x.dtype, device=x.device)

        y_xj_r = torch.dot(self.y, xj_r)

        consensus = xj_r - self.xbar_r

        penalty = (self.rho / 2) * torch.norm(consensus) ** 2
        f_xr = total_c + y_xj_r + penalty

        return f_xr

    
  
    def eq_constraints(self, x):
    

        n_int = len(self.x_int)
        n_bound = len(self.x_bound)

        X_int = x[:n_int * 4].reshape((4, -1))
        X_bound = x[n_int * 4 :].reshape((2, -1))

        pd_int = self.net['bus'][self.x_int, 2].to(x.device)
        qd_int = self.net['bus'][self.x_int, 3].to(x.device)
        pg_int = X_int[0, :]
        qg_int = X_int[1, :]

        cons1, cons2 = self.power_balance_constraints_vectorized(
            X_int, X_bound, pd_int, qd_int, pg_int, qg_int, self.x_int, self.x_bound
        )

        return torch.cat([cons1, cons2],dim=0)


    def power_balance_constraints_vectorized(self, X_int, X_bound, pd_int, qd_int, pg_int, qg_int, x_int, x_bound):
        device = X_int.device

        ei = X_int[2, :]  # Shape: (n_int,)
        fi = X_int[3, :]
        e_bound = X_bound[0, :]  # Shape: (n_bound,)
        f_bound = X_bound[1, :]

        # Ensure indices are of type torch.long and on CPU for indexing
        x_int_cpu = x_int.long().cpu()
        x_bound_cpu = x_bound.long().cpu()

        # Extract G and B matrices
        G_int = self.G[x_int_cpu][:, x_int_cpu].to(device)      # Shape: (n_int, n_int)
        B_int = self.B[x_int_cpu][:, x_int_cpu].to(device)
        G_bound = self.G[x_int_cpu][:, x_bound_cpu].to(device)  # Shape: (n_int, n_bound)
        B_bound = self.B[x_int_cpu][:, x_bound_cpu].to(device)

        # Compute interactions with interior buses
        ei_ej = ei.unsqueeze(1) * ei.unsqueeze(0)  # Shape: (n_int, n_int)
        fi_fj = fi.unsqueeze(1) * fi.unsqueeze(0)
        ei_fj = ei.unsqueeze(1) * fi.unsqueeze(0)
        ej_fi = ei.unsqueeze(0) * fi.unsqueeze(1)

        term1_int = G_int * (ei_ej + fi_fj)
        term2_int = -B_int * (ei_fj - ej_fi)
        sum_terms_int = term1_int + term2_int
        sum_terms_int = sum_terms_int - torch.diag(torch.diag(sum_terms_int))  # Exclude diagonal terms
        sum_over_j_int = torch.sum(sum_terms_int, dim=1)

        # Compute interactions with boundary buses
        ei_ej_bound = ei.unsqueeze(1) * e_bound.unsqueeze(0)  # Shape: (n_int, n_bound)
        fi_fj_bound = fi.unsqueeze(1) * f_bound.unsqueeze(0)
        ei_fj_bound = ei.unsqueeze(1) * f_bound.unsqueeze(0)
        ej_fi_bound = fi.unsqueeze(1) * e_bound.unsqueeze(0)  # Shape: (n_int, n_bound)

        # Verify that shapes match
        assert G_bound.shape == ei_ej_bound.shape, f"G_bound shape {G_bound.shape} does not match ei_ej_bound shape {ei_ej_bound.shape}"

        term1_bound = G_bound * (ei_ej_bound + fi_fj_bound)
        term2_bound = -B_bound * (ei_fj_bound - ej_fi_bound)
        sum_over_j_bound = torch.sum(term1_bound + term2_bound, dim=1)

        # Corrected extraction of diagonal elements
        G_diag = self.G[x_int_cpu, x_int_cpu].to(device)
        B_diag = self.B[x_int_cpu, x_int_cpu].to(device)

        # Combine terms for cons1
        cons1 = (
            G_diag * (ei ** 2 + fi ** 2)
            - pg_int
            + pd_int
            + sum_over_j_int
            + sum_over_j_bound
        )

        # Compute cons2 similarly
        term1_int = -B_int * (ei_ej + fi_fj)
        term2_int = -G_int * (ei_fj - ej_fi)
        sum_terms_int = term1_int + term2_int
        sum_terms_int = sum_terms_int - torch.diag(torch.diag(sum_terms_int))
        sum_over_j_int = torch.sum(sum_terms_int, dim=1)

        term1_bound = -B_bound * (ei_ej_bound + fi_fj_bound)
        term2_bound = -G_bound * (ei_fj_bound - ej_fi_bound)
        sum_over_j_bound = torch.sum(term1_bound + term2_bound, dim=1)

        cons2 = (
            B_diag * (ei ** 2 + fi ** 2)
            - qg_int
            + qd_int
            + sum_over_j_int
            + sum_over_j_bound
        )

        return cons1, cons2




    def ineq_constraints(self,x):


        n_int = len(self.x_int)
        n_bound = len(self.x_bound)

        # Extract internal and boundary variables
        X_int = x[: n_int * 4].reshape((4, -1))
        X_bound = x[n_int * 4 :].reshape((2, -1))

        ei = X_int[2, :]  # Real voltage components at internal buses
        fi = X_int[3, :]  # Imaginary voltage components at internal buses

        # Convert Vmax and Vmin to tensors using x_int as indices
        Vmax = torch.tensor(
            self.net['bus'][self.x_int.cpu().numpy(), 11], dtype=x.dtype, device=x.device
        )
        Vmin = torch.tensor(
            self.net['bus'][self.x_int.cpu().numpy(), 12], dtype=x.dtype, device=x.device
        )

        # Call the thermal limit constraints function
        cons3, cons4 = self.thermal_limit_buses_vectorized(X_int, X_bound, self.x_int, self.x_bound)

        # Voltage magnitude inequality constraints
        cons5 = Vmin**2 - (ei**2 + fi**2)
        cons6 = (ei**2 + fi**2) - Vmax**2

        # Concatenate all inequality constraints
        return torch.cat([cons3, cons4, cons5, cons6])


    def thermal_limit_buses_vectorized(self,X_int, X_bound, x_int, x_bound):
        # Extract voltage components
        ei = X_int[2, :]       # Shape: (n_int,)
        fi = X_int[3, :]
        e_bound = X_bound[0, :]  # Shape: (n_bound,)
        f_bound = X_bound[1, :]

        # Ensure x_int and x_bound are of dtype torch.int64 for indexing
        x_int = x_int.long()
        x_bound = x_bound.long()

        # Extract submatrices from G, B, and S tensors using advanced indexing
        G_int = self.G[x_int][:, x_int]
        B_int = self.B[x_int][:, x_int]
        S_int = self.S[x_int][:, x_int]

        G_bound = self.G[x_int][:, x_bound]
        B_bound = self.B[x_int][:, x_bound]
        S_bound = self.S[x_int][:, x_bound]

        # Interactions with interior buses
        delta_e = ei.unsqueeze(1) - ei.unsqueeze(0)  # Shape: (n_int, n_int)
        delta_f = fi.unsqueeze(1) - fi.unsqueeze(0)

        pij_int = -G_int * delta_e - B_int * delta_f
        qij_int = B_int * delta_e - G_int * delta_f
        S_limit_int = pij_int**2 + qij_int**2 - S_int**2
        cons3 = torch.sum(S_limit_int, dim=1)  # Summing over j for each i

        # Interactions with boundary buses
        delta_e_bound = ei.unsqueeze(1) - e_bound.unsqueeze(0)  # Shape: (n_int, n_bound)
        delta_f_bound = fi.unsqueeze(1) - f_bound.unsqueeze(0)

        pij_bound = -G_bound * delta_e_bound - B_bound * delta_f_bound
        qij_bound = B_bound * delta_e_bound - G_bound * delta_f_bound
        S_limit_bound = pij_bound**2 + qij_bound**2 - S_bound**2
        cons4 = torch.sum(S_limit_bound, dim=1)  # Summing over j for each i

        return cons3, cons4
