import jax.numpy as jnp
import numpy as np
import jax

class LocalUpdate_ACOPF:

    def __init__(self,net,regions,region,G,B,S,xbar,rho,y,idx_buses_before,idx_buses_after,z):

        #Transform each data structure of the net into jax numpy
        for key,matrix in net.items():
            net[key] = jnp.array(matrix)
        self.net = net
        self.regions = regions
        self.region = region
        self.G =  jnp.array(G)
        self.B = jnp.array(B)
        self.S = jnp.array(S)
        self.xbar = xbar
        self.rho = rho
        self.y = y
        self.idx_buses_before = idx_buses_before
        self.idx_buses_after = idx_buses_after
        self.z = z
        
    
    def objective(self,x):
        "Local Objective function"
        x_int = jnp.array(list(self.regions[self.region][0]))
        x_bound = jnp.array(list(self.regions[self.region][1]))
               
        #X_int = x.reshape((4,-1))[:,x_int]
        #X_bound = x.reshape((4,-1))[:,x_bound][2:,:]
        X_int = x[:len(x_int) * 4].reshape((4,-1))
        X_bound = x[len(x_int) * 4:].reshape((2,-1))        
        
        pg = x[:len(x_int)]
        gencost_data_r = self.net['gencost'][:,4:]

        a = gencost_data_r[:,0]
        b = gencost_data_r[:,1]
        c = gencost_data_r[:,2]
                
        #c_r(x)
        total_c = 0
        idx = 0
        for i in self.net['gencost'][:,0]:
            total_c += a[idx] *  pg[i.astype(int)] ** 2 + b[idx] * pg[i.astype(int)] + c[idx]
            idx += 1

        
        X_bound_v = X_bound.reshape(-1)
        X_bound_v_e = X_bound_v[:len(x_bound)]
        X_bound_v_f = X_bound_v[len(x_bound):]


        Ar_xr_e = jnp.concatenate([jnp.zeros(self.idx_buses_before),X_bound_v_e,jnp.zeros(self.idx_buses_after)])
        Ar_xr_f = jnp.concatenate([jnp.zeros(self.idx_buses_before),X_bound_v_f,jnp.zeros(self.idx_buses_after)])

        Ar_xr = jnp.concatenate([Ar_xr_e,Ar_xr_f])

        #xbar
        xbar_e = self.xbar[:len(self.xbar) // 2]
        xbar_f = self.xbar[len(self.xbar) // 2:]

        xbar_e_r = xbar_e[self.idx_buses_before:self.idx_buses_before + len(x_bound)]
        xbar_f_r = xbar_f[self.idx_buses_before:self.idx_buses_before + len(x_bound)]

        Br_xbar_e = jnp.concatenate([jnp.zeros(self.idx_buses_before),xbar_e_r,jnp.zeros(self.idx_buses_after)])
        Br_xbar_f = jnp.concatenate([jnp.zeros(self.idx_buses_before),xbar_f_r,jnp.zeros(self.idx_buses_after)])
        
        Br_xbar = jnp.concatenate([Br_xbar_e,Br_xbar_f])
        
        #z
        z_e = self.z[:len(self.z) // 2]
        z_f = self.z[len(self.z) // 2:]

        z_e_r = z_e[self.idx_buses_before:self.idx_buses_before + len(x_bound)]
        z_f_r = z_f[self.idx_buses_before:self.idx_buses_before + len(x_bound)]

        Br_z_e = jnp.concatenate([jnp.zeros(self.idx_buses_before),z_e_r,jnp.zeros(self.idx_buses_after)])
        Br_z_f = jnp.concatenate([jnp.zeros(self.idx_buses_before),z_f_r,jnp.zeros(self.idx_buses_after)])
        
        Br_z = jnp.concatenate([Br_z_e,Br_z_f])



        #Arx^r -  xbar
        y_Ax_r = jnp.dot(self.y,Ar_xr)
        #penalty
        consensus = Ar_xr - Br_xbar +  Br_z

        #eq_arr = self.eq_constraints(x)
        #eq_arr_scaled = (eq_arr - jnp.min(eq_arr)) / (jnp.max(eq_arr) - jnp.min(eq_arr))
        penalty =  self.rho /2 * (jnp.linalg.norm(consensus)) ** 2

        return jnp.sum(total_c + y_Ax_r + penalty)
    
    def eq_constraints(self,x):
        x_int = jnp.array(list(self.regions[self.region][0]),dtype=jnp.int32)
        x_bound = jnp.array(list(self.regions[self.region][1]),dtype=jnp.int32)
        n_int = len(x_int)
        n_bound = len(x_bound)

        X_int = x[:n_int * 4].reshape((4, -1))
        X_bound = x[n_int * 4:].reshape((2, -1))

        pd_int = self.net['bus'][x_int, 2]
        qd_int = self.net['bus'][x_int, 3]
        pg_int = X_int[0, :]
        qg_int = X_int[1, :]

        cons1, cons2 = self.power_balance_constraints_vectorized(
            X_int, X_bound, pd_int, qd_int, pg_int, qg_int, x_int, x_bound
        )

        return jnp.concatenate([cons1, cons2])


    def power_balance_constraints_vectorized(self,X_int, X_bound, pd_int, qd_int, pg_int, qg_int, x_int, x_bound):
        ei = X_int[2, :]  # Shape (n_int,)
        fi = X_int[3, :]
        e_bound = X_bound[0, :]  # Shape (n_bound,)
        f_bound = X_bound[1, :]

        # Compute G and B matrices for interior and boundary buses
        G_int = self.G[x_int][:, x_int]
        B_int = self.B[x_int][:, x_int]
        G_bound = self.G[x_int][:, x_bound]
        B_bound = self.B[x_int][:, x_bound]

        # Compute interactions with interior buses
        ei_ej = ei[:, None] * ei[None, :]
        fi_fj = fi[:, None] * fi[None, :]
        ei_fj = ei[:, None] * fi[None, :]
        ej_fi = ei[None, :] * fi[:, None]

        term1_int = G_int * (ei_ej + fi_fj)
        term2_int = -B_int * (ei_fj - ej_fi)
        sum_terms_int = term1_int + term2_int
        sum_terms_int = sum_terms_int - jnp.diag(jnp.diag(sum_terms_int))  # Exclude diagonal terms
        sum_over_j_int = jnp.sum(sum_terms_int, axis=1)

        # Compute interactions with boundary buses
        ei_ej_bound = ei[:, None] * e_bound[None, :]
        fi_fj_bound = fi[:, None] * f_bound[None, :]
        ei_fj_bound = ei[:, None] * f_bound[None, :]
        ej_fi_bound = e_bound[None, :] * fi[:, None]

        term1_bound = G_bound * (ei_ej_bound + fi_fj_bound)
        term2_bound = -B_bound * (ei_fj_bound - ej_fi_bound)
        sum_over_j_bound = jnp.sum(term1_bound + term2_bound, axis=1)

        # Combine terms for cons1
        G_diag = self.G[x_int, x_int]
        cons1 = G_diag * (ei**2 + fi**2) - pg_int + pd_int + sum_over_j_int + sum_over_j_bound

        # Compute cons2 similarly
        term1_int = -B_int * (ei_ej + fi_fj)
        term2_int = -G_int * (ei_fj - ej_fi)
        sum_terms_int = term1_int + term2_int
        sum_terms_int = sum_terms_int - jnp.diag(jnp.diag(sum_terms_int))
        sum_over_j_int = jnp.sum(sum_terms_int, axis=1)

        term1_bound = -B_bound * (ei_ej_bound + fi_fj_bound)
        term2_bound = -G_bound * (ei_fj_bound - ej_fi_bound)
        sum_over_j_bound = jnp.sum(term1_bound + term2_bound, axis=1)

        B_diag = self.B[x_int, x_int]
        cons2 = -B_diag * (ei**2 + fi**2) - qg_int + qd_int + sum_over_j_int + sum_over_j_bound

        return cons1, cons2


    def ineq_constraints(self,x):
        x_int = jnp.array(list(self.regions[self.region][0]),dtype=jnp.int32)
        x_bound = jnp.array(list(self.regions[self.region][1]),dtype=jnp.int32)
        n_int = len(x_int)
        n_bound = len(x_bound)

        X_int = x[:n_int * 4].reshape((4, -1))
        X_bound = x[n_int * 4:].reshape((2, -1))

        ei = X_int[2, :]
        fi = X_int[3, :]
        Vmax = self.net['bus'][x_int, 11]
        Vmin = self.net['bus'][x_int, 12]

        cons3, cons4 = self.thermal_limit_buses_vectorized(X_int, X_bound, x_int, x_bound)
        cons5 = Vmin**2 - (ei**2 + fi**2)
        cons6 = (ei**2 + fi**2) - Vmax**2

        return jnp.concatenate([cons3, cons4, cons5, cons6])


    def thermal_limit_buses_vectorized(self,X_int, X_bound, x_int, x_bound):
        ei = X_int[2, :]
        fi = X_int[3, :]
        e_bound = X_bound[0, :]
        f_bound = X_bound[1, :]

        # Interactions with interior buses
        ei_ej = ei[:, None] - ei[None, :]
        fi_fj = fi[:, None] - fi[None, :]
        delta_e = ei_ej
        delta_f = fi_fj

        G_int = self.G[x_int][:, x_int]
        B_int = self.B[x_int][:, x_int]
        S_int = self.S[x_int][:, x_int]

        pij_int = -G_int * (delta_e) - B_int * (delta_f)
        qij_int = B_int * (delta_e) - G_int * (delta_f)
        S_limit_int = pij_int**2 + qij_int**2 - S_int**2
        S_limit_int = jnp.sum(S_limit_int, axis=1)

        # Interactions with boundary buses
        delta_e_bound = ei[:, None] - e_bound[None, :]
        delta_f_bound = fi[:, None] - f_bound[None, :]

        G_bound = self.G[x_int][:, x_bound]
        B_bound = self.B[x_int][:, x_bound]
        S_bound = self.S[x_int][:, x_bound]

        pij_bound = -G_bound * (delta_e_bound) - B_bound * (delta_f_bound)
        qij_bound = B_bound * (delta_e_bound) - G_bound * (delta_f_bound)
        S_limit_bound = pij_bound**2 + qij_bound**2 - S_bound**2
        S_limit_bound = jnp.sum(S_limit_bound, axis=1)

        cons3 = S_limit_int
        cons4 = S_limit_bound

        return cons3, cons4

