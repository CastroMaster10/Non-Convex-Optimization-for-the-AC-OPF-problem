import jax.numpy as jnp
import numpy as np

class GlobalUpdate_ACOPF:

    def __init__(self,net,regions,rho,y,idx_buses_arr,z):
        #Transform each data structure of the net into jax numpy
        for key,matrix in net.items():
            net[key] = jnp.array(matrix)
        self.net = net
        self.regions = regions
        self.rho = rho  
        self.y = y
        self.idx_buses_arr = idx_buses_arr 
        self.z = z #Slack Variable
    
        

    def global_update(self,x_r_list):
        Ar_xr_arr = []
        m = len(x_r_list) #Number of machines
        

        for idx,x_r in enumerate(x_r_list):
            region = idx + 1
            idx_buses_before = self.idx_buses_arr[region][0]
            idx_buses_after = self.idx_buses_arr[region][1]

            x_int = jnp.array(list(self.regions[region][0]))
            x_bound = jnp.array(list(self.regions[region][1]))


            X_int = x_r[:len(x_int) * 4].reshape((4,-1))
            X_bound = x_r[len(x_int) * 4:].reshape((2,-1))


            X_bound_v = X_bound.reshape(-1)
            X_bound_v_e = X_bound_v[:len(x_bound)]
            X_bound_v_f = X_bound_v[len(x_bound):]
            Ar_xr_e = jnp.concatenate([jnp.zeros(idx_buses_before),X_bound_v_e,jnp.zeros(idx_buses_after)])
            Ar_xr_f = jnp.concatenate([jnp.zeros(idx_buses_before),X_bound_v_f,jnp.zeros(idx_buses_after)])

            Ar_xr = jnp.concatenate([Ar_xr_e,Ar_xr_f])


            Ar_xr_arr.append(Ar_xr)

        

        x_i_new = jnp.sum(jnp.array(Ar_xr_arr),axis=0)
        #xbar_new = (self.y + self.rho * (x_i_new + self.z)) / self.rho * (m)

        xbar_new = 1/m * (x_i_new + self.z) + (1 / (2 * m * self.rho)) * self.y
        return xbar_new

    def eq_constraints(self,x):
        return jnp.array([0],dtype=jnp.float32)

    def ineq_constraints(self,x):
        return jnp.array([0],dtype=jnp.float32)


