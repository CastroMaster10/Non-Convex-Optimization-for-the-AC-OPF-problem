import jax.numpy as jnp

class GlobalUpdate_ACOPF:

    def __init__(self,net,regions,rho,x_r_list,alpha_arr,idx_buses_arr):
        #Transform each data structure of the net into jax numpy
        for key,matrix in net.items():
            net[key] = jnp.array(matrix)
        self.net = net
        self.regions = regions
        self.rho = rho  
        self.x_r_list = x_r_list
        self.alpha_arr = alpha_arr
        self.idx_buses_arr = idx_buses_arr 
    
        
    
    def update_xbar(self,x):
        "Global Objective function"

        Ar_xr_arr = jnp.array([])
        m = len(self.x_r_list) #Number of machines
        

        for idx,x_r in enumerate(self.x_r_list):
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

            #Dual variable update of region r

            jnp.append(Ar_xr_arr,Ar_xr)
            #print(Ar_xr)
            #print("---------------")
            #print(self.alpha_arr[idx])
        

        
        new_xbar = 1/m * jnp.sum(Ar_xr_arr,axis=0) + (1 / (2 * m * self.rho)) * jnp.sum(self.alpha_arr,axis=0) #update of new_k
        return new_xbar



    def eq_constraints(self,x):
        return  0
    def ineq_constraints(self,x):
        return 0
    


