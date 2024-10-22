import jax.numpy as jnp
import numpy as np

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
    
        
    
    def objective(self,x):
        "Global Objective function"

        Ar_xr_arr = []
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


            Ar_xr_arr.append(Ar_xr)

        

        
        z = 1/m * jnp.sum(jnp.array(Ar_xr_arr),axis=0) + (1 / (2 * m * self.rho)) * jnp.sum(self.alpha_arr,axis=0) #update of 
        
        #z_e = z[:len(z) // 2] #
        #z_f = z[:len(z) // 2:] #

        #ej_bar = x[:len(x) // 2]
        #fj_bar = x[len(x) // 2:]


        #Projection 
        return 1/2 * (jnp.linalg.norm(x - z))
        #print(z.shape)
    

    def eq_constraints(self,x):
        return jnp.array([0])

    def ineq_constraints(self,x):
        ej_bar = x[:len(x) // 2]
        fj_bar = x[len(x) // 2:]

        #v_max = self.net['bus'][:,11]
        cons = []

        for idx,x_r in enumerate(self.x_r_list):
            region = idx + 1
            idx_buses_before = self.idx_buses_arr[region][0]
            idx_buses_after = self.idx_buses_arr[region][1]

            x_int = jnp.array(list(self.regions[region][0]))
            x_bound = jnp.array(list(self.regions[region][1]))

            X_int = x_r[:len(x_int) * 4].reshape((4,-1))
            X_bound = x_r[len(x_int) * 4:].reshape((2,-1))    

            ej_r = X_bound[0,:]
            fj_r = X_bound[1,:]
            
            v_max_bound = self.net['bus'][x_bound,:][:,11]
            cons.append(jnp.concatenate([jnp.absolute(ej_r) -  v_max_bound,jnp.absolute(fj_r) -  v_max_bound]))
            
        return jnp.concatenate([cons[0],cons[1]])   
    


