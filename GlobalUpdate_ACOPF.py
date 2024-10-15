import jax.numpy as jnp

class GlobalUpdate_ACOPF:

    def __init__(self,net,regions,rho,x_r_list,alpha):
        #Transform each data structure of the net into jax numpy
        for key,matrix in net.items():
            net[key] = jnp.array(matrix)
        self.net = net
        self.regions = regions
        #self.xbar = xbar
        self.rho = rho  
        self.x_r_list = x_r_list
        self.alpha = alpha
        #self.idx_xbar_r_list =  idx_xbar_r
    
        
    
    def objective(self):
        "Global Objective function"

        idx_xbar_r = 0
        consensus_list = []
        #alpha_Bx = jnp.array(len(self.x_r_list))
        #penalty = jnp.array(len(self.x_r_list))

        for idx,x_r in enumerate(self.x_r_list):
            region = idx + 1 #1 based-index
            
            x_int = jnp.array(list(self.regions[region][0]))
            x_bound = jnp.array(list(self.regions[region][1]))

            alpha_e_r = self.alpha[:len(self.alpha) // 2][idx_xbar_r: idx_xbar_r + len(x_bound)]
            alpha_j_r = self.alpha[len(self.alpha) // 2:][idx_xbar_r:idx_xbar_r + len(x_bound)]
        
            alpha_r = jnp.concatenate([alpha_e_r,alpha_j_r])
            Ar_xr_k = x_r[4 * len(x_int):]
            consensus = Ar_xr_k + alpha_r / self.rho 
            consensus_list.append(jnp.array(consensus))

            #<lambda^k,Br * x>
            #alpha_Bx= jnp.dot(,x)
            #<lambda^k,A_r *  * x^{r,   k+1}>
            

            #rho * ||A_r * x^r + B_r * \bar{x}^k||^2
            
            #e_jbar_r = self.x[:len(x) // 2][idx_xbar_r: idx_xbar_r + len(x_bound)]
            #f_jbar_r = self.x[len(x) // 2:][idx_xbar_r: idx_xbar_r + len(x_bound)]

            #Br_xbar =  jnp.concatenate([e_jbar_r,f_jbar_r])
            #consensus = Ax_jr  - Br_xbar 
            #penalty = 1/2 * self.rho * jnp.linalg.norm(consensus)

            idx_xbar_r += len(x_bound)

        
        x_bar_k = 1 / len(self.x_r_list) * jnp.concatenate([consensus_list[0],consensus_list[1]])
        return x_bar_k


    def ineq_constraints(self,x):
        return 0
    


