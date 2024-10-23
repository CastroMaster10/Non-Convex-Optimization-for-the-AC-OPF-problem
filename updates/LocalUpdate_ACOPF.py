import jax.numpy as jnp
import numpy as np
import jax

class LocalUpdate_ACOPF:

    def __init__(self,net,regions,region,G,B,S,xbar,rho,alpha,idx_buses_before,idx_buses_after):

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
        self.alpha = alpha
        self.idx_buses_before = idx_buses_before
        self.idx_buses_after = idx_buses_after
        
    
    def objective(self,x):
        "Local Objective function"
        x_int = jnp.array(list(self.regions[self.region][0]))
        x_bound = jnp.array(list(self.regions[self.region][1]))
               
        #X_int = x.reshape((4,-1))[:,x_int]
        #X_bound = x.reshape((4,-1))[:,x_bound][2:,:]
        X_int = x[:len(x_int) * 4].reshape((4,-1))
        X_bound = x[len(x_int) * 4:].reshape((2,-1))        
        
        pg = x[:len(x_int)]
        gencost_data_r = self.net['gencost'][x_int, :][:,4:]

        a = gencost_data_r[:,0]
        b = gencost_data_r[:,1]
        c = gencost_data_r[:,2]
            
        #c_r(x)
        total_c = 0
        for i in range(len(x_int)):
            total_c += a[i] *  pg[i] ** 2 + b[i] * pg[i] + c[i]

        
        X_bound_v = X_bound.reshape(-1)
        X_bound_v_e = X_bound_v[:len(x_bound)]
        X_bound_v_f = X_bound_v[len(x_bound):]



        Ar_xr_e = jnp.concatenate([jnp.zeros(self.idx_buses_before),X_bound_v_e,jnp.zeros(self.idx_buses_after)])
        Ar_xr_f = jnp.concatenate([jnp.zeros(self.idx_buses_before),X_bound_v_f,jnp.zeros(self.idx_buses_after)])


        Ar_xr = jnp.concatenate([Ar_xr_e,Ar_xr_f])
        #Arx^r -  xbar
        consensus = Ar_xr - self.xbar
        #penalty
        alpha_Ax_r = jnp.dot(self.alpha,consensus)
        penalty =  self.rho /2 * jnp.linalg.norm(consensus)

        return jnp.sum(total_c + alpha_Ax_r + penalty)
    
    def eq_constraints(self,x):
        "Equality constraints"

        x_int = jnp.array(list(self.regions[self.region][0]))
        x_bound = jnp.array(list(self.regions[self.region][1]))

        X_int = x[:len(x_int) * 4].reshape((4,-1))
        X_bound = x[len(x_int) * 4:].reshape((2,-1))


        pd_int = self.net['bus'][x_int, :][:,2]
        qd_int = self.net['bus'][x_int, :][:,3]


        cons1 =  []
        cons2 =  []

        #start with calculating the cosntraints
        for i in range(X_int.shape[1]):
            bus_idx_i = x_int[i]
            pg_i = X_int[0,i]
            qg_i = X_int[1,i]
            ei = X_int[2,i]
            fi = X_int[3,i]
            pd_i = pd_int[i]
            qd_i = qd_int[i]


            cons1_i,cons2_i = self.power_balance_constraints(X_int,X_bound,pd_i,qd_i,pg_i,qg_i,ei,fi,bus_idx_i,x_int,x_bound)

            #cons1 += cons1_i
            #cons2 += cons2_i
            cons1.append(cons1_i)
            cons2.append(cons2_i)


        return jnp.concatenate([jnp.array(cons1),jnp.array(cons2)])

        
    def power_balance_constraints(self,X_int,X_bound,pd_i,qd_i,pg_i,qg_i,ei,fi,bus_idx_i,x_int,x_bound):

        #active power constraint
        cons1 = jnp.array(self.G[bus_idx_i][bus_idx_i] * (ei ** 2 + fi ** 2) - pg_i + pd_i,dtype=jnp.float32)
        #print(cons1)

        #Interior buses
        for j in range(X_int.shape[1]): 
            
            bus_idx_j = x_int[j]
            ej = X_int[2,j]
            fj = X_int[3,j]

            def sum_interior_buses_True(cons1): 
                delta = self.G[bus_idx_i][bus_idx_j] * (ei * ej + fi * fj) - self.B[bus_idx_i][bus_idx_j] * (ei * fj - ej * fi)
                return cons1 + jnp.array(delta,dtype=jnp.float32)

            def sum_interior_buses_False(cons1):
                return cons1

            cons1 = jax.lax.cond(bus_idx_i != bus_idx_j,sum_interior_buses_True,sum_interior_buses_False,cons1)

                

        #Boundary Buses
        for j in range(X_bound.shape[1]):
            bus_idx_j = x_bound[j]
            ej = X_bound[0,j]
            fj = X_bound[1,j]

            def sum_boundary_buses_True(cons1): 
                delta = self.G[bus_idx_i][bus_idx_j] * (ei * ej + fi * fj) - self.B[bus_idx_i][bus_idx_j] * (ei * fj - ej * fi)
                return cons1 + jnp.array(delta,dtype=jnp.float32)

            def sum_boundary_buses_False(cons1):
                return cons1

            cons1 = jax.lax.cond(bus_idx_i != bus_idx_j,sum_boundary_buses_True,sum_boundary_buses_False,cons1)


        #reactive power constraint
        cons2 = jnp.array(-self.B[bus_idx_i][bus_idx_i] * (ei ** 2 + fi ** 2) - qg_i + qd_i,dtype=jnp.float32)
        #print(type(cons2))
        #Interior buses
        for j in range(X_int.shape[1]):
            bus_idx_j = x_int[j]
            ej = X_int[2,j]
            fj = X_int[3,j]
            def sum_interior_buses_True(cons2): 
                delta = -self.B[bus_idx_i][bus_idx_j] * (ei * ej + fi * fj) -self.G[bus_idx_i][bus_idx_j] * (ei * fj - ej * fi)
                return cons2 + jnp.array(delta,dtype=jnp.float32)

            def sum_interior_buses_False(cons2):
                return cons2

        
            cons2 = jax.lax.cond(bus_idx_i != bus_idx_j,sum_interior_buses_True,sum_interior_buses_False,cons2) 


        #Boundary Buses
        for j in range(X_bound.shape[1]):
            
            bus_idx_j = x_bound[j]
            ej = X_bound[0,j]
            fj = X_bound[1,j]
            def sum_boundary_buses_True(cons2): 
                delta = -self.B[bus_idx_i][bus_idx_j] * (ei * ej + fi * fj) -self.G[bus_idx_i][bus_idx_j] * (ei * fj - ej * fi)
                return cons2 + jnp.array(delta,dtype=jnp.float32)

            def sum_boundary_buses_False(cons2):
                return cons2

        
            cons2 = jax.lax.cond(bus_idx_i != bus_idx_j,sum_boundary_buses_True,sum_boundary_buses_False,cons2) 


        return cons1,cons2

    def ineq_constraints(self,x):
        "Inequality constraints"
        x_int = jnp.array(list(self.regions[self.region][0]))
        x_bound = jnp.array(list(self.regions[self.region][1]))

        #X_int = x.reshape((4,-1))[:,x_int]
        #X_bound = x.reshape((4,-1))[:,x_bound][2:,:]

        X_int = x[:len(x_int) * 4].reshape((4,-1))
        X_bound = x[len(x_int) * 4:].reshape((2,-1))

        cons3 = []
        cons4 = []
        cons5 = []
        cons6 = []

        #voltage limits
        Vmax = self.net['bus'][:,11]
        Vmin = self.net['bus'][:,12]

        for i in range(X_int.shape[1]):
            bus_idx_i = x_int[i]
            ei = X_int[2,i]
            fi = X_int[3,i]
            cons3_i,cons4_i = self.thermal_limit_buses(X_int,X_bound,ei,fi,x_int,x_bound,bus_idx_i)
            #cons3 += cons3_i 
            #cons4 += cons4_i 
            #cons5 +=  Vmin[bus_idx_i] ** 2 -(ei ** 2)  - (fi ** 2)
            #cons6 +=   ei ** 2  + fi ** 2 - (Vmax[bus_idx_i] ** 2)
            cons3.append(cons3_i)
            cons4.append(cons4_i)
            cons5.append(Vmin[bus_idx_i] ** 2 -(ei ** 2)  - (fi ** 2))
            cons6.append(ei ** 2  + fi ** 2 - (Vmax[bus_idx_i] ** 2))




        return jnp.concatenate([jnp.array(cons3),jnp.array(cons4),jnp.array(cons5),jnp.array(cons6)])

    def thermal_limit_buses(self,X_int,X_bound,ei,fi,x_int,x_bound,bus_idx_i):


        cons3 = jnp.array(0.0, dtype=jnp.float32)  # or dtype=jnp.float64
        #print(type(cons3))
        for j in range(X_int.shape[1]):
            bus_idx_j = x_int[j]
            ej = X_int[2,j] 
            fj = X_int[3,j]

            def sum_interior_buses_True(cons3):
                pij = -self.G[bus_idx_i][bus_idx_j] * (ei ** 2 + fi ** 2 - ei * ej - fi * fj) - self.B[bus_idx_i][bus_idx_j] * (ei * fj - ej * fi)
                qij =  self.B[bus_idx_i][bus_idx_j] * (ei ** 2 + fi ** 2 - ei * ej - fi * fj) - self.G[bus_idx_i][bus_idx_j] * (ei * fj - ej * fi)
                delta = pij ** 2 + qij ** 2  -(self.S[bus_idx_i][bus_idx_j] ** 2)
                return cons3 + jnp.array(delta,dtype=jnp.float32)
            
            
            def sum_interior_buses_False(cons3):
                return cons3
            
            cons3 = jax.lax.cond(bus_idx_i != bus_idx_j,sum_interior_buses_True,sum_interior_buses_False,cons3) 
            
            
        cons4 = jnp.array(0.0, dtype=jnp.float32)
        #print(type(cons4))
        for j in range(X_bound.shape[1]):
            
            bus_idx_j = x_bound[j]
            ej = X_bound[0,j]
            fj = X_bound[1,j]

            def sum_boundary_buses_True(cons4):
                pij = -self.G[bus_idx_i][bus_idx_j] * (ei ** 2 + fi ** 2 - ei * ej - fi * fj) - self.B[bus_idx_i][bus_idx_j] * (ei * fj - ej * fi)
                qij =  self.B[bus_idx_i][bus_idx_j] * (ei ** 2 + fi ** 2 - ei * ej - fi * fj) - self.G[bus_idx_i][bus_idx_j] * (ei * fj - ej * fi)
                delta = pij ** 2 + qij ** 2  -(self.S[bus_idx_i][bus_idx_j] ** 2)
                return cons4 + jnp.array(delta,dtype=jnp.float32)
            
            
            def sum_boundary_buses_False(cons4):
                return cons4
            
            cons4 = jax.lax.cond(bus_idx_i != bus_idx_j,sum_boundary_buses_True,sum_boundary_buses_False,cons4) 
        

            
        return cons3,cons4
