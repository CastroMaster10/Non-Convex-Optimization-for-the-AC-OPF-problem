import jax.numpy as jnp

class ACOPF_Problem:

    def __init__(self,net,regions,region,G,B,S,xbar,rho):
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
    
        
    
    def objective(self,x):
        "Objective function"
        x_int = jnp.array(list(self.regions[self.region][0]))
        x_bound = jnp.array(list(self.regions[self.region][1]))
        alpha = jnp.ones((1,len(x_bound) * 2))

        
        pg = x[:len(x_int)]
        gencost_data_r = self.net['gencost'][x_int, :][:,4:]

        a = gencost_data_r[:,0]
        b = gencost_data_r[:,1]
        c = gencost_data_r[:,2]
        
        #c_r(x)
        total_c = 0
        for i in range(len(x_int)):
            total_c += a[i] *  pg[i] ** 2 + b[i] * pg[i] + c[i]
        

        #<lambda^k,A_r * x^r>
        x_jr = x[4 * len(x_int):]
        alpha_Ax_r = jnp.dot(alpha,x_jr)


        #rho * ||A_r * x^r + B_r * \bar{x}^k||^2
        consensus = x[4 * len(x_int):] - self.xbar[4 * len(x_int)]
        penalty = 1/2 * self.rho * jnp.linalg.norm(consensus)

        return jnp.sum(total_c + alpha_Ax_r + penalty)

    def eq_constraints(self,x):
        "Equality constraints"

        x_int = jnp.array(list(self.regions[self.region][0]))
        x_bound = jnp.array(list(self.regions[self.region][1]))

        X_int = x[:len(x_int) * 4].reshape((4,-1))
        X_bound = x[len(x_int) * 4:].reshape((2,-1))

        #mask = jnp.isin(self.net['bus'][:,0],x_int) 
        pd_int = self.net['bus'][x_int, :][:,2]

        #mask = jnp.isin(self.net['bus'][:,0],x_int) 
        qd_int = self.net['bus'][x_int, :][:,3]


        cons1 =  0
        cons2 =  0

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

            cons1 += cons1_i
            cons2 += cons2_i
        

        return jnp.array([cons1,cons2])

    
    def power_balance_constraints(self,X_int,X_bound,pd_i,qd_i,pg_i,qg_i,ei,fi,bus_idx_i,x_int,x_bound):
        
        #active power constraint
        cons1 = self.G[bus_idx_i][bus_idx_i] * (ei ** 2 + fi ** 2) - pg_i + pd_i

        #Interior buses
        for j in range(X_int.shape[1]): 

            bus_idx_j = x_int[j]
            ej = X_int[2,j]
            fj = X_int[3,j]
            cons1 += self.G[bus_idx_i][bus_idx_j] * (ei * ej + fi * fj) - self.B[bus_idx_i][bus_idx_j] * (ei * fj - ej * fi)
            

        #Boundary Buses
        for j in range(X_bound.shape[1]): 
            bus_idx_j = x_bound[j]
            ej = X_bound[0,j]
            fj = X_bound[1,j]
            cons1 += self.G[bus_idx_i][bus_idx_j] * (ei * ej + fi * fj) - self.B[bus_idx_i][bus_idx_j] * (ei * fj - ej * fi)    


        #reactive power constraint
        cons2 = -self.B[bus_idx_i][bus_idx_i] * (ei ** 2 + fi ** 2) - qg_i + qd_i

        #Interior buses
        for j in range(X_int.shape[1]): 
            bus_idx_j = x_int[j]
            ej = X_int[2,j]
            fj = X_int[3,j]
            cons2 += -self.B[bus_idx_i][bus_idx_j] * (ei * ej + fi * fj) - self.G[bus_idx_i][bus_idx_j] * (ei * fj - ej * fi)

        #Boundary Buses
        for j in range(X_bound.shape[1]): 
            bus_idx_j = x_bound[j]
            ej = X_bound[0,j]
            fj = X_bound[1,j]
            cons2 += -self.B[bus_idx_i][bus_idx_j] * (ei * ej + fi * fj) - self.G[bus_idx_i][bus_idx_j] * (ei * fj - ej * fi)


        return cons1,cons2

    def ineq_constraints(self,x):
        "Inequality constraints"
        x_int = jnp.array(list(self.regions[self.region][0]))
        x_bound = jnp.array(list(self.regions[self.region][1]))

        X_int = x[:len(x_int) * 4].reshape((4,-1))
        X_bound = x[len(x_int) * 4:].reshape((2,-1))

        cons3 = 0
        cons4 = 0

        for i in range(X_int.shape[1]):
            bus_idx_i = x_int[i]
            ei = X_int[2,i]
            fi = X_int[3,i]

            cons3_i,cons4_i = self.thermal_limit_buses(X_int,X_bound,ei,fi,x_int,x_bound,bus_idx_i)
            
            cons3 += cons3_i 
            cons4 += cons4_i 

        
        return jnp.array([cons3,cons4])

    def thermal_limit_buses(self,X_int,X_bound,ei,fi,x_int,x_bound,bus_idx_i):


        cons3 = 0
        for j in range(X_int.shape[1]):

            bus_idx_j = x_int[j]
            ej = X_int[2,j]
            fj = X_int[3,j]
            pij = -self.G[bus_idx_i][bus_idx_j] * (ei ** 2 + fi ** 2 - ei * ej - fi * fj) - self.B[bus_idx_i][bus_idx_j] * (ei * fj - ej * fi)
            qij =  self.B[bus_idx_i][bus_idx_j] * (ei ** 2 + fi ** 2 - ei * ej - fi * fj) - self.G[bus_idx_i][bus_idx_j] * (ei * fj - ej * fi)
            cons3 +=   pij ** 2 + qij ** 2  -(self.S[bus_idx_i][bus_idx_j] ** 2)
        
        cons4 = 0
        for j in range(X_bound.shape[1]):
            bus_idx_j = x_bound[j]
            ej = X_bound[0,j]
            fj = X_bound[1,j]
            pij = -self.G[bus_idx_i][bus_idx_j] * (ei ** 2 + fi ** 2 - ei * ej - fi * fj) - self.B[bus_idx_i][bus_idx_j] * (ei * fj - ej * fi)
            qij =  self.B[bus_idx_i][bus_idx_j] * (ei ** 2 + fi ** 2 - ei * ej - fi * fj) - self.G[bus_idx_i][bus_idx_j] * (ei * fj - ej * fi)
            cons4 +=   pij ** 2 + qij ** 2  -(self.S[bus_idx_i][bus_idx_j] ** 2)

        
        return cons3,cons4

