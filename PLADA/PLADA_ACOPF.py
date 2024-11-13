import numpy as np
from autograd import grad   # The only autograd function you may ever need
from jax import config
import jax.numpy as jnp
from jax import jit,grad,jacfwd,jacrev
from cyipopt import minimize_ipopt


class PLADA_ACOPF:

    def __init__(self,net,n_buses,G,B,S,x0,bnds_x,alpha,beta,rho,lr,tao,delta):
        #Transform each data structure of the net into jax numpy
        for key,matrix in net.items():
            net[key] = jnp.array(matrix)
        self.net = net
        self.n_buses = n_buses
        self.G = jnp.array(G)
        self.B = jnp.array(B)
        self.S = jnp.array(S)
        self.x0 = jnp.array(x0)
        self.bnds_x = bnds_x
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.lr = lr
        self.tao = tao
        self.delta = delta
        

    #Initialize values
    def algorithm(self,max_iter,tol_constr):
        
        #Initialize values
        self.tol_constr = tol_constr
        self.max_iter = max_iter

        self.x = self.x0

        self.u  = jnp.zeros(len(self.ineq_constraints(self.x)))
        self.lambd = jnp.ones(len(self.ineq_constraints(self.x)))
        self.mu = jnp.zeros(len(self.ineq_constraints(self.x)))
        self.z = (self.lambd - self.mu) / self.alpha

        gamma0 = 0.8

        k = 1
        while k <= max_iter:

            # Perform the minimization
     
            sol_x = self.ipopt_x(self.update_x,self.eq_constraints,self.x,self.bnds_x)
            new_x = sol_x['x']
                
            sol_u = self.ipopt_u(self.update_u,self.u)
            new_u = sol_u['x']

            self.gamma = min(gamma0,(self.rho * self.delta) /(jnp.linalg.norm(self.lambd - self.mu) +1)) 

            new_mu = self.mu  + self.gamma * (self.lambd - self.mu) / self.rho

            new_lambd = new_mu + self.rho * (self.ineq_constraints(new_x) - self.tol_constr + new_u)
            new_z = (1/self.alpha) * (new_lambd  - new_mu)


            
            #Update values
            self.x = new_x
            self.u = new_u
            self.mu = new_mu
            self.lambd = new_lambd
            self.z = new_z 
            
            self.x = new_x
            print(f'\nIteration {k}')


            k += 1

        return self.x
    
    def update_u(self,u):
        u_update = jnp.dot(grad(self.lagrange,argnums=2)(self.x,self.lambd,self.u,self.mu),u - self.u) +  1/(2 * self.tao) * jnp.linalg.norm(u - self.u) ** 2
        return u_update
    
    
    def lagrange(self,x,lambd,u,mu):

        f_l =  self.objective(x) + jnp.dot(lambd,self.ineq_constraints(x) - self.tol_constr + u) - 1/(2 * self.rho) * jnp.linalg.norm(lambd - mu) ** 2
        return f_l

    def update_x(self,x):

        new_x = jnp.dot(grad(self.objective)(self.x),x) + jnp.dot(self.lambd,self.ineq_constraints(x) - self.tol_constr) + 1/(2 * self.lr) * jnp.linalg.norm(x - self.x) ** 2
        return new_x

    def objective(self, x):
        # Assuming self.n_buses and self.net['gencost'] are available
        n_buses = self.n_buses  # Should be an integer
        x_int = np.arange(n_buses, dtype=np.int32)  # Use NumPy array
        generators_idx = self.net['gencost'][:, 0].astype(np.int32)  # NumPy array

        # Use integer constants for lengths
        len_x_int = n_buses  # n_buses is an integer

        # Slice x using known integer length
        pg_full = x[:len_x_int]

        # Index pg_full using NumPy array (static indices)
        pg = pg_full[generators_idx]

        # Extract generator cost coefficients
        gencost_data_r = self.net['gencost'][:, 4:]

        # Convert to NumPy arrays
        a = gencost_data_r[:, 0]
        b = gencost_data_r[:, 1]
        c = gencost_data_r[:, 2]

        # Convert to JAX arrays
        a = jnp.array(a)
        b = jnp.array(b)
        c = jnp.array(c)

        # Compute total cost
        total_c = jnp.sum(a * pg ** 2 + b * pg + c)

        return total_c

       


    def ipopt_u(self,objective,u0):

        config.update("jax_enable_x64",True)

        obj_jit = jit(objective)
        obj_grad = jit(grad(obj_jit))  # objective gradient
        obj_hess = jit(jacrev(jacfwd(obj_jit))) # objective hessian

        res = minimize_ipopt(obj_jit,jac=obj_grad,hess=obj_hess,x0=u0,options={
            "tol": 1e-8,
            "check_derivatives_for_naninf": "yes",                           # Tighten convergence tolerance for better accuracy
            "hessian_approximation": 'limited-memory',
            "max_iter": 5000,                      # Increase max iterations to allow more time for convergence
            "mu_strategy": "adaptive",             # Use adaptive barrier parameter strategy for stability

            })
        return res


    def ipopt_x(self,objective,con_eq,x0,bnds):

        config.update("jax_enable_x64",True)

        obj_jit = jit(objective)
        con_eq_jit = jit(con_eq)
        #con_ineq_jit = jit(con_ineq)

        #build the derivatives and jit them

        obj_grad = jit(grad(obj_jit))  # objective gradient
        obj_hess = jit(jacrev(jacfwd(obj_jit))) # objective hessian
        con_eq_jac = jit(jacfwd(con_eq_jit))  # jacobian
        #con_ineq_jac = jit(jacfwd(con_ineq_jit))  # jacobian
        con_eq_hess = jacrev(jacfwd(con_eq_jit)) # hessian
        #con_eq_hessvp = jit(lambda x, v: con_eq_hess(x) * v[0]) # hessian vector-product
        def con_eq_hessvp(x, v):
            H = con_eq_hess(x)  # H has shape (m, n, n)
            # Compute the weighted sum of the Hessians
            Hv = jnp.tensordot(v, H, axes=1)  # Sum over m constraints
            return Hv  # Returns an array of shape (n, n)
        

        # JIT compile the function
        con_eq_hessvp = jit(con_eq_hessvp)
        #con_ineq_hess = jacrev(jacfwd(con_ineq_jit))  # hessian
        #con_ineq_hessvp = jit(lambda x, v: con_ineq_hess(x) * v[0]) # hessian vector-product


        #constraints
        cons = [
            {
                "type": 'eq',
                'fun': con_eq_jit,
                'jac':con_eq_jac,
                'hess': con_eq_hessvp
            },

        ]
        
        

        res = minimize_ipopt(obj_jit,jac=obj_grad,hess=obj_hess,constraints=cons,x0=x0,bounds=bnds,options={
        "tol": 1e-8,
        "check_derivatives_for_naninf": "yes",                           # Tighten convergence tolerance for better accuracy
        #"constr_viol_tol": 1e-12,               # Reduce constraint violation tolerance to improve feasibility
        "hessian_approximation": 'limited-memory',
        #"obj_scaling_factor": 1e8,
        #"nlp_scaling_max_gradient": 1e8,
        #"limited_memory_max_history": 20,
        #"limited_memory_update_type": "sr1" ,
        #"limited_memory_initialization": "scalar2" ,
        #"acceptable_tol": 1e-7,                # Lower acceptable tolerance for intermediate solutions
        #"acceptable_constr_viol_tol": 1e-10,    # Lower acceptable constraint violation tolerance for intermediate solutions
        "max_iter":3000,                      # Increase max iterations to allow more time for convergence
        "mu_strategy": "adaptive",             # Use adaptive barrier parameter strategy for stability
        #"mu_target": 1e-8,                     # Target a small barrier parameter to focus on feasibility
        #"bound_relax_factor": 1e-10,            # Reduce bound relaxation for stricter constraint adherence
        #"nlp_scaling_method": "gradient-based",# Use gradient-based scaling to handle large variable ranges
        #"compl_inf_tol": 1e-6                  # Set a tighter complementarity tolerance to enforce feasibility
        })

        return res




    def eq_constraints(self,x):

        x_int = jnp.array([i for i in range(self.n_buses)],dtype=jnp.int32)
        x_bound = jnp.array([],dtype=jnp.int32)

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

        x_int = jnp.array([i for i in range(self.n_buses)],dtype=jnp.int32)
        x_bound = jnp.array([],dtype=jnp.int32)

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

    