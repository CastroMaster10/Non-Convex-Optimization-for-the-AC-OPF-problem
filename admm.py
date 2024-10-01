import numpy as np
import torch

def local_update(bus_index, x_m, z, lambda_m, rho, network_data):
    """
    Performs the local update for a given bus.
    """
    # Extract local variables for the bus
    PG_i = x_m['PG_i'][bus_index]
    QG_i = x_m['QG_i'][bus_index]
    V_i = x_m['V_i'][bus_index]
    theta_i = x_m['theta_i'][bus_index]
    
    # Get neighboring buses
    neighbors = network_data['neighbors'][bus_index]
    num_neighbors = len(neighbors)
    
    # Get Pij and Qij for this bus
    Pij_i = x_m['Pij'][bus_index, neighbors]
    Qij_i = x_m['Qij'][bus_index, neighbors]
    
    # Define the local objective function
    def local_objective(vars):
        """
        Computes the local objective for the bus.
        
        Parameters:
        - vars: Numpy array containing [PG_i, QG_i, V_i, theta_i, Pij_i..., Qij_i...]
        
        Returns:
        - total_objective: Scalar value of the objective function.
        """
        # Unpack variables
        PG_i_var, QG_i_var, V_i_var, theta_i_var = vars[:4]
        Pij_i_var = vars[4:4 + num_neighbors]
        Qij_i_var = vars[4 + num_neighbors:]
        
        # Generation cost (quadratic)
        # Handle multiple generators at the same bus
        # If multiple generators exist, sum their costs
        if isinstance(network_data['gen_costs'][bus_index], list):
            cost = 0
            for gen_cost in network_data['gen_costs'][bus_index]:
                a_i = gen_cost['a']
                b_i = gen_cost['b']
                c_i = gen_cost['c']
                # Assuming one generator per 'a_i', 'b_i', 'c_i'
                cost += a_i * PG_i_var**2 + b_i * PG_i_var + c_i
        else:
            a_i = network_data['gen_costs'][bus_index]['a']
            b_i = network_data['gen_costs'][bus_index]['b']
            c_i = network_data['gen_costs'][bus_index]['c']
            cost = a_i * PG_i_var**2 + b_i * PG_i_var + c_i
        
        # Augmented Lagrangian terms
        consensus_violation_P = Pij_i_var - z['Pij'][bus_index, neighbors]
        consensus_violation_Q = Qij_i_var - z['Qij'][bus_index, neighbors]
        consensus_violation = np.concatenate([consensus_violation_P, consensus_violation_Q])
        
        dual_terms = np.sum(
            lambda_m['Pij'][bus_index, neighbors] * consensus_violation_P + 
            lambda_m['Qij'][bus_index, neighbors] * consensus_violation_Q
        )
        
        penalty_terms = (rho / 2.0) * np.sum(consensus_violation**2)
        
        # Compute constraint violations
        constraints = compute_constraints(vars, bus_index, network_data)
        
        # Penalty coefficient for constraints
        penalty_coefficient = network_data.get('penalty_coefficient', 1e5)
        constraint_penalty = penalty_coefficient * np.sum(np.array(constraints)**2)
        
        # Total objective
        total_objective = cost + dual_terms + penalty_terms + constraint_penalty
        
        return total_objective
    
    # Initial guess for optimization
    vars_init = np.concatenate([
        [PG_i, QG_i, V_i, theta_i],
        Pij_i,
        Qij_i
    ])
    
    # Bounds for variables
    bounds = define_variable_bounds(bus_index, neighbors, network_data)
    
    # Prepare arguments for gradient computation
    args = {
        'gen_cost': network_data['gen_costs'][bus_index],
        'lambda_Pij': lambda_m['Pij'][bus_index, neighbors],
        'lambda_Qij': lambda_m['Qij'][bus_index, neighbors],
        'z_Pij': z['Pij'][bus_index, neighbors],
        'z_Qij': z['Qij'][bus_index, neighbors],
        'rho': rho,
        'neighbors': neighbors,
        'penalty_coefficient': network_data.get('penalty_coefficient', 1e5),
        'bus_index': bus_index,
        'network_data': network_data
    }
    
    # Solve the optimization problem
    optimized_vars = simple_gradient_descent(
        objective_func=local_objective,
        vars_init=vars_init,
        constraints_func=compute_constraints,  # Constraints function
        bounds=bounds,
        args=args
    )
    
    # Update local variables
    x_m['PG_i'][bus_index] = optimized_vars[0]
    x_m['QG_i'][bus_index] = optimized_vars[1]
    x_m['V_i'][bus_index] = optimized_vars[2]
    x_m['theta_i'][bus_index] = optimized_vars[3]
    x_m['Pij'][bus_index, neighbors] = optimized_vars[4:4 + num_neighbors]
    x_m['Qij'][bus_index, neighbors] = optimized_vars[4 + num_neighbors:]


def synchronize(x_m, z, lambda_m, rho, network_data):
    """
    Performs the synchronization step (global update) to update shared variables z.
    """
    num_buses = network_data['num_buses']
    for i in range(num_buses):
        for j in network_data['neighbors'][i]:
            # Update Pij shared variable
            z['Pij'][i, j] = (x_m['Pij'][i, j] + x_m['Pij'][j, i]) / 2.0
            # Update Qij shared variable
            z['Qij'][i, j] = (x_m['Qij'][i, j] + x_m['Qij'][j, i]) / 2.0
            # Ensure symmetry
            z['Pij'][j, i] = z['Pij'][i, j]
            z['Qij'][j, i] = z['Qij'][i, j]

def update_dual_variables(x_m, z, lambda_m, rho, network_data):
    """
    Updates the dual variables lambda_m.
    """
    num_buses = network_data['num_buses']
    for i in range(num_buses):
        for j in network_data['neighbors'][i]:
            # Update lambda for Pij
            lambda_m['Pij'][i, j] += rho * (x_m['Pij'][i, j] - z['Pij'][i, j])
            # Update lambda for Qij
            lambda_m['Qij'][i, j] += rho * (x_m['Qij'][i, j] - z['Qij'][i, j])


def check_convergence(x_m, x_m_prev, tol):
    """
    Checks if the algorithm has converged by comparing the current and previous values of x_m.
    """
    diff = 0
    for key in x_m.keys():
        diff += np.linalg.norm(x_m[key] - x_m_prev[key])
    return diff < tol



def admm_algorithm(x_m_init, network_data, rho=1.0, max_iter=100, tol=1e-4):
    """
    Implements the ADMM algorithm.
    """
    # Initialize variables
    x_m = x_m_init.copy()
    num_buses = network_data['num_buses']
    
    # Initialize shared variables z
    z = {
        'Pij': np.zeros((num_buses, num_buses)),
        'Qij': np.zeros((num_buses, num_buses))
    }
    
    # Initialize dual variables lambda_m
    lambda_m = {
        'Pij': np.zeros((num_buses, num_buses)),
        'Qij': np.zeros((num_buses, num_buses))
    }
    
    # Iterative process
    for iteration in range(max_iter):
        print(f"Iteration {iteration+1}")
        
        # Store previous x_m for convergence check
        x_m_prev = x_m.copy()
        
        # Local updates
        for i in range(num_buses):
            local_update(i, x_m, z, lambda_m, rho, network_data)
        
        # Synchronization (global update)
        synchronize(x_m, z, lambda_m, rho, network_data)
        
        # Dual variable update
        update_dual_variables(x_m, z, lambda_m, rho, network_data)
        
        # Check convergence
        if check_convergence(x_m, x_m_prev, tol):
            print("Convergence achieved.")
            break
    
    return x_m, z, lambda_m



def define_variable_bounds(bus_index, neighbors, network_data):
    """
    Defines bounds for the optimization variables for a given bus.

    Parameters:
    - bus_index: Index of the bus.
    - neighbors: List of neighboring bus indices.
    - network_data: Dictionary containing network data.

    Returns:
    - bounds: List of tuples specifying (min, max) bounds for each variable.
    """
    gen_limits = network_data['gen_limits'].get(bus_index, [])
    voltage_limits = network_data['voltage_limits'][bus_index]
    
    bounds = []
    
    if gen_limits:
        # Assuming one generator per bus; modify if multiple generators exist
        # If multiple generators, you need to handle each one separately
        PG_min = gen_limits[0]['PG_min']
        PG_max = gen_limits[0]['PG_max']
        bounds.append((PG_min, PG_max))
    
        QG_min = gen_limits[0]['QG_min']
        QG_max = gen_limits[0]['QG_max']
        bounds.append((QG_min, QG_max))
    else:
        # No generator at this bus
        bounds.append((0.0, 0.0))  # PG_i
        bounds.append((0.0, 0.0))  # QG_i
    
    # Bounds for V_i
    V_min = voltage_limits['V_min']
    V_max = voltage_limits['V_max']
    bounds.append((V_min, V_max))
    
    # Bounds for theta_i (typically between -pi and pi)
    theta_min = -np.pi
    theta_max = np.pi
    bounds.append((theta_min, theta_max))
    
    # Bounds for Pij_i and Qij_i
    for _ in range(2 * len(neighbors)):
        bounds.append((None, None))  # No specific bounds; adjust as needed
    
    return bounds


def compute_constraints(vars, bus_index, network_data):
    """
    Computes the constraints for the local optimization problem.
    """
    # Unpack variables
    num_neighbors = len(network_data['neighbors'][bus_index])
    PG_i, QG_i, V_i, theta_i = vars[:4]
    Pij_i = vars[4:4+num_neighbors]
    Qij_i = vars[4+num_neighbors:]
    
    # Power balance equations and other constraints
    constraints = []
    # Compute power injections, flows, etc.
    # Implement the AC power flow equations for the bus
    # For illustration purposes, we'll assume the constraints are satisfied
    # In practice, you need to implement the actual equations
    
    # Return a list of constraint values
    return constraints



def simple_gradient_descent(objective_func, vars_init, constraints_func, bounds, args, max_iter=100, alpha=0.01):
    """
    A simple gradient descent optimizer that incorporates constraints.
    
    Parameters:
    - objective_func: The objective function to minimize.
    - vars_init: Initial guess for the variables (numpy array).
    - constraints_func: Function that computes the constraints.
    - bounds: List of tuples specifying (min, max) bounds for each variable.
    - args: Dictionary containing additional arguments required by objective and constraints functions.
    - max_iter: Maximum number of iterations for the optimizer.
    - alpha: Learning rate.
    
    Returns:
    - vars: Optimized variables as a numpy array.
    """
    vars = vars_init.copy()
    
    for iteration in range(max_iter):
        # Compute the objective value
        cost = objective_func(vars)
        
        # Compute the gradient
        grad = compute_gradient(objective_func, vars, args)
        
        # Update variables using gradient descent
        vars = vars - alpha * grad
        
        # Project variables onto their bounds
        lower_bounds = np.array([b[0] if b[0] is not None else -np.inf for b in bounds])
        upper_bounds = np.array([b[1] if b[1] is not None else np.inf for b in bounds])
        vars = np.clip(vars, lower_bounds, upper_bounds)
        
        # Compute constraint violations
        constraint_vals = constraints_func(vars, args['bus_index'], args['network_data'])
        
        # Check if all constraints are satisfied within the tolerance
        if np.all(np.abs(constraint_vals) < 1e-4):
            print(f"Converged in {iteration+1} iterations.")
            break
    else:
        print(f"Reached maximum iterations ({max_iter}) without full convergence.")
    
    return vars



def compute_gradient(objective_func, vars, args):
    """
    Computes the gradient of the objective function with respect to vars.

    Parameters:
    - objective_func: The objective function to compute the gradient of.
    - vars: The current values of the variables (numpy array).
    - args: A dictionary containing additional arguments needed for the objective function.

    Returns:
    - grad: The gradient vector (numpy array).
    """
    # Unpack variables
    PG_i, QG_i, V_i, theta_i = vars[:4]
    num_neighbors = len(args['neighbors'])
    Pij_i = vars[4:4 + num_neighbors]
    Qij_i = vars[4 + num_neighbors:]

    # Initialize gradient vector
    grad = np.zeros_like(vars)

    # Extract necessary data from args
    gen_cost = args['gen_cost']
    lambda_Pij = args['lambda_Pij']
    lambda_Qij = args['lambda_Qij']
    z_Pij = args['z_Pij']
    z_Qij = args['z_Qij']
    rho = args['rho']

    # Gradient w.r.t. PG_i
    if isinstance(gen_cost, list):
        # Multiple generators at this bus
        grad_PG = 0
        for gc in gen_cost:
            grad_PG += 2 * gc['a'] * PG_i + gc['b']
        grad[0] = grad_PG
    else:
        # Single generator
        a_i = gen_cost['a']
        b_i = gen_cost['b']
        grad[0] = 2 * a_i * PG_i + b_i

    # Gradient w.r.t. QG_i
    # Assuming no cost term involving QG_i
    grad[1] = 0.0

    # Gradient w.r.t. V_i and theta_i
    grad[2] = 0.0  # V_i
    grad[3] = 0.0  # theta_i

    # Gradient w.r.t. Pij_i
    for idx in range(num_neighbors):
        grad[4 + idx] = lambda_Pij[idx] + rho * (Pij_i[idx] - z_Pij[idx])

    # Gradient w.r.t. Qij_i
    for idx in range(num_neighbors):
        grad[4 + num_neighbors + idx] = lambda_Qij[idx] + rho * (Qij_i[idx] - z_Qij[idx])

    return grad







