# Nonconvex Optimization for the ACOPF problem

This project aims to replicate the results and problem structure from the paper:

**"A Two-level ADMM Algorithm for AC OPF with Global Convergence Guarantees"**

by **Kaizhao Sun** and **Xu Andy Sun**.

Inside the main Jupyter file, we will:

- Implement **Vanilla ADMM** and **Two-level ADMM** algorithms for solving the AC Optimal Power Flow (ACOPF) problem.
- Compute the **equality and inequality constraints** associated with the ACOPF problem.
- Utilize the **IPOPT** solver to handle the optimization updates within the ADMM framework.

## Overview of the AC Optimal Power Flow (ACOPF) Problem

The ACOPF problem is a fundamental optimization task in power systems engineering. It seeks to determine the optimal operating conditions for an electrical power network while satisfying both physical laws and operational constraints.The AC- problem can be formulated as a non-convex optimization problem, aiming to minimize the cost of electricity generation subject to:

- **Power balance equations** (Kirchhoff's laws)
- **Voltage limits** at generation and load buses
- **Thermal limits** on transmission lines
- **Generator limits** (active and reactive power limits)

The objective function typically represents the cost of generation, and the constraints ensure that the grid operates within safe and efficient parameters.

### Objective function

The objective is to minimize the total generation cost:

$$
\begin{align*}
\min_{\{p_i^g\}} \sum_{i \in \mathcal{N}} f_i(p_i^g)
\end{align*}
$$

where $f_i(p_i^g)$ is the cost function of the generator at bus $i$, typically modeled as:

$$
\begin{align*}
f_i(p_i^g) = a_i (p_i^g)^2 + b_i p_i^g + c_i
\end{align*}
$$

**Subject to:**

**Power Balance Equations (Equality Constraints):**

$$
p_i^g - p_i^d = G_{ii}(e_i^2 + f_i^2) + \sum_{j \in \delta(i)} \left[ G_{ij}(e_i e_j + f_i f_j) - B_{ij}(e_i f_j - e_j f_i) \right]
\quad \forall i \in N
$$

$$
q_i^g - q_i^d = B_{ii}(e_i^2 + f_i^2) + \sum_{j \in \delta(i)} \left[ G_{ij}(e_i f_j + e_j f_i) + B_{ij}(e_i e_j - f_i f_j) \right]
\quad \forall i \in N
$$

**Branch Flow Limits (Inequality Constraints)**:

$$
p_{ij}^2 + q_{ij}^2 \leq s_{ij}^2, \quad \forall (i, j) \in \mathcal{E}
$$

Where:

$$
p_{ij} = -G_{ij}(e_i^2 + f_i^2 - e_i e_j - f_i f_j) - B_{ij}(e_i f_j - e_j f_i)
$$

$$
q_{ij} = B_{ij}(e_i^2 + f_i^2 - e_i e_j - f_i f_j) - G_{ij}(e_i f_j - e_j f_i)
$$

**Voltage Magnitude Limits**:

$$
v_i^2 \leq e_i^2 + f_i^2 \leq \bar{v}_i^2, \quad \forall i \in \mathcal{N}
$$

**Generator Capacity Limits**:

$$
\underline{p}_i^g \leq p_i^g \leq \bar{p}_i^g, \quad \underline{q}_i^g \leq q_i^g \leq \bar{q}_i^g, \quad \forall i \in \mathcal{N}
$$

**Variables:**

- $p_i^g,\, q_i^g$: Active and reactive power generation at bus $i$.
- $p_i^d,\, q_i^d$: Active and reactive power demand at bus $i$.
- $e_i,\, f_i$: Real and imaginary parts of the voltage at bus $i$.
- $v_i,\, \bar{v}_i$: Minimum and maximum voltage magnitude limits at bus $i$.
- $\underline{p}_i^g,\, \bar{p}_i^g$: Minimum and maximum active power generation limits at bus $i$.
- $\underline{q}_i^g,\, \bar{q}_i^g$: Minimum and maximum reactive power generation limits at bus $i$.
- $G_{ij},\, B_{ij}$: Conductance and susceptance of the line between buses $i$ and $j$.
- $\mathcal{N}$: Set of buses (nodes) in the network.
- $\delta(i)$: Set of buses connected to bus $i$.
- $\mathcal{E}$: Set of edges (lines) in the network.
- $s_{ij}$: Apparent power flow limit for line $(i, j)$.
- $p_{ij},\, q_{ij}$: Active and reactive power flows from bus $i$ to bus $j$.

**Constraints Explanation**

- **Power Balance Equations (1b, 1c):** Ensure that the generated power meets the demand and adheres to the network's physical laws (Kirchhoff's laws).
- **Branch Flow Limits (1d):** Ensure that the power flow on each transmission line does not exceed its thermal limit.
- **Voltage Magnitude Limits (1e):** Maintain the voltage magnitude at each bus within acceptable operational limits.
- **Generator Capacity Limits (1f):** Ensure that generators operate within their specified capacity ranges.

## IPOPT Solver

**IPOPT (Interior Point OPTimizer)** is a software package designed for large-scale nonlinear optimization problems. It is particularly well-suited for solving the type of nonlinear, nonconvex problems encountered in ACOPF.

## Problem Structure for IPOPT

IPOPT solves optimization problems of the form:

$$
\begin{align*}
\min_{x} \quad & f(x) \\
\text{subject to} \quad & g_{\text{L}} \leq g(x) \leq g_{\text{U}} \\
& x_{\text{L}} \leq x \leq x_{\text{U}}
\end{align*}
$$

**Features of IPOPT**

- **Interior Point Method:** Utilizes a primal-dual interior point algorithm, which is efficient for handling large-scale and sparse problems.
- **Derivative Information:** Can exploit first and second derivatives (gradients and Hessians) provided by the user or estimated via finite differences.
- **Flexibility:** Supports equality and inequality constraints, variable bounds, and can handle nonconvex problems.

## Application in This Notebook

In this notebook, IPOPT will be used to solve the optimization subproblems arising in the ADMM iterations for the ACOPF problem:

- **Local Updates:** During each iteration of ADMM, local optimization problems are solved using IPOPT to update the variables associated with each agent or region.
- **Handling Nonlinear Constraints:** IPOPT efficiently manages the nonlinear equality and inequality constraints inherent in the ACOPF problem.
- **Integration with ADMM:** By integrating IPOPT within the ADMM framework, we aim to leverage its optimization capabilities to achieve convergence to a feasible and optimal solution.

---

**Note:** The Alternating Direction Method of Multipliers (ADMM) algorithms, including the Vanilla and Two-level versions, will be detailed in subsequent sections of the notebook.
