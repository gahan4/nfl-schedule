#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 15:58:00 2025

@author: neil
"""
from ortools.linear_solver import pywraplp
from scipy.sparse import lil_matrix, csr_matrix, vstack
from scipy.optimize import milp, LinearConstraint, Bounds
import pandas as pd
import numpy as np


def get_optimal_solution(A_eq, A_in, b_eq, b_in, f, verbose=True):
    """
    Given a set of equality and inequality constraints in matrices, as well as an
    objective function vector, uses OR-TOOLS to find an optimal result.

    Parameters
    ----------
    A_in : scipy.sparse.lil_matrix
        Contains inequality constraints, with one constraint per row.
    A_eq : scipy.sparse.lil_matrix
        Contains equality constraints, with one constraint per row.
    b_in : np array
        Right hand side of inequality constraints.
    b_eq : np array
        Right hand side of equality constraints..
    f : np array
        Objective function vector.
    Verbose: bool
        Whether to print CBC solver's ongoing results to console

    Returns
    -------
    opt_sol - Vector with optimal solution.
    opt_objective - Objective function value at optimal solution.

    """
        
    # Convert sparse lil_matrix to csr_matrix
    A_in = A_in.tocsr()
    A_eq = A_eq.tocsr()

    #A_in_nonzero_rows = ~np.all(A_in != 0, axis=1)
    A_in_nonzero_rows = [row for row in range(A_in.shape[0]) if A_in[row].nnz > 0]
    A_in = A_in[A_in_nonzero_rows]
    #A_in = A_in.toarray()
    b_in = b_in[A_in_nonzero_rows]
    A_eq_nonzero_rows = [row for row in range(A_eq.shape[0]) if A_eq[row].nnz > 0]
    A_eq = A_eq[A_eq_nonzero_rows]
    #A_eq = A_eq.toarray()
    b_eq = b_eq[A_eq_nonzero_rows]
    
    print(f"There are {A_in.shape[0]} inequality constraints and {A_eq.shape[0]} equality constraints" )
    
    # Define solver
    solver = pywraplp.Solver.CreateSolver('CBC')
    if verbose:
        solver.EnableOutput()

    x = {}
    for j in range(A_in.shape[1]):
        x[j] = solver.IntVar(lb=0, ub=1, name=f"x{j}")

    objective = solver.Objective()
    for v in range(A_in.shape[1]):
        objective.SetCoefficient(x[v], f[v])
    objective.SetMaximization()
    
    # Define equality constraints
    for r in range(A_eq.shape[0]):
        constraint = solver.RowConstraint(b_eq[r], b_eq[r])
        for v in range(A_eq.shape[1]):
            constraint.SetCoefficient(x[v], A_eq[r, v])
    print("Completed entering equality constraints")
    # Define inequality constraints
    for r in range(A_in.shape[0]):
        constraint = solver.RowConstraint(-solver.infinity(), b_in[r])
        for v in range(A_in.shape[1]):
            constraint.SetCoefficient(x[v], A_in[r, v])
    print("Completed entering inequality constraints")
    
    solver.SetTimeLimit(1000 * 60 * 30)  # units are milliseconds

    status = solver.Solve()
    
    if status == pywraplp.Solver.OPTIMAL:
        print("Solution found")
        opt_sol = [x[i].solution_value() for i in range(len(x))]
        opt_objective = solver.Objective().Value()
        print(f"Problem solved in {solver.wall_time():d} milliseconds")
        print(f"Problem solved in {solver.iterations():d} iterations")
        print(f"Problem solved in {solver.nodes():d} branch-and-bound nodes")

        return opt_sol, opt_objective
        
    else:
        print("Solution not found")
        return None
    
