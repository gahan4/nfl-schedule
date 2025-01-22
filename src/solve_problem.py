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
from cvxopt import matrix
from cvxopt import glpk
from cvxopt.glpk import ilp
#import pulp
#import pyomo.environ as pyo
from rpy2 import robjects
from rpy2.robjects import numpy2ri


def get_optimal_solution(A_eq, A_in, b_eq, b_in, f, teams,
                         num_teams = 32, num_stadiums = 32,
                         num_weeks = 18, num_slots = 1):
    """
    

    Parameters
    ----------
    A_in : TYPE
        DESCRIPTION.
    A_eq : TYPE
        DESCRIPTION.
    b_in : TYPE
        DESCRIPTION.
    b_eq : TYPE
        DESCRIPTION.
    f : TYPE

    Returns
    -------
    None.

    """
    
    # Define solver
    #solver = pywraplp.Solver.CreateSolver('CBC')
    
    #num_variables = len(f)
    
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
    
    #np.savetxt("results/A_in.csv", A_in.toarray(), delimiter=",")
    #np.savetxt("results/A_eq.csv", A_eq.toarray(), delimiter=",")
    #np.savetxt("results/b_in.csv", b_in, delimiter=",")
    #np.savetxt("results/b_eq.csv", b_eq, delimiter=",")
    #np.savetxt("results/f.csv", f, delimiter=",")

    
    #prob = pulp.LpProblem("Binary_ILP", pulp.LpMaximize)
    
    #num_variables = len(f)
    #x = pulp.LpVariable.dicts("x", range(num_variables), cat="Binary")
    
    # Objective function: Maximize f*x
    #prob += pulp.lpSum(f[i] * x[i] for i in range(num_variables))
    
    # Constraints
    # A_eq @ x = b_eq (equality constraints)
    #equality_constraints = []
    #for i in range(len(b_eq)):
    #    constraint = pulp.lpSum(A_eq[i, j] * x[j] for j in range(num_variables)) == b_eq[i]
    #    equality_constraints.append(constraint)
    #    print(i)
    #prob += pulp.lpSum(equality_constraints)
    
    # A_in @ x <= b_in (inequality constraints)
    #for i in range(len(b_in)):
    #    prob += pulp.lpSum(A_in[i, j] * x[j] for j in range(num_variables)) <= b_in[i]
    
    
    # Solve the problem
    #prob.solve()
    
    #if pulp.LpStatus[prob.status] == 'Optimal':
    #    solution = [pulp.value(x[i]) for i in range(num_variables)]
    #else:
    #    print("No Solution found")

    #model = pyo.ConcreteModel()
    
    #model.x = pyo.Var(range(num_variables), within=pyo.Binary)
    
    #model.obj = pyo.Objective(expr=sum(f[i] * model.x[i] for i in range(num_variables)), sense=pyo.maximize)
    
    #model.ineq_constraints = pyo.ConstraintList()
    #for i in range(A_in.shape[0]):
    #    print(i)
    #    model.ineq_constraints.add(sum(A_in[i,j] * model.x[j] for j in range(num_variables)) <= b_in[i])


    # Create variables
    #x = [solver.IntVar(lb = 0.0, ub = 1.0, name = f"x_{v}") for v in range(num_variables)]
    #x = cp.Variable(num_variables, boolean=True)
    #for i in range(num_teams):
    #    for j in range(num_stadiums):
    #        for k in range(num_weeks):
    #            for l in range(num_slots):
       #             team_desc = teams.loc[teams['team_id'] == i, 'team_abbr'].iloc[0]
      #              stadium_desc = teams.loc[teams['team_id'] == j, 'team_abbr'].iloc[0]
     #               slot_desc = slots.loc[slots['slot_id'] == l, 'slot_desc'].iloc[0]
     #               week_desc = str(k + 1)
     #                x[i, j, k, l] = solver.IntVar(lb=0.0, ub=1.0, name = f"x_{i}_{j}_{k}_{l}") #, 
                                                 #name=f"{team_desc}_{stadium_desc}_{slot_desc}_{week_desc}")
    
        
    '''numpy2ri.activate()
    r = robjects.r
    r('library(Rsymphony)')
    r_A_eq = robjects.r['matrix'](A_eq, nrow=A_eq.shape[0], ncol=A_eq.shape[1])
    r_A_in = robjects.r['matrix'](A_in, nrow=A_in.shape[0], ncol=A_in.shape[1])
    r_b_eq = robjects.FloatVector(b_eq.flatten())  # Flattening in case it's 2D
    r_b_in = robjects.FloatVector(b_in.flatten())
    r_f = robjects.FloatVector(f.flatten())

    glpk.options['tm_lim'] = 1000
    glpk.options['msg_lev'] = 'GLP_MSG_ON'
    result = ilp(c = -1*matrix(f),
                 G = matrix(A_in),
                 h = matrix(b_in),
                 A = matrix(A_eq),
                 b = matrix(b_eq),
                 B = set(range(A_eq.shape[1])))'''
    
    # Define objective funtion
    '''objective = solver.Objective()
    for v in range(num_variables):
        objective.SetCoefficient(x[v], f[v])
    #objective.SetMaximization()
    #objective = cp.Maximize(f @ x)
    
    # Define equality constraints
    for r_eq in range(A_eq_csr.shape[0]):
        print(r_eq)
        constraint = solver.Constraint(b_eq[r_eq], b_eq[r_eq])
        for v in range(num_variables):
            if A_eq_csr[r_eq, v] != 0:
                constraint.SetCoefficient(x[v], A_eq_csr[r_eq, v])
    # Define inequality constraints
    for r_in in range(A_in_csr.shape[0]):
        constraint = solver.Constraint(-solver.infinity(), b_in[r_in])
        for v in range(num_variables):
            if A_in_csr[r_in, v] != 0:
                constraint.SetCoefficient(x[v], A_in_csr[r_in, v])
    
    status = solver.Solve()
    
    if status == pywraplp.Solver.OPTIMAL:
        print("Solution found")
        optimal_solution = x.solution_value()
        return optimal_solution
        
    else:
        print("Solution not found")
        return None'''
    
    # combine constraints...
    constraints = LinearConstraint(A= vstack([A_in, A_eq]), 
                                   lb=np.concatenate((-np.ones_like(b_in)*np.inf, b_eq)), 
                                   ub=np.concatenate((b_in, b_eq)))
    
        
    result = milp(
        c=f, # objective function
        integrality=np.ones_like(f), # all variables are integers
        bounds=Bounds(lb=0,ub=1),
        constraints=constraints,
        options={"time_limit":10000000,
                 "mip_rel_gap": 0.1}
        )
    
    return result
