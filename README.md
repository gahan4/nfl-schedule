This repository contains code, data, and results related to an NFL schedule
optimization. It was conceived as a one-ish week project to practice
my skills, and does not add anything novel to the practice of
NFL scheduling.

Note also that this project does not utilize Gurobi, Cplex, or any
other commercial solver. As a result, some shortcuts needed to be taken
for the optimization. As conceived, the problem has in the range of 
20000 variables and 3000 constraints (1000 ineq, 2000 eq). 
