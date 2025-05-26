
from sage.all import *

def find_equilibrium(equations, variables, constraints=None):
    if constraints is not None:
        equations.append(constraints)
    return solve(equations, variables)

def filter_solutions(solutions):
    valid_solutions = []

    for sol in solutions:
        values = [float(sol[i].rhs()) for i in range(len(sol))]
        if values[0] > values[2] and values[2] > values[1] and values[1] > 0: 
            valid_solutions.append(values)
    return valid_solutions

def get_system_matrices(equations, variables, states):
    A = jacobian(equations, variables)
    return A
     


def get_equilibrium_equations(system_equations,states, control):
    eqn = []
    for i in range(len(system_equations)):
        eqn.append(system_equations[i] == 0)

    solutions = find_equilibrium(eqn, states)
    solution = solutions[0]

    A = jacobian(system_equations, states)
    b = jacobian(system_equations, control)

    return solution, A, b

def solve_for_equilibrium(A, b, equilibrium, solution):
    for i in range(len(solution)):
        equilibrium[f'x{i+1}'] = float(solution[i](u=equilibrium['u']).rhs())
        
    A_e = A(**equilibrium)
    b_e = b(**equilibrium)

    return A_e, b_e