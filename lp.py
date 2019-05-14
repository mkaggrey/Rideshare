from pulp import *
import numpy as np
import torch
def optimal(state, request):
    """
    Given a state and a request, this method should solve an LP and a solution describing the optimal pricing grid.
    In this case, we define optimal as maxizming both our fairness and profit objectives

    Params---
    state : torch
        The current state of the ridesharing env
    request : torch
        The grid representing incoming requests


     return : torch
        returns a grid of the same size of the state space, where each element contains a pricing scheme that
        maximizes some objective function (probably our fairness and profit objectives as mentioned above)
    """
    return None

def match(env, pricing, request, state=None):

    """
    Matching drivers to riders in order to maximize some objective function
    Params---
    state : numpy.ndarray
        The current state of the ridesharing env (as of right now this is just a vector containing the number
        of drivers in each location, there are n locations)
    request : torch
        The grid representing incoming requests (so an integer k in location (i,j) means there are k people requesting
        to go from location i to location j at this time step)
    pricing: torch
        The grid containing the prices of a ride between locations i and j

    return: None
    """

    #Initialize linear program
    prob = pulp.LpProblem("Ridesharing", pulp.LpMaximize)

    #List of terms to make affine combination
    terms = []

    #List for constraints
    constraints = []

    #Get Willingness to pay
    wtp = env.wtp(pricing)

    #Notice state
    if state is None:
        state = env.state()

    #Iterate over the rows of the request grid
    for ix in range(len(request)):
        depart = []
        for iy in range(len(request[ix])):
            requests = request[ix][iy]
            for i in range(requests):
                #Create Variable
                #if ix != iy:
                #if pricing[ix][iy] <= env.max_pricing()[ix][iy]:
                x = LpVariable("%d %d %d" % (ix, iy, i), lowBound=0, upBound=1, cat="Integer")
                obj_term = ((x, pricing[ix][iy]*wtp[ix][iy]))
                terms.append(obj_term)

                constrain_term = (x, 1)
                depart.append(constrain_term)

        c = pulp.LpConstraint(e=LpAffineExpression(depart), sense=pulp.LpConstraintLE, name=None, rhs=state[ix])
        constraints.append(c)

    #Add the affine expression of variables to the linear program
    prob += pulp.LpAffineExpression(terms)

    #Add the constraints to the linear program
    for c in constraints: prob += c

    prob.solve()

    return matchAdd(prob.variables(),request)


def matchAdd(variables,request):
    additive = np.zeros(shape=request.shape)
    for v in variables:
        sep = v.name.split("_")
        if "dummy" not in sep:
            x, y = int(sep[0]), int(sep[1])
            additive[x][y] += v.varValue
        else:
            x, y = 0,0
    leftover = request - additive
    return (additive, leftover)