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

def match(state, requests, pricing):
    """
    Matching drivers to riders in order to maximize some objective function
    Params---
    state : torch
        The current state of the ridesharing env
    request : torch
        The grid representing incoming requests
    pricing: torch
        The grid containing the prices of a ride between locations i and j

    return: torch
        A grid, the same size as the current state, which shows how many requests were satisfied.  All requests between
        the same two locations (say location a to f) are identical, so we may simply count how many of these requests
        are satisfied by way of the matching
    """

    return None