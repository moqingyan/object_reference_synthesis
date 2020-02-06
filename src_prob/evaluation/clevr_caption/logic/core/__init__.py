from .goal import conj, disj, cond, eq, fail
from .hoas import bind, query
from .reify import reify
from .state import empty
from .variable import var, isVariable

# recreating part of the interface from muKanren
def run(goal, numResults=None, state=None):
    # construct the state if one not provided
    if state is None:
        state = empty()
    # construct the result iterable
    results = goal(state)
    # if we need a smaller number of states, slice iterable
    if numResults:
        results = islice(stream, numResults)
    # return the results
    yield from results
