from .goal import Goal
from .variable import var
from .reify import reify

from inspect import signature
from collections import namedtuple

# wrapper for higher-order abstract syntax - a useful paradigm for avoiding naming variables
class BindGoal(Goal):
    ''' Bind Goal - converts a closure that constructs a goal from variables to a goal'''
    def __init__(self, closure):
        # use introspection to get the parameters from the closure
        # helps with printing, allows us to keep the constructor variadic
        self._parameters = tuple(signature(closure).parameters.keys())
        self._closure = closure

    def execute(self, state):
        # get a fresh variable from the state for every variable in the closure's parameters
        arguments = []
        for param in self._parameters:
            x, state = state.addFreshVariable()
            arguments.append(x)
        # construct the goal
        goal = self._closure(*arguments)
        # evalute under the state
        return goal.execute(state)

    def __str__(self):
        # construct a variable from each parameter, and construct the goal from the closure
        arguments = [var(x) for x in self._parameters]
        goal = self._closure(*arguments)
        argumentStrings = ", ".join([str(x) for x in arguments])
        # return the string rep of the goal
        return str(f"∃{argumentStrings}.{goal}")

# importantly, not a goal - produces an iterable of python objects, not a stream of states
class Query:
    # closure maps variables to a goal
    def __init__(self, closure):
        self._parameters = tuple(signature(closure).parameters.keys())
        self._closure = closure

        # to construct results, we look at the parameters - if there's just one, we return it
        if len(self._parameters) == 0:
            raise ValueError()
        elif len(self._parameters) == 1:
            self._mkAnswer = lambda result: result
        else:
            Result = namedtuple("Result", " ".join(self._parameters))
            self._mkAnswer = lambda *result: Result(*results)

    def __str__(self):
        arguments = [var(x) for x in self._parameters]
        goal = self._closure(*arguments)
        argumentStrings = ", ".join([str(x) for x in arguments])
        return str(f"{argumentStrings} <- {goal}")

    # we keep the calling interface the same so that we can use the same RUN interface at the top-level
    def __call__(self, state):
        return self.execute(state)

    def execute(self, state):
        arguments = []
        for _ in self._parameters:
            x, state = state.addFreshVariable()
            arguments.append(x)
        goal = self._closure(*arguments)
        for result in goal(state):
            yield self._mkAnswer(*[reify(x, result) for x in arguments])

# top level wrapper around the goal
def bind(closure):
    return BindGoal(closure)
def query(closure):
    return Query(closure)