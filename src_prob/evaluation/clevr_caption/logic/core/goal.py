from .stream import Stream
from .unify import UnificationFailure

from functools import reduce

# we operate under the assumption that the closure has type state -> iter(state)
class Goal:
    '''Goal base constructor - converts a closure of type State -> iter(State) to a goal'''
    def __init__(self, closure, representation=None):
        self._closure = closure
        self._repr = representation

    # executing a goal to produce states
    def execute(self, state):
        return self._closure(state)
    
    # self._repr is used for printing of the goal structure
    def __str__(self):
        if self._repr:
            return self._repr
        else:
            return "ClosureGoal"

    # for combining goals
    def _conj(self, *others):
        return AndGoal(self, *others)
    def _disj(self, *others):
        return OrGoal(self, *others)
    
    # minor utility syntax
    def __and__(self, other):
        return self._conj(other)
    def __or__(self, other):
        return self._disj(other)
    def __call__(self, state):
        return self.execute(state)
    
# combination goals
class AndGoal(Goal):
    '''Conjunctive goal constructor - ensures all subgoals are successful'''
    def __init__(self, *args):
        self._arguments = args
    
    def execute(self, state):
        base, *rest = self._arguments
        return reduce(Stream.bind, rest, base(state))
    
    def __str__(self):
        subgoals = " & ".join([str(arg) for arg in self._arguments])
        return f"({subgoals})"
    
class OrGoal(Goal):
    '''Disjunctive goal constructor - ensures at least one subgoal is successful'''
    def __init__(self, *args):
        self._arguments = args
    
    def execute(self, state):
        base, *rest = [arg(state) for arg in self._arguments]
        return reduce(Stream.mplus, rest, base)

    def __str__(self):
        subgoals = " | ".join([str(arg) for arg in self._arguments])
        return f"({subgoals})"

# base goal constructor
class EqGoal(Goal):
    def __init__(self, left, right):
        self._left = left
        self._right = right
    
    def execute(self, state):
        try:
            return Stream.unit(state.unify(self._left, self._right))
        except UnificationFailure:
            return Stream.mzero()

    def __str__(self):
        return f"{str(self._left)} = {str(self._right)}"

# failure goal
class FailureGoal(Goal):
    def __init__(self): pass
    
    def execute(self, state):
        return Stream.mzero()

    def __str__(self):
        return f"Fail"

# Top level helper functions
def conj(*goals):
    return AndGoal(*goals)
def disj(*goals):
    return OrGoal(*goals)
def cond(*clauses):
    return disj(*[conj(*clause) for clause in clauses])

def eq(left, right):
    return EqGoal(left, right)

def fail():
    return FailureGoal()