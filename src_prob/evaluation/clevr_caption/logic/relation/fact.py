from ..core.stream import Stream
from ..core.goal import Goal
from ..core import disj, eq

class Fact(Goal):
    # relation can be anything hashable
    def __init__(self, relation, *arguments):
        self.relation = relation
        self.arguments = list(arguments)
    
    # facts themselves are goals
    def execute(self, state):
        # if the state doesn't have the relation extension, our job is pretty easy
        if not state.hasExtension("relation"):
            return Stream.mzero()
        # if it does, we need to pull out all ground facts and check that we can unify against one
        ground = state["relation"][self.relation]
        disjuncts = [eq(self.arguments, groundFact) for groundFact in ground]
        return disj(*disjuncts)(state)

    def __str__(self):
        return f"{self.relation}({', '.join([str(arg) for arg in self.arguments])})"

# function for construction
def fact(rel, *args):
    return Fact(rel, *args)