from ..logic.core import conj, reify, var
from ..logic.relation import fact

from itertools import chain

from .scene import sceneDescription

class Caption:
    def __init__(self, objectRef, clauses):
        self._ref = objectRef
        self.clauses = clauses
        self.timeout = False
    
    def __str__(self):
        clauses = [str(c) for c in self.clauses]
        return f"{self._ref} <- {', '.join(clauses)}"

    def __len__(self):
        return len(self.clauses)

    def variables(self):
        yield self._ref
        for clause in self.clauses:
            yield from clause.variables()

    def size(self):
        return len(self._clauses)

    def conjoin(self, *clauses):
        newClauses = list(chain(clauses, self.clauses))
        return Caption(self._ref, newClauses)
        
    def __and__(self, other):
        return self.conjoin(other)

    def evaluate(self, scene):
        # construct the query
        clauses = [clause.convert() for clause in self.clauses]
        query = conj(fact("object", self._ref), *clauses)
        # and the initial state
        state = sceneDescription(scene)
        # and get the results
        for result in query(state):
            yield reify(self._ref, result)

# simple constructor picks the reference for us
def top():
    return Caption(var("x"), [])