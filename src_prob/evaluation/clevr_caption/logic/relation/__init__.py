from ..core.state import State
from ..core import conj
from .extension import RelationExtension
from .fact import fact
# relation interface is quite simple:
# 1 - observations provide a mechanism for declaring intensional values

# NB - observe is not a goal, as evaluation order matters quite a bit
# instead, observe just creates/extends a state with the observations
def observe(*facts, state=None):
    # make sure we have a state with a relation extension
    if state is None:
        state = State()
    else:
        state = state.clone()
    if not state.hasExtension("relation"):
        state.registerExtension("relation", RelationExtension())
    
    # then register each of the facts
    for fact in facts:
        state["relation"].addFact(fact)

    # and finally, return the state
    return state

# extra syntax for defining clauses
def clause(*facts):
    return conj(*facts)
