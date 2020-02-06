from .variable import isVariable
from .substitution import Substitution

from collections.abc import Iterable

# unlike unification, this shouldn't get called often, so recursion is probably fine here
def reify(term, s):
    if not isinstance(s.substitution, Substitution):
        print(s)
        raise Exception()
    sub = s.substitution
    # if term is a var, just look it up in s
    if isVariable(term) and sub.isBound(term):
        return reify(sub[term], s)
    # if we can represent term by a dict, recurse on the dict
    elif hasattr(term, "__dict__"):
        d = reify(term.__dict__, s)
        if d == term.__dict__:
            return term
        else:
            obj = object.__new__(type(term))
            obj.__dict__.update(d)
            return obj
    # if term _is_ a dict, recurse on values
    elif isinstance(term, dict):
        return {k : reify(v, s) for k, v in term.items()}
    # if term is a tuple, map reifying over everything
    elif isinstance(term, tuple):
        return tuple(reify(list(term), s))
    # if term is a list, map reifying over everything
    elif isinstance(term, list):
        return [reify(st, s) for st in term]
    # otherwise, just pass the term back as-is
    else:
        return term
    