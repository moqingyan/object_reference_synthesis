from .substitution import Substitution
from .variable import Variable
from .unify import unify

from itertools import count
from copy import deepcopy

class State:
    '''State objects maintain all constraints necessary to satisfy goals'''

    # we treat states as immutable
    __slots__ = ("substitution", "variables", "_extensions")
    def __init__(self):
        self.variables = set()
        self.substitution = Substitution()
        self._extensions = {}
    def clone(self):
        return deepcopy(self)

    # when using HOAS, we want to be able to make fresh variable easily
    def freshVariable(self):
        proposals = (Variable("?.{}".format(i)) for i in count())
        for prop in proposals:
            if prop not in self.variables:
                return prop
    def addFreshVariable(self):
        var = self.freshVariable()
        result = self.clone()
        result.variables.add(var)
        return var, result

    # TODO - also print extension information
    def __str__(self):
        try:
            summaries = ", ".join([f"{key}: {ext.summary()}" for key, ext in self._extensions.items()])
        except:
            summaries = ""
        if summaries:
            return f"{self.substitution} | {summaries}"
        else:
            return f"{self.substitution}"

    # performs unification in the context of the state
    # TODO - modify so that unify takes advantage of extensions
    # not needed now, as the only constraints we build are unification constraints
    # to add numeric / negation / interval, etc., unification must produce diff. constraints
    def unify(self, left, right):
        result = self.clone()
        result.substitution = unify(left, right, self.substitution)
        return result

    # extension interface
    def hasExtension(self, key):
        return key in self._extensions.keys()
    def registerExtension(self, key, ext):
        self._extensions[key] = ext
    def __getitem__(self, key):
        return self._extensions[key]

def empty():
    return State()