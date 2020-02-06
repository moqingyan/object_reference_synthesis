from ..logic.core import *
from itertools import chain

def relation(scene, rel, x, y):
    conjuncts = [eq( (x, y), r) for r in scene.relations[rel]]
    return disj(*conjuncts)

def attribute(scene, attr, obj, val):
    attrs = [(obj, getattr(obj, attr)) for obj in scene.objects]
    conjuncts = [eq( (obj, val), r) for r in attrs]
    return disj(*conjuncts)

def exists(scene, obj):
    conjuncts = [eq(obj, o) for o in scene.objects]
    return disj(*conjuncts)

class Relation:
    def __init__(self, key, x, y):
        self.key = key
        self.x, self.y = x, y
    def convert(self, scene):
        return relation(scene, self.key, self.x, self.y)
    def variables(self):
        if isVariable(self.x): yield self.x
        if isVariable(self.y): yield self.y
    def __str__(self):
        if self.key == "left":
            return f"{self.y} <- {self.x}"
        elif self.key == "right":
            return f"{self.x} -> {self.y}"
        elif self.key == "front":
            return f"{self.y} in front of {self.x}"
        elif self.key == "behind":
            return f"{self.y} behind {self.x}"
        return f"{self.key}({self.x}, {self.y})"

class Attribute:
    def __init__(self, attr, obj, val):
        self.attr = attr
        self.obj, self.val = obj, val
    def convert(self, scene):
        return attribute(scene, self.attr, self.obj, self.val)
    def variables(self):
        if isVariable(self.obj): yield self.obj
        if isVariable(self.val): yield self.val
    def __str__(self):
        try:
            return f"{self.obj} is {self.val.name}"
        except AttributeError:
            return f"{self.obj}.{self.attr} == {self.val}"
class Exists:
    def __init__(self, obj):
        self.obj = obj
    def convert(self, scene):
        return exists(scene, self.obj)
    def variables(self):
        if isVariable(self.obj): yield self.obj
    def __str__(self):
        return f"∃{self.obj}"

class Query:
    def __init__(self, resultVar, *conjuncts):
        self.resultVar = resultVar
        self.conjuncts = conjuncts

    def run(self, scene):
        conjuncts = [c.convert(scene) for c in self.conjuncts]
        for state in conj(*conjuncts)(empty()):
            yield reify(self.resultVar, state)

    def __call__(self, scene):
        return self.run(scene)
    
    def variables(self):
        return list(chain(*[c.variables() for c in self.conjuncts]))

    def conjoin(self, conjunct):
        return Query(self.resultVar, *self.conjuncts, conjunct)
    def __and__(self, other):
        return self.conjoin(other)

    def __str__(self):
        return f"rule({self.resultVar}) :- {', '.join([str(c) for c in self.conjuncts])}"

# functions to make manipulation easier
def left(x, y):
    return Relation("left", x, y)
def right(x, y):
    return Relation("right", x, y)
def front(x, y):
    return Relation("front", x, y)
def behind(x, y):
    return Relation("behind", x, y)
def shape(x, v):
    return Attribute("shape", x, v)
def size(x, v):
    return Attribute("size", x, v)
def color(x, v):
    return Attribute("color", x, v)
def material(x, v):
    return Attribute("material", x, v)
def bind(x):
    return Exists(x)

def rule(x, conjuncts):
    return Query(x, *conjuncts)