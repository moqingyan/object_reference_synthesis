from ..logic.relation import fact
from ..logic.core import isVariable

class Clause:
    def __init__(self, relation, *args):
        self.relation = relation
        self.arguments = tuple(args)
    def convert(self):
        return fact(self.relation, *self.arguments)
    def variables(self):
        # we care about the variables introduced during synthesis, not those buried in python terms
        for argument in self.arguments:
            if isVariable(argument):
                yield argument
    def __str__(self):
        arguments = ", ".join([str(arg) for arg in self.arguments])
        return f"{self.relation}({arguments})"