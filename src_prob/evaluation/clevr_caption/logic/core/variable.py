class Variable:
    def __init__(self, reference):
        self._ref = reference
    
    def __eq__(self, other):
        if isinstance(other, Variable):
            return self._ref == other._ref
        else:
            return False
    
    def __str__(self):
        return str(self._ref)
    
    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash(self._ref)

def isVariable(obj):
    return isinstance(obj, Variable)

def var(v):
    return Variable(v)