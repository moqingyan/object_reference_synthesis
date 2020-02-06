from .variable import isVariable

class Substitution:
    def __init__(self, bindings=None):
        if bindings:
            self._dict = dict(bindings)
        else:
            self._dict = {}

    def find(self, key):
        if isVariable(key) and key in self._dict.keys():
            return self.find(self._dict[key])
        else:
            return key
    def __getitem__(self, key):
        return self.find(key)

    # treat as a static object - simply construct a new instance
    def extend(self, key, value):
        maps = dict(self._dict)
        maps[key] = value
        return Substitution(maps)

    def isBound(self, key):
        return key in self._dict.keys()

    def __str__(self):
        pairs = ("{}/{}".format(key, value) for key, value in self._dict.items())
        return "[{}]".format(", ".join(pairs))