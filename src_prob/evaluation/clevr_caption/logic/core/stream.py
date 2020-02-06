from itertools import zip_longest

class Stream:
    # ensure the item we're storing is an iterable if possible
    def __init__(self, iterable):
        try:
            self._iter = iter(iterable)
        except TypeError:
            self._iter = iterable

    # streams are, of course, iterables
    def __iter__(self):
        return self
    
    def __next__(self):
        return next(self._iter)
        
    # specialized constructors or empty and singleton streams
    @classmethod
    def mzero(cls):
        return cls([])

    @classmethod
    def unit(cls, value):
        return cls([value])

    # combine streams - interleave for completeness
    def mplus(self, other):
        def closure():
            for s, o in zip_longest(self, other):
                if s != None:
                    yield s
                if o != None:
                    yield o
        return Stream(closure())

    # effectively monadic bind, if we assume goal : state -> stream(state) and self : stream(state)
    def bind(self, goal):
        try:
            hd = next(self)
            return goal(hd) + (self >> goal)
        except StopIteration:
            return Stream.mzero()

    # alternative syntax
    # a + b == mplus(a, b)
    def __add__(self, other):
        return self.mplus(other)

    # a >> g == bind(a, g)
    def __rshift__(self, other):
        return self.bind(other)