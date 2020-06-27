
# CFG grammar = """
# <NP> -> <Det>? <Adj>* N <PP>*
# PP -> P (NP)
# """

class Tree():

    def __init__(self, value, children):
        self.value = value
        self.children = children

    def get_pprint(self, _last=True, _prefix = ""):
        rep = _prefix

        if _last:
            rep += "`- "
        else:
            rep += "|- "
        
        rep += self.value 
        rep += "\n"

        for (i, child) in enumerate(self.children):
            rep += child.get_pprint(((i + 1) == len(self.children)), _prefix + "    ")

        return rep 

    def __str__(self):
        return self.get_pprint()

t3 = Tree("the", [])
t4 = Tree("object", [])
t1 = Tree("<Det>", [t3])
t2 = Tree("<N>", [t4])
t0 = Tree("<NP>", [t1, t2])
print(t0)