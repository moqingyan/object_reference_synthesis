from ..core import eq, conj
from collections import defaultdict

class RelationExtension:
    
    # simple association map for observations
    def __init__(self):
        self.facts = defaultdict(lambda: [])
    
    # only GROUND facts should be added, although we aren't really checking that...
    def addFact(self, fact):
        self.facts[fact.relation].append(fact.arguments)

    def summary(self):
        numFactsPerRelation = [
            len(self.facts[relation]) for relation in self.facts.keys()
        ]
        return f"{sum(numFactsPerRelation)} facts"

    # easy access to the facts map
    def __getitem__(self, key):
        return self.facts[key]