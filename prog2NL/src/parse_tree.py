import nltk
from nltk.data import find
from bllipparser import RerankingParser

text_ref = "The big things that are behind the second one of the big thing from front and to the right of the first one of the large spheres from left"


text_conj = "The green object on the right side of an object and the object behind the green object"
text_lit = "The green object in front of some object and the object behind the green object"
text = "The target object is a large green cube"
text = "Both of object1 and object3 are on the left of target."
text = "From left to right are apple, banana and cube."
text = "What color is the food?"


parser = RerankingParser.fetch_and_load('WSJ-PTB3', verbose=True)
# rrp.simple_parse("It's that easy.")

best = parser.parse(text)
res = best.get_reranker_best()

print(best.get_reranker_best())
print(best.get_parser_best())
