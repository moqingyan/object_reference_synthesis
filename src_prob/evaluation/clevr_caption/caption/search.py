from heapq import *
from itertools import permutations, count

from ..clevr import Size, Shape, Material, Color

from .clause import Clause
from .caption import top
from .query import *

def score(caption):
    return caption.size()

def freshVariable(caption):
    variables = set(caption.variables())
    for i in count(0):
        proposal = var(f"_.{i}")
        if proposal not in variables:
            return proposal

def proposeClauses(caption):
    variables = set(caption.variables())
    # we'll build a list then yield to filter by existing clauses
    clauses = []

    # add a new relation tieing existing variables together
    if len(variables) >= 2:
        for x, y in permutations(variables, 2):
            clauses.extend([
                Clause("left", x, y),
                Clause("right", x, y),
                Clause("front", x, y),
                Clause("behind", x, y)
            ])
    
    # add a relation introducing a new variable
    fresh = freshVariable(caption)
    for var in variables:
        for x, y in permutations([fresh, var], 2):
            clauses.extend([
                Clause("left", x, y),
                Clause("right", x, y),
                Clause("front", x, y),
                Clause("behind", x, y)
            ])

    # add a relation restricting a concrete attribute
    for x in variables:
        # add sizes
        for size in Size:
            clauses.append(Clause("size", x, size))
        # add shapes
        for shape in Shape:
            clauses.append(Clause("shape", x, shape))
        # add colors:
        for color in Color:
            clauses.append(Clause("color", x, color))
        # add materials
        for material in Material:
            clauses.append(Clause("material", x, material))

    # now, yield as appropriate
    for clause in clauses:
        if clause not in caption.clauses:
            yield clause


def containsSolution(rule, obj, scene):
    img = set(rule(scene))
    return obj in img
def imgSize(rule, scene):
    return len(set(rule(scene)))

# greedily select the "best" proposal
def greedySelectClause(caption, target, scene):
    output, outputImageSize = None, len(scene.objects)
    # check each proposed clause
    for clause in proposeClauses(caption):
        updatedCaption = caption & clause
        image = set(updatedCaption.evaluate(scene))
        # if the new updated caption is better, select the clause
        if target in image and len(image) < outputImageSize:
            output, outputImageSize = clause, len(image)
    return output

# greedily search
def greedy(target, scene, timeout=10):
    # construct initial solution and target image
    solution = top()
    targetImage = set([target])
    # until we cant...
    while True:
        print( f" current solution len: {len(solution)}")
        # get a new clause
        clause = greedySelectClause(solution, target, scene)
        if clause is not None:
            solution = solution & clause
            image = set(solution.evaluate(scene))
            if image == targetImage:
                break
            if len(solution.clauses) > timeout:
                solution.timeout = True
                break
        else:
            break
    return solution

# complete search
def getPriority(caption):
    return (score(caption), hash(caption), caption)

def complete(target, scene):
    # effectively just a frontier search using heapq
    frontier = [getPriority(top())]
    while frontier != []:
        _, _, caption = heappop(frontier)

        # evaluate
        image = set(caption.evaluate(scene))
        # if we've found a solution, return
        if image == set([target]):
            return rule
        # if we haven't already excluded the object of desire...
        if target in image:
            # generate a bunch of proposals and plug them in to the frontier
            for clause in proposeClauses(caption):
                proposal = caption & clause
                heappush(frontier, getPriority(proposal))
    return None