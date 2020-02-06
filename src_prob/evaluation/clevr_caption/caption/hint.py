from enum import Enum
from ..logic.core import var
from ..clevr import Size, Shape, Material, Color
from itertools import permutations, chain, count
from .clause import Clause
from .caption import top

# hint enumeration - coarse-grained choices during search
Hint = Enum("Hint", "left right front behind size shape color material")

# for i/o purposes
def stringToHint(str):
    return Hint[str]

# defining split b/t relations and attributes
def isRelation(hint):
    return hint in [
        Hint.left,
        Hint.right,
        Hint.front,
        Hint.behind
    ]
def isAttribute(hint):
    return not isRelation(hint)

# given a hint and arguments, refine to get clauses
def refineRelation(hint, x, y):
    if hint is Hint.left:
        yield Clause("left", x, y)
    elif hint is Hint.right:
        yield Clause("right", x, y)
    elif hint is Hint.front:
        yield Clause("front", x, y)
    elif hint is Hint.behind:
        yield Clause("behind", x, y)

def refineAttribute(hint, x):
    if hint is Hint.size:
        for size in Size:
            yield Clause("size", x, size)
    elif hint is Hint.shape:
        for shape in Shape:
            yield Clause("shape", x, shape)
    elif hint is Hint.color:
        for color in Color:
            yield Clause("color", x, color)
    elif hint is Hint.material:
        for material in Material:
            yield Clause("material", x, material)

# for constructing fresh variables
def freshVariable(variables):
    for i in count(0):
        proposal = var(f"_.{i}")
        if proposal not in variables:
            return proposal

def refineHint(hint, variablesInUse):
    # split hint based on relation/attribute
    if isRelation(hint):
        # generate the canonical fresh variable
        fresh = freshVariable(variablesInUse)
        # generate all permutations of variables
        for x, y in permutations(chain(variablesInUse, [fresh]), 2):
            yield from refineRelation(hint, x, y)
    # if we're an attribute, don't need fresh var
    else:
        for x in variablesInUse:
            yield from refineAttribute(hint, x)

def refineCaption(caption, hint):
    variables = set(caption.variables())
    for clause in refineHint(hint, variables):
        yield caption & clause

def refine(hints):
    captions = [ top() ]
    # refine for each hint
    for hint in hints:
        captions = chain(*map(lambda c: refineCaption(c, hint), captions))
    # then spit out the result
    yield from captions

def refineWithEvidence(hints, target, scene):
    captions = [ top() ]
    for hint in hints:
        proposals = chain(*map(lambda c: refineCaption(c, hint), captions))
        captions = filter(lambda c: target in c.evaluate(scene), proposals)
    yield from captions

def greedyPick(captions, target, scene):
    caption, imgSize = None, len(scene.objects)
    # pick caption with smallest image containing target
    for proposal in captions:
        image = set(proposal.evaluate(scene))
        if target in image and len(image) < imgSize:
            caption, imgSize = proposal, len(image)
    # return the best
    return caption

def refineGreedy(hints, target, scene):
    caption = top()

    for hint in hints:
        try:
            caption = greedyPick(refineCaption(caption, hint), target, scene)
        except:
            pass

    yield caption
