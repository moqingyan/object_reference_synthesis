from .search import greedy, complete
from .caption import top
from .hint import Hint, refine, refineWithEvidence, refineGreedy

def pickBestCaption(captions, target, scene):
    caption, imgSize = None, len(scene.objects)
    # pick caption with smallest image containing target
    for proposal in captions:
        image = set(proposal.evaluate(scene))
        if target in image and len(image) < imgSize:
            caption, imgSize = proposal, len(image)
    # return the best
    return caption