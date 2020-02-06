from clevr import Database
from caption import Hint, refine, refineWithEvidence, pickBestCaption, refineGreedy

from argparse import ArgumentParser

parser = ArgumentParser("CLEVR Utility - hint refining tool")
parser.add_argument("-i", "--index", required=True, type=int)
parser.add_argument("-d", "--dataset", required=True)
parser.add_argument("-s", "--split", default="train", choices=("train", "test", "validation"))
parser.add_argument('hints', metavar='Hints', type=str, nargs='+',
                    help='list of hints to be refined')
parser.add_argument("--strategy", default="simple", choices=("simple", "evidence", "greedy"))

args = parser.parse_args()

# convert the hints
hints = [Hint[h] for h in args.hints]

# load the scene
print(f"Loading dataset at {args.dataset}...")
data = Database(args.dataset)
scene = getattr(data, args.split)[args.index]
print(f"Image {args.index} loaded. Found {len(scene.objects)} objects to caption.")

# generate captions
print(f"Generating captions from hint using the {args.strategy} strategy.")
for obj in scene.objects:
    print(f"\nGenerating caption for {obj}...")
    if args.strategy == "simple":
        refinements = refine(hints)
        caption = pickBestCaption(refinements, obj, scene)
    elif args.strategy == "evidence":
        refinements = refineWithEvidence(hints, obj, scene)
        caption = pickBestCaption(refinements, obj, scene)
    elif args.strategy == "greedy":
        refinements = refineGreedy(hints, obj, scene)
        caption = pickBestCaption(refinements, obj, scene)
    else:
        caption = None
    print(f"|\n+->\t{caption}")