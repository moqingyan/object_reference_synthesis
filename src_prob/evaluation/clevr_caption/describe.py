from .clevr import Database
from .caption import greedy, complete

from argparse import ArgumentParser

parser = ArgumentParser("CLEVR Utility - captioning tool")
parser.add_argument("-i", "--index", required=True, type=int)
parser.add_argument("-d", "--dataset", required=True)
parser.add_argument("-s", "--split", default="train", choices=("train", "test", "validation"))
parser.add_argument("--strategy", default="greedy", choices=("greedy", "complete"))

args = parser.parse_args()

# load the data
print(f"Loading dataset at {args.dataset}...")
data = Database(args.dataset)
# pull the relevant scene
scene = getattr(data, args.split)[args.index]
print(f"Image {args.index} loaded. Found {len(scene.objects)} objects to caption.")
print(f"Generating captions using a {args.strategy} strategy.")
# generate captions
for obj in scene.objects:
    print(f"\nGenerating caption for {obj}...")
    if args.strategy == "greedy":
        caption = greedy(obj, scene)
    elif args.strategy == "complete":
        caption = complete(obj, scene)
    else:
        caption = None
    print(f"|\n+->\t{caption}")
