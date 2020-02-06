from json import load, dump
from os.path import join
from argparse import ArgumentParser

parser = ArgumentParser("CLEVR Utility - data subsampling")
parser.add_argument("-i", "--input", required=True)
parser.add_argument("-o", "--output", required=True)
parser.add_argument("-c", "--count", type=int, default=1000)

args = parser.parse_args()

with open(args.input) as f:
    json = load(f)

output = {
    "info": json["info"],
    "scenes": json["scenes"][:args.count]
}

with open(args.output, 'w') as f:
    dump(output, f)
