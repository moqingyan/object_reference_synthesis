import clevr
import sqlite3
from time import time
from numpy import uint32

from argparse import ArgumentParser

parser = ArgumentParser("CLEVR Utility - database construction")
parser.add_argument("-i", "--input", required=True)
parser.add_argument("-o", "--output", required=True)
parser.add_argument("-c", "--count", type=int, default=1000)
parser.add_argument("-s", "--split", default="train", choices=("train", "test", "val"))

args = parser.parse_args()

# if the split is test, exit out - no ground truth provided there
if args.split == "test":
    print("No ground truth available for test split")
    exit(-1)

# otherwise, we're in training or validation, and that's fine
# load the data
print("Loading data...")
db = clevr.Database(args.input)

# get scenes
print("Getting scenes...")
if args.split == "train":
    try:
        scenes = db.train[:args.count]
    except IndexError:
        print("Not enough scenes in training split")
elif args.split == "val":
    try:
        scenes = db.validation[:args.count]
    except IndexError:
        print("Not enough scenes in validation split")

# construct database with relevant schema - hardcoded for now
print("Initializing database at {}...", args.output)
conn = sqlite3.connect(args.output)
cursor = conn.cursor()

attributes = ["shape", "size", "material", "color"]
relations = ["left", "right", "front", "behind"]

# make object table
cursor.execute('''CREATE TABLE objects (scene text, object text)''')

# make attribute tables
for attribute in attributes:
    cursor.execute(f'''CREATE TABLE {attribute} (object text, value text)''')

# make relation tables
for relation in relations:
    cursor.execute(f'''CREATE TABLE {relation} (source text, destination text)''')

# save the structure
conn.commit()

# get id for an object
def getID(obj):
    return uint32(hash(obj))

# function for importing scenes
def importScene(scene):
    for obj in scene.objects:
        key = getID(obj)
        cursor.execute(f'''INSERT into objects values ({scene.index}, {key})''')
        for attribute in attributes:
            cursor.execute(f'''INSERT into {attribute} values ({key}, '{getattr(obj, attribute).name}')''')
    for relation in relations:
        for (src, dest) in scene.relations[relation]:
            cursor.execute(f'''INSERT into {relation} values ({getID(src)}, {getID(dest)})''')
    conn.commit()

# now start loading up scenes
print("Importing scenes...")
scene, *tail = scenes

startTime = time()
importScene(scene)
elapsed = time() - startTime

print(f"First scene took {elapsed} seconds. Expected total time is {elapsed * args.count} seconds...")

for scene in tail:
    importScene(scene)
print("Done. Database constructed.")

# and at the end of the day, close
cursor.close()