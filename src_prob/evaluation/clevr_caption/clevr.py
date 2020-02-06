from enum import Enum
from os.path import abspath, join
from json import load

# ENUMERATIONS FOR OBJECT ATTRIBUTES
Size = Enum("Size", "small large")
Color = Enum("Color", "gray blue brown yellow red green purple cyan")
Material = Enum("Material", "rubber metal")
Shape = Enum("Shape", "cube sphere cylinder")

# OBJECT WRAPPER
class CLEVRObject:
    def __init__(self, jsonRep):
        # pull out spatial attributes from jsonRep
        # self.real_coordinates = tuple(jsonRep["3d_coords"])
        # self.pixel_coordinates = tuple(jsonRep["pixel_coords"])
        # self.rotation = jsonRep["rotation"]

        # otherwise, stick all the attributes in a dict
        self.attributes = {
            "size" : Size[jsonRep["size"]],
            "color" : Color[jsonRep["color"]],
            "material" : Material[jsonRep["material"]],
            "shape" : Shape[jsonRep["shape"]]
        }

    def __str__(self):
        size = self.attributes["size"].name
        color = self.attributes["color"].name
        material = self.attributes["material"].name
        shape = self.attributes["shape"].name
        return f"{size} {color} {material} {shape}"

# ENUMERATIONS FOR SCENE ATTRIBUTES
Split = Enum("Split", "train test val")

# UTILITIES FOR SCENE CONSTRUCTION
def assocListToRelation(assoc):
    for src, dests in enumerate(assoc):
        for dest in dests:
            yield (src, dest)

# SCENE WRAPPER
class Scene:
    def __init__(self, jsonRep):
        # pulling some attributes out is straightforward
        # self.split = Split[jsonRep["split"]]
        # self.index = jsonRep["image_index"]
        # self.filename = jsonRep["image_filename"]
        # not sure what these are right now, but ok
        # self.directions = {
        #     k : tuple(jsonRep["directions"][k]) 
        #         for k in ("left", "right", "front", "behind", "below", "above")
        # }
        # the objects
        self.objects = [
            CLEVRObject(objRep) for objRep in jsonRep["objects"]
        ]
        # the relations, storing references to the objects
        self.relations = {
            k : [
                (self.objects[src], self.objects[dest]) 
                    for src, dest in assocListToRelation(jsonRep["relationships"][k])
            ]
            for k in ("left", "right", "front", "behind")
        }

# DATABASE CONNECTION
class Database:
    def __init__(self, rootPath):
        self._root = abspath(rootPath)
        # load the scenes
        with open(join(self._root, "scenes", "CLEVR_train_scenes.json")) as f:
            self.train = [Scene(s) for s in load(f)["scenes"]]
        with open(join(self._root, "scenes", "CLEVR_val_scenes.json")) as f:
            self.validation = [Scene(s) for s in load(f)["scenes"]]

