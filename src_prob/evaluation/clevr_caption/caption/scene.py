from ..logic.relation import observe, fact

# extract relevant facts from a scene
def facts(scene):
    # pull out the objects
    for object in scene.objects:
        # for quantification purposes
        yield fact("object", object)

        # pull out attributes
        for (attribute, value) in object.attributes.items():
            yield fact(attribute, object, value)

    # pull out the relations
    for relation in scene.relations.keys():
        for args in scene.relations[relation]:
            yield fact(relation, *args)

def sceneDescription(scene):
    return observe(*facts(scene))