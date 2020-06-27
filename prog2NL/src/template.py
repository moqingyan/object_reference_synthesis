
attrs = {}
attrs["size"] = ["large", "small"]
attrs["color"] = ["blue", "red", "yellow", "green", "gray", "brown", "purple", "cyan"]
attrs["shape"] = ["cube", "cylinder", "sphere"]
attrs["material"] = ["rubber", "metal"] 

order = ["size", "color", "material", "shape"]

pos = {}
pos["front"] = " in front of "
pos["behind"] = " behind "
pos["left"] = " on the left side of "
pos["right"] = " on the right side of "

vowels = ['a', 'e', 'i', 'o', 'u']
var_num = 3

def neg(clause):
    neg_cls = []
    if clause[0] == "front":
        neg_cls = ["behind", clause[2], clause[1]]
    elif clause[0] == "behind":
        neg_cls = ["front", clause[2], clause[1]]
    elif clause[0] == "left":
        neg_cls = ["right", clause[2], clause[1]]
    elif clause[0] == "right":
        neg_cls = ["left", clause[2], clause[1]]
    else:
        raise Exception("clause not found")
    return neg_cls

def get_article(object_reference, var, occured):
    
    if var == 0 or occured:
        article = "the "
    else:
        if object_reference[var][0] in vowels:
            article = "an "
        else: 
            article = "a "

    return article

def describe_attrs(prog):
    description = {}
    for i in range(var_num):
        description[i] = []

    for clause in prog:
        if type(clause[1]) == str:
            if clause[2] not in description.keys():
                description[clause[2]] = []
            description[clause[2]].append((clause[0], clause[1])) 

    object_reference = {}
    for i in range(var_num):
        object_reference[i] = ""
    
    for var, attrs in description.items():

        has_shape = False

        for o in order:
            for attr in attrs:
                if attr[0] == o:
                    if o == "shape":
                        has_shape = True
                        object_reference[var]+=(attr[1])
                    else:
                        object_reference[var]+=(attr[1] + " ")
                    
        
        if not has_shape:
            object_reference[var] += "object"

        

    return object_reference

def get_object(clause):
    obj = set()
    if type(clause[1]) == int:
        obj.add(clause[1])
    if type(clause[2]) == int:
        obj.add(clause[2])
    return list(obj)

def describe(prog):
    sentence = ""
    object_reference = describe_attrs(prog)

    # Process the positions
    access = set([0])
    to_process = list(filter(lambda c: type(c[1]) == int, prog))
    pos_prog = []

    # rearrange the prog
    while not (to_process == []):
        c = to_process.pop()
        if (c[1] in access):
            pos_prog.append(c)
            access.add(c[2])
        elif (c[2] in access):
            pos_prog.append(neg(c))
            access.add(c[1])
        else:
            to_process.append(c)

    # generate the sentence
    parts = []
    occurred = []
    curr_var = -1


    for clause in pos_prog:
        part = ""
        if not (curr_var == clause[1]):
            part += get_article(object_reference, clause[1], clause[1] in occurred)
            part += (object_reference[clause[1]])
            occurred.append(clause[1])
            curr_var = clause[1]

        part += (pos[clause[0]])

        part += get_article(object_reference, clause[2], clause[2] in occurred)
        part += (object_reference[clause[2]])
        occurred.append(clause[2])
        parts.append(part)

    sentence = " and ".join(parts)
    return sentence

prog = [["color", "green", 0], ["behind", 1, 0], ["right", 0, 1]]
print(describe(prog))