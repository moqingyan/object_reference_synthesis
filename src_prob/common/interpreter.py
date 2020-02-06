import time
import numpy as np

# Represents a set of ground tuples, e.g., extracted from
# an image.
#
# unary: {str: [int]}
# binary: {str: [(int, int)]} ({name: (o0, o1)} maps a relation
#                              with the given name to the list of
#                              pairs of objects that satisfy that
#                              relation)
class GroundTuples:
    def __init__(self, unary, binary):
        self.unary = unary
        self.binary = binary

    def __str__(self):
        s = ''
        for name in self.unary:
            for object in self.unary[name]:
                s += '{}(o{})\n'.format(name, object)
        for name in self.binary:
            for (object0, object1) in self.binary[name]:
                s += '{}(o{}, o{})\n'.format(name, object0, object1)
        return s[:-1]

# Represents a variable binding.
#
# binding: [int] (binding[i] is the object bound to variable i)
class Binding:
    def __init__(self, binding):
        self.binding = binding

    def __str__(self):
        return '{' + ', '.join(['v{}: o{}'.format(var, object) for var, object in enumerate(self.binding)]) + '}'

# Represents the bindings of individual variables.
#
# marginal_binding: np.array([n_vars, n_objects], dtype=np.int)
class MarginalBinding:
    def __init__(self, marginal_binding):
        self.marginal_binding = marginal_binding

    def __str__(self):
        return '{' + ', '.join(['v{}: {}'.format(i, '[' + ', '.join(['o{}'.format(j) for j, object in enumerate(var) if object == 1]) + ']') for i, var in enumerate(self.marginal_binding)]) + '}'

# Represents a unary clause name(var).
#
# name: str
# var: int
class UnaryClause:
    def __init__(self, name, var):
        self.name = name
        self.var = var

    # Checks whether the given binding satisfies this clause
    # in the context of the given ground tuples.
    #
    # ground: GroundTuples
    # binding: Binding
    def check(self, ground, binding):
        return binding.binding[self.var] in ground.unary[self.name]

    def to_tuple(self):
        return (self.name, self.var)

    def __str__(self):
        return '{}(v{})'.format(self.name, self.var)

# Represents a binary clause name(var0, var1).
#
# name: str
# var0: int
# var1: int
class BinaryClause:
    def __init__(self, name, var0, var1):
        self.name = name
        self.var0 = var0
        self.var1 = var1

    # Checks whether the given binding satisfies this clause
    # in the context of the given ground tuples.
    #
    # ground: GroundTuples
    # binding: Binding
    def check(self, ground, binding):
        return (binding.binding[self.var0], binding.binding[self.var1]) in ground.binary[self.name]

    def to_tuple(self):
        return (self.name, self.var0, self.var1)

    def __str__(self):
        return '{}(v{}, v{})'.format(self.name, self.var0, self.var1)

# n_vars: int (number of variables)
# n_objects: int (number of objects)
# return: [[int]]
def _build_bindings(n_vars, n_objects):
    bindings = [[]]
    for _ in range(n_vars):
        next_bindings = []
        for binding in bindings:
            for object in range(n_objects):
                next_bindings.append(binding + [object])
        bindings = next_bindings
    return [Binding(binding) for binding in bindings]

# unary_names: [str]
# binary_names: [str]
# n_vars: int
# return: [UnaryClause | BinaryClause]
def _build_clauses(unary_names, binary_names, n_vars):
    clauses = []
    for name in unary_names:
        for var in range(n_vars):
            clauses.append(UnaryClause(name, var))
    for name in binary_names:
        for var0 in range(n_vars):
            for var1 in range(n_vars):
                clauses.append(BinaryClause(name, var0, var1))
    return clauses

# ground: GroundTuples
# ground_uncertain: GroundTuples
# n_vars: int (number of variables)
# n_objects: int (number of objects)
def _build(ground, ground_uncertain, n_vars, n_objects):
    bindings = _build_bindings(n_vars, n_objects)
    unary_names = set(ground.unary.keys()).union(set(ground_uncertain.unary.keys()))
    binary_names = set(ground.binary.keys()).union(set(ground_uncertain.binary.keys()))
    clauses = _build_clauses(unary_names, binary_names, n_vars)

    clause_lookup = {}
    for i, clause in enumerate(clauses):
        clause_lookup[clause.to_tuple()] = i

    binding_lookup = np.zeros([n_vars, n_objects, len(bindings)])
    for var in range(n_vars):
        for object in range(n_objects):
            for j, binding in enumerate(bindings):
                if binding.binding[var] == object:
                    binding_lookup[var, object, j] = 1

    semantics_lookup = np.zeros([len(clauses), len(bindings)], dtype=np.int)
    for i, clause in enumerate(clauses):
        for j, binding in enumerate(bindings):
            # no ground tuple should be both certain and uncertain
            if clause.check(ground, binding) and clause.check(ground_uncertain, binding):
                raise Exception()
            if clause.check(ground, binding):
                semantics_lookup[i, j] = 1
            if clause.check(ground_uncertain, binding):
                semantics_lookup[i, j] = 2

    return clauses, clause_lookup, bindings, binding_lookup, semantics_lookup

# Represents a program.
#
# clauses: [UnaryClause | BinaryClause]
class Program:
    def __init__(self, clauses):
        self.clauses = clauses

    def __str__(self):
        return ' /\ '.join([str(clause) for clause in self.clauses])

# The interpreter works by using the mapping
#
#   T -> 1
#   F -> 0
#   ? -> 2
#
# Truth table for conjunction in 3-valued logic:
#
#   & T F ?
#   T T F ?
#   F - F F
#   ? - - ?
#
# Clipped multiplication table (i.e., np.clip(s * s', 0, 2)):
#
#   * 1 0 2
#   1 1 0 2
#   0 - 0 0
#   2 - - 2
#
# Note that this table is equivalent to the truth table for conjunction in 3-valued logic.
# Thus, the mapping is a homomorphism.
#
class Interpreter:
    def __init__(self, ground, ground_uncertain, n_vars, n_objects):
        clauses, clause_lookup, bindings, binding_lookup, semantics_lookup = _build(ground, ground_uncertain, n_vars, n_objects)
        self.clauses = clauses
        self.clause_lookup = clause_lookup
        self.bindings = bindings
        self.binding_lookup = binding_lookup
        self.semantics_lookup = semantics_lookup

    def reset(self):
        return np.ones(len(self.bindings), dtype=np.int)

    def step(self, state, clause, is_uncertain):
        if is_uncertain:
            return np.clip(state * self.semantics_lookup[self.clause_lookup[clause.to_tuple()]], 0, 2)
        else:
            return state * self.semantics_lookup[self.clause_lookup[clause.to_tuple()]]

    def get_bindings(self, state, is_uncertain):
        b_star = 2 if is_uncertain else 1
        bindings = []
        for j, b in enumerate(state):
            if b == b_star:
                bindings.append(self.bindings[j])
        return bindings

    # self.binding_lookup: np.array([n_vars, n_objects, n_bindings])
    # state: np.array([n_bindings])
    # is_uncertain: bool
    def get_marginal_bindings(self, state, is_uncertain):
        state = (state == (2 if is_uncertain else 1)).astype(np.int)
        return MarginalBinding((np.tensordot(self.binding_lookup, state, ((2), (0))) > 0).astype(np.int))

    def is_clause_useful(self, state, clause, is_uncertain):
        return np.sum(np.square(self.step(state, clause, is_uncertain) - state)) != 0

    def is_state_correct(self, target_object, state, is_uncertain):
        if is_uncertain:
            state_certain = (state == 1).astype(np.int)
            marginal_target_binding_certain = (np.matmul(self.binding_lookup[0], state_certain) > 0).astype(np.int)
            state_uncertain = (state == 2).astype(np.int)
            marginal_target_binding_uncertain = (np.matmul(self.binding_lookup[0], state_uncertain) > 0).astype(np.int)
            return marginal_target_binding_certain[target_object] == 1 and marginal_target_binding_certain.sum() == 1 and marginal_target_binding_uncertain[target_object] == marginal_target_binding_uncertain.sum()
        else:
            marginal_target_binding = (np.matmul(self.binding_lookup[0], state) > 0).astype(np.int)
            return marginal_target_binding[target_object] == 1 and marginal_target_binding.sum() == 1

def execute(interpreter, program, is_uncertain):
    state = interpreter.reset()
    for h, clause in enumerate(program.clauses):
        state = interpreter.step(state, clause, is_uncertain)
    return state

def synthesize(interpreter, target_object, state, is_uncertain, max_depth):
    # search depth 1
    for i, c_i in enumerate(interpreter.clauses):
        state_i = interpreter.step(state, c_i, is_uncertain)
        if interpreter.is_state_correct(target_object, state_i, is_uncertain):
            return Program([c_i])

    if max_depth <= 1:
        return None

    # search depth 2
    for i, c_i in enumerate(interpreter.clauses):
        state_i = interpreter.step(state, c_i, is_uncertain)
        for j, c_j in enumerate(interpreter.clauses):
            state_ij = interpreter.step(state_i, c_j, is_uncertain)
            if interpreter.is_state_correct(target_object, state_ij, is_uncertain):
                return Program([c_i, c_j])

    if max_depth <= 2:
        return None

    # search depth 3
    for i, c_i in enumerate(interpreter.clauses):
        state_i = interpreter.step(state, c_i, is_uncertain)
        for j, c_j in enumerate(interpreter.clauses):
            state_ij = interpreter.step(state_i, c_j, is_uncertain)
            for h, c_h in enumerate(interpreter.clauses):
                state_ijh = interpreter.step(state_ij, c_h, is_uncertain)
                if interpreter.is_state_correct(target_object, state_ijh, is_uncertain):
                    return Program([c_i, c_j, c_h])

    if max_depth <= 3:
        return None

    # search depth 4
    for i, c_i in enumerate(interpreter.clauses):
        state_i = interpreter.step(state, c_i, is_uncertain)
        for j, c_j in enumerate(interpreter.clauses):
            state_ij = interpreter.step(state_i, c_j, is_uncertain)
            for h, c_h in enumerate(interpreter.clauses):
                state_ijh = interpreter.step(state_ij, c_h, is_uncertain)
                for k, c_k in enumerate(interpreter.clauses):
                    state_ijhk = interpreter.step(state_ijh, c_k, is_uncertain)
                    if interpreter.is_state_correct(target_object, state_ijhk, is_uncertain):
                        return Program([c_i, c_j, c_h, c_k])

    return None

if __name__ == '__main__':
    # Step 1: Parameters
    n_vars = 3
    n_objects = 7

    # Step 2: Grounding
    unary = {'large': [0, 1, 2, 3, 4, 5, 6, 7], 'small': [], 'red': [0, 2], 'blue': [1], 'yellow': [], 'green': [], 'purple': [], 'gray': [], 'brown': [], 'cyan': [], 'metal': [], 'rubber': [], 'sphere': [], 'cube': [], 'cylinder': []}
    binary = {'left': [(0, 1), (0, 2), (1, 2)], 'front': []}
    ground = GroundTuples(unary, binary)

    unary_uncertain = {'large': [], 'small': [], 'red': [], 'blue': [], 'yellow': [], 'green': [], 'purple': [], 'gray': [], 'brown': [], 'cyan': [], 'metal': [], 'rubber': [], 'sphere': [1, 2], 'cube': [], 'cylinder': [1, 2]}
    binary_uncertain = {'left': [], 'front': []}
    ground_uncertain = GroundTuples(unary_uncertain, binary_uncertain)

    # Step 3: Interpreter
    interpreter = Interpreter(ground, ground_uncertain, n_vars, n_objects)

    # Step 4: Program
    #program = Program([UnaryClause('red', 0), BinaryClause('left', 0, 1), UnaryClause('blue', 1)])
    #program = Program([UnaryClause('red', 0), BinaryClause('left', 0, 1), UnaryClause('sphere', 1), UnaryClause('blue', 1)])
    program = Program([BinaryClause('left', 0, 1), BinaryClause('left', 1, 2)])
    target_object = 0

    # Step 5: Printing
    print('Ground tuples:')
    print(ground)
    print()

    print('Clauses:')
    for clause in interpreter.clauses:
        print(clause)
    print()

    print('Possible bindings:')
    for binding in interpreter.bindings:
        print(binding)
    print()

    print('Lookup table for semantics:')
    print(interpreter.semantics_lookup)
    print()

    print('Program: {}'.format(program))
    print()

    # Step 6: Interpret program
    state = interpreter.reset()
    for h, clause in enumerate(program.clauses):
        # Step 6b: Incremental execution
        state = interpreter.step(state, clause, True)

        # Step 6b: Printing
        print('Clause {}: {}'.format(h, clause))
        print()
        print('State: {}'.format(state))
        print()
        print('Bindings:')
        for binding in interpreter.get_bindings(state, False):
            print(binding)
        print()
        print('Uncertain bindings:')
        for binding in interpreter.get_bindings(state, True):
            print(binding)
        print()
        print('Marginal bindings:')
        print(interpreter.get_marginal_bindings(state, False))
        print()
        print('Uncertain marginal bindings:')
        print(interpreter.get_marginal_bindings(state, True))
        print()
        print('Is state correct: {}'.format(interpreter.is_state_correct(target_object, state, True)))
        print()


    # Step 7: Synthesis
    t = time.time()
    program_synth = synthesize(interpreter, target_object, interpreter.reset(), True, 3)
    print('Program: {}'.format(program_synth))
    t = time.time() - t
    print('Time: {}'.format(t))
    print()

    # Step 8: Check clause usefulness
    state = interpreter.reset()
    print('Clause usefulness:')
    for clause in interpreter.clauses:
        is_useful = interpreter.is_clause_useful(state, clause, True)
        print('({}) {}'.format(is_useful, clause))
    print()
