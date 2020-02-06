
# Reference: https://www.python-course.eu/graphs_python.php

class BaseDiGraph(object):

    def __init__(self, graph_dict=None):
        """ initializes a graph object 
            If no dictionary or None is given, 
            an empty dictionary will be used
        """
        if graph_dict == None:
            graph_dict = {}
        self.__graph_dict = graph_dict

    def get_names(self):
        return list(self.__graph_dict.keys())

    def vertices(self):
        """ returns the vertices of a graph """
        return self.__graph_dict.keys()

    def edges(self):
        """ returns the edges of a graph """
        return self.__generate_edges()
 
    def add_vertex(self, vertex):
        """ If the vertex "vertex" is not in 
            self.__graph_dict, a key "vertex" with an empty
            list as a value is added to the dictionary. 
            Otherwise nothing has to be done. 
        """
        if vertex not in self.__graph_dict.keys():
            self.__graph_dict[vertex] = []


    def remove_edge(self, src, target):
        assert src in self.__graph_dict.keys()
        assert target in self.__graph_dict.keys()

        self.__graph_dict[src] = list(filter(lambda x: not x[0] == target, self.__graph_dict[src]))
        
    def remove_vertex(self, vertex):
        assert vertex in self.__graph_dict.keys()

        for v, n in self.__graph_dict.items():
            self.__graph_dict[v] = list(filter(lambda x: not x[0] == vertex, n))

        self.__graph_dict.pop(vertex)
        
    # Must create node first then edges
    def add_edge(self, vertex1, vertex2, edge_type):
        """ assumes that edge is of type set, tuple or list; 
            between two vertices can be multiple edges! 
        """
        # print(self.__name_list)
        
        assert vertex1 in self.__graph_dict.keys()
        assert vertex2 in self.__graph_dict.keys()

        self.__graph_dict[vertex1].append((vertex2, edge_type))

    def add_undirected_edge(self, vertex1, vertex2, edge_type):
        self.add_edge(vertex1, vertex2, edge_type)
        self.add_edge(vertex2, vertex1, edge_type)

    def __generate_edges(self):
        """ A static method generating the edges of the 
            graph "graph". Edges are represented as sets 
            with one (a loop back to the vertex) or two 
            vertices 
        """
        edges = [], []
        edge_type = []

        lookup_list = list(self.__graph_dict.keys())
        lookup_table = dict(zip(lookup_list, range(0, len(lookup_list)))) 

        for vertex in self.__graph_dict:
            for neighbour in self.__graph_dict[vertex]:
                    edges[0].append(lookup_table[vertex])
                    edges[1].append(lookup_table[neighbour[0]])
                    if not type(neighbour[1]) == int:
                        edge_type.append(neighbour[1].value)
                    else:
                        edge_type.append(neighbour[1])
        return edges, edge_type
    
    def is_neighbour(self, v1, v2):
        assert v1 in self.__graph_dict.keys()
        assert v2 in self.__graph_dict.keys()

        v1_idx = self.__graph_dict.keys().index(v1)
        v2_idx = self.__graph_dict.keys().index(v2)

        return v2_idx in self.__graph_dict[v1_idx]

    def get_neighbour(self, v1):
        assert v1 in self.__graph_dict.keys()
        if not v1 in self.__graph_dict.keys():
            return []
        neighbour = self.__graph_dict[v1]
        return [n for n, e in neighbour]

    def __str__(self):
        res = "vertices: "
        for k in self.__graph_dict.keys():
            res += str(k) + " "
        res += "\nedges: "
        for edge in self.__generate_edges():
            res += str(edge) + " "
        return res

if __name__ == "__main__":
    name_list = ["Shanghai", "London", "San_Fransico", "Kyoto"]

    graph_dict = {}
    graph_dict[name_list[0]] = [(name_list[1],1), (name_list[2],1),(name_list[3],1)]
    graph_dict[name_list[1]] = [(name_list[2],1), (name_list[3],1)]
    graph_dict[name_list[2]] = [(name_list[3],1)]
    graph_dict[name_list[3]] = [(name_list[0],1)]

    g = BaseDiGraph(graph_dict)
    print(g)

    g.add_edge("Kyoto", "London", 2)
    print(g)

    g.add_vertex("Berlin")
    g.add_edge("Berlin", "Shanghai", 3)
    print(g)

    g.remove_vertex("London")
    print(g)

    g.remove_edge("Shanghai", "San_Fransico")
    print(g)

    print(g.get_neighbour("Shanghai"))
