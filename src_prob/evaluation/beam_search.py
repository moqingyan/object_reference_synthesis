import os
import sys
import json 
import torch 
from torch_geometric.data import Data, DataLoader
import heapq
from dataclasses import dataclass, field
from typing import Any
import random 

common_path = os.path.abspath(os.path.join(__file__, "../../common"))
src_path = os.path.abspath(os.path.join(__file__, "../../"))

print(common_path)
print(src_path)
sys.path.append(common_path)
sys.path.append(src_path)

from cmd_args import cmd_args, logging
from env import Env
from embedding import GNN, SceneDataset, create_dataset
from utils import get_config
from copy import deepcopy

zero_suc = [2, 3, 6, 22, 25, 33, 34, 36, 38, 46, 48, 49, 60, 72, 114, 126, 132, 134, 137, 145, 149, 152, 153, 154, 158, 162, 171, 192, 204, 208, 216, 217, 221, 224, 225, 231, 233, 236, 238, 245, 246, 247, 249, 255, 266, 267, 307, 316, 345, 354, 356, 359, 361, 362, 367, 391, 396, 397, 409, 410, 412, 427, 428, 432, 444, 445, 447, 463, 472, 489, 491, 495, 497, 498, 515, 522, 523, 525, 544, 546, 560, 565, 569, 574, 581, 605, 606, 610, 614, 619, 641, 672, 673, 698, 702, 706, 708, 712, 716, 717, 720, 725, 757, 778, 781, 783, 785, 790, 813, 819, 827, 829, 839, 842, 844, 845, 849, 852, 861, 871, 892, 895, 897, 900, 903, 904, 910, 913, 919, 954, 971, 973, 980, 987, 1021, 1026, 1027, 1038, 1047, 1048, 1052, 1056, 1059, 1097, 1100, 1113, 1115, 1119, 1121, 1136, 1137, 1138, 1139, 1143, 1156, 1161, 1162, 1163, 1166, 1173, 1174, 1176, 1182, 1198, 1200, 1224, 1231, 1234, 1249, 1257, 1258, 1262, 1263, 1265, 1269, 1271, 1280, 1283, 1309, 1313, 1328, 1332, 1338, 1349, 1352, 1366, 1367, 1369, 1370, 1375, 1376, 1398, 1399, 1401, 1405, 1406, 1408, 1430, 1439, 1442, 1452, 1457, 1464, 1478, 1480, 1481, 1486, 1511, 1521, 1523, 1524, 1527, 1530, 1532, 1534, 1550, 1551, 1571, 1573, 1574, 1576, 1602, 1603, 1607, 1609, 1618, 1626, 1631, 1662, 1668, 1671, 1676, 1683, 1688, 1689, 1690, 1705, 1724, 1732, 1759, 1762, 1775, 1776, 1780, 1786, 1787, 1788, 1789, 1793, 1799, 1810, 1815, 1820, 1835, 1852, 1857, 1869, 1870, 1876, 1878, 1884, 1888, 1895, 1896, 1897, 1900, 1903, 1905, 1940, 1941, 1945, 1947, 1948, 1954, 1963, 1964, 1969, 1974, 1975, 1976, 1977, 1982, 1989, 1995, 1996, 1998, 2004, 2005, 2008, 2015, 2016, 2022, 2030, 2032, 2035, 2036, 2058, 2062, 2063, 2064, 2068, 2069, 2087, 2093, 2096, 2115, 2116, 2118, 2129, 2131, 2132, 2138, 2146, 2148, 2149, 2153, 2154, 2155, 2157, 2158, 2169, 2174, 2175, 2176, 2185, 2186, 2192, 2193, 2195, 2218, 2236, 2239, 2240, 2241, 2243, 2244, 2254, 2265, 2269, 2271, 2276, 2277, 2292, 2293, 2294, 2295, 2298, 2299, 2308, 2310, 2311, 2315, 2334, 2342, 2343, 2351, 2364, 2370, 2371, 2384, 2387, 2389, 2402, 2440, 2442, 2460, 2461, 2475, 2480, 2483, 2502, 2504, 2505, 2509, 2513, 2521, 2522, 2527, 2540, 2541, 2545, 2550, 2552, 2553, 2577, 2579, 2581, 2584, 2589, 2593, 2607, 2610, 2620, 2623, 2625, 2627, 2629, 2633, 2634, 2635, 2636, 2642, 2662, 2679, 2695, 2699, 2703, 2710, 2712, 2716, 2729, 2754, 2756, 2760, 2761, 2775, 2792, 2795, 2803, 2813, 2817, 2818, 2832, 2834, 2836, 2838, 2845, 2869, 2875, 2884, 2886, 2892, 2895, 2899, 2906, 2907, 2908, 2915, 2924, 2925, 2927, 2928, 2932, 2935, 2949, 2994, 3026, 3027, 3031, 3034, 3037, 3065, 3069, 3077, 3081, 3087, 3090, 3091, 3092, 3103, 3138, 3139, 3207, 3215, 3229, 3230, 3232, 3233, 3235, 3236, 3237, 3238, 3241, 3243, 3250]
one_suc = [7, 18, 19, 26, 35, 37, 47, 52, 55, 57, 58, 71, 74, 81, 88, 116, 118, 141, 167, 176, 177, 178, 202, 205, 206, 214, 218, 220, 229, 235, 244, 251, 254, 261, 299, 304, 306, 309, 312, 315, 317, 318, 319, 321, 333, 360, 392, 395, 398, 416, 434, 436, 454, 455, 458, 459, 462, 466, 473, 478, 496, 503, 506, 519, 536, 538, 541, 563, 566, 572, 575, 578, 580, 583, 609, 612, 630, 634, 636, 638, 645, 646, 694, 695, 696, 703, 711, 750, 755, 759, 761, 779, 780, 782, 784, 788, 791, 792, 793, 795, 808, 811, 812, 817, 821, 824, 830, 838, 840, 846, 847, 851, 857, 862, 875, 888, 912, 956, 958, 994, 995, 997, 999, 1044, 1045, 1049, 1050, 1076, 1090, 1091, 1092, 1094, 1098, 1099, 1102, 1104, 1107, 1108, 1114, 1118, 1120, 1125, 1129, 1132, 1135, 1157, 1167, 1171, 1194, 1196, 1197, 1199, 1202, 1205, 1219, 1227, 1239, 1255, 1259, 1261, 1281, 1285, 1289, 1306, 1307, 1310, 1312, 1327, 1333, 1340, 1368, 1372, 1373, 1381, 1385, 1393, 1409, 1432, 1446, 1455, 1458, 1463, 1476, 1479, 1483, 1496, 1498, 1518, 1519, 1522, 1525, 1529, 1531, 1547, 1565, 1575, 1580, 1581, 1611, 1617, 1620, 1625, 1628, 1669, 1672, 1674, 1677, 1691, 1696, 1703, 1715, 1725, 1727, 1728, 1730, 1740, 1745, 1755, 1763, 1764, 1766, 1772, 1773, 1774, 1783, 1784, 1791, 1798, 1805, 1809, 1811, 1817, 1821, 1836, 1858, 1860, 1873, 1874, 1875, 1902, 1904, 1907, 1922, 1925, 1928, 1929, 1931, 1933, 1938, 1949, 1950, 1951, 1961, 1962, 1965, 1972, 1978, 1984, 1991, 1999, 2003, 2012, 2017, 2027, 2029, 2041, 2052, 2053, 2072, 2073, 2079, 2094, 2121, 2124, 2134, 2143, 2150, 2165, 2170, 2189, 2196, 2197, 2203, 2206, 2207, 2216, 2219, 2220, 2222, 2238, 2251, 2272, 2273, 2300, 2309, 2325, 2335, 2336, 2338, 2340, 2341, 2344, 2362, 2372, 2376, 2386, 2391, 2404, 2417, 2420, 2437, 2439, 2446, 2463, 2469, 2503, 2508, 2510, 2515, 2517, 2518, 2520, 2529, 2530, 2531, 2533, 2535, 2539, 2544, 2551, 2555, 2572, 2578, 2591, 2613, 2624, 2637, 2648, 2649, 2653, 2661, 2689, 2706, 2746, 2753, 2768, 2770, 2772, 2773, 2777, 2785, 2794, 2802, 2805, 2808, 2816, 2821, 2841, 2843, 2846, 2847, 2865, 2866, 2870, 2872, 2874, 2877, 2879, 2887, 2891, 2896, 2898, 2923, 2929, 2943, 2964, 2993, 2996, 2999, 3006, 3023, 3024, 3025, 3054, 3055, 3076, 3094, 3104, 3141, 3177, 3181, 3206, 3208, 3210, 3213, 3216, 3231, 3234, 3247]
two_suc = [10, 13, 41, 69, 102, 121, 127, 131, 146, 173, 175, 191, 195, 209, 212, 223, 250, 259, 262, 292, 308, 310, 313, 314, 326, 335, 342, 363, 368, 370, 390, 394, 407, 411, 453, 468, 476, 480, 499, 504, 505, 511, 524, 539, 542, 552, 567, 577, 603, 611, 613, 618, 651, 666, 693, 719, 723, 724, 727, 728, 751, 760, 814, 816, 818, 822, 825, 826, 850, 856, 863, 866, 873, 907, 918, 935, 947, 949, 953, 970, 974, 993, 998, 1037, 1042, 1051, 1066, 1070, 1073, 1085, 1088, 1089, 1096, 1111, 1116, 1133, 1155, 1160, 1170, 1172, 1178, 1185, 1215, 1220, 1221, 1222, 1223, 1226, 1229, 1230, 1247, 1250, 1256, 1266, 1267, 1287, 1300, 1319, 1320, 1330, 1334, 1344, 1361, 1387, 1402, 1419, 1423, 1424, 1434, 1435, 1436, 1443, 1444, 1445, 1461, 1462, 1465, 1477, 1489, 1492, 1499, 1512, 1526, 1540, 1546, 1552, 1563, 1577, 1578, 1598, 1600, 1601, 1610, 1612, 1621, 1623, 1654, 1656, 1680, 1685, 1708, 1716, 1731, 1733, 1744, 1752, 1761, 1767, 1770, 1778, 1785, 1794, 1800, 1801, 1802, 1808, 1812, 1827, 1859, 1861, 1871, 1891, 1906, 1924, 1926, 1927, 1935, 1946, 1952, 1994, 1997, 2002, 2043, 2051, 2054, 2055, 2060, 2076, 2078, 2082, 2091, 2092, 2102, 2106, 2123, 2137, 2141, 2151, 2180, 2183, 2205, 2209, 2217, 2223, 2278, 2281, 2287, 2289, 2297, 2305, 2313, 2322, 2352, 2357, 2359, 2419, 2444, 2445, 2449, 2450, 2471, 2479, 2536, 2538, 2549, 2568, 2597, 2631, 2647, 2655, 2665, 2666, 2680, 2685, 2704, 2707, 2709, 2726, 2735, 2736, 2765, 2771, 2786, 2796, 2806, 2807, 2811, 2820, 2825, 2837, 2839, 2840, 2844, 2849, 2850, 2868, 2873, 2882, 2888, 2903, 2909, 2936, 2940, 2941, 2948, 2950, 2952, 2961, 2991, 3003, 3007, 3010, 3013, 3051, 3078, 3102, 3105, 3130, 3143, 3149, 3150, 3193, 3203, 3209, 3220]
three_suc = [16, 17, 28, 40, 44, 84, 85, 133, 148, 169, 170, 215, 219, 240, 265, 273, 327, 343, 344, 358, 383, 399, 435, 456, 484, 516, 535, 545, 586, 604, 643, 649, 653, 658, 669, 670, 671, 700, 752, 756, 787, 828, 843, 854, 868, 874, 899, 948, 989, 1030, 1036, 1057, 1077, 1112, 1126, 1127, 1140, 1158, 1240, 1279, 1294, 1308, 1362, 1364, 1365, 1388, 1418, 1427, 1447, 1495, 1497, 1508, 1509, 1513, 1558, 1562, 1568, 1569, 1572, 1587, 1588, 1597, 1619, 1670, 1699, 1701, 1710, 1719, 1741, 1742, 1760, 1779, 1782, 1804, 1837, 1867, 1936, 1953, 2031, 2044, 2075, 2085, 2135, 2145, 2163, 2246, 2270, 2274, 2275, 2282, 2328, 2330, 2339, 2345, 2403, 2435, 2452, 2458, 2484, 2528, 2554, 2573, 2594, 2596, 2605, 2632, 2650, 2659, 2663, 2673, 2675, 2690, 2732, 2734, 2752, 2757, 2797, 2827, 2833, 2854, 2867, 2881, 2894, 2904, 2919, 2937, 2944, 3009, 3014, 3043, 3053, 3057, 3115, 3133, 3200, 3217]

@dataclass(order=True)
class PrioritizedEnv:
    priority: int
    env: Any=field(compare=False)

# We want to generate refered varibles, and limit the constraint
# to either choosing the attributes of refered object, or choosing
# the relation involves the refered objects
def get_refs(clauses):
    refs = [0]
    for clause in clauses:
        if 1 in clause and not 1 in refs:
            refs.append(1)
        if 2 in clause and not 2 in refs:
            refs.append(2)
    return refs
    
# Beam search algorithm 
class Beam():

    def __init__(self, beam_size=cmd_args.beam_size):
        self.heap = []
        self.beam_size = beam_size

    # we want to rank by the maximum probability
    def insert():
        raise NotImplementedError
      
    def __iter__(self):
        return iter(self.heap)

    def __len__(self):
        return len(self.heap)

class ClauseBeam(Beam):

    def __init__(self, beam_size=cmd_args.beam_size):
        super().__init__(beam_size)

    def insert(self, prob, partial_porg, state, node_embeddings, cur_embedding):
        heapq.heappush(self.heap, (prob[0], partial_porg, state, node_embeddings, cur_embedding))
        if len(self.heap) > self.beam_size:
            heapq.heappop(self.heap)

class ProgBeam(Beam):

    def __init__(self,  beam_size=cmd_args.beam_size):
        super().__init__(beam_size)

    def insert(self, prob, env):
        item = PrioritizedEnv(priority=prob, env=env)
        heapq.heappush(self.heap, item)
        if len(self.heap) > self.beam_size:
            heapq.heappop(self.heap)

    def __iter__(self):
        for item in self.heap:
            yield (item.priority, item.env)

def transfer_to_clause(decoder, partial_prog):
        clause = []
        sel_rela_or_attr = partial_prog[0]
        sel_var1 = partial_prog [1]
        if len(partial_prog) == 2:
            operation = decoder.get_attr_operation(sel_rela_or_attr, graph.config)
            clause.append(operation)
            clause.append(sel_rela_or_attr)
            clause.append(sel_var1)
        else:
            sel_var2 = partial_prog [2]
            clause.append(sel_rela_or_attr)
            clause.append(sel_var1)
            clause.append(sel_var2)
        return clause

class SearchClause():

    def __init__(self, refrl, eps=cmd_args.eps):
        self.gnn = refrl.gnn
        self.policy = refrl.policy
        self.decoder = refrl.policy.decoder 
        self.eps = eps 

    def get_probs(self, locs, elements, layer, graph, node_embeddings):

        # setup 
        node_num = node_embeddings.shape[0]
        byte = self.decoder.locs_to_byte(locs, node_num)

        # pass forward
        embeddings = node_embeddings[byte]
        if (embeddings.shape[0] == 0):
            raise Exception("Wrong shape!")
        probs = layer(embeddings)

        if not self.eps == None:
            probs = probs * (1 - self.eps) + self.eps / probs.shape[0]

        for i in range(len(probs)):

            prob = torch.index_select(probs, 0, torch.tensor(i))
            sel_element = elements[i]
            embedding = torch.index_select(embeddings, 0, torch.tensor(i))
            yield (prob, sel_element, embedding)

    def get_attr_or_rela_probs(self, graph, node_embeddings):
        
        # setup 
        attr_or_rela_locs = graph.attr_or_rela.values()
        attr_or_rela = list(graph.attr_or_rela.keys())
        layer = self.decoder.attr_or_rela_score_layer

        # pass forward
        return self.get_probs(attr_or_rela_locs, attr_or_rela, layer, graph, node_embeddings)
        
    def get_var1_probs(self, graph, node_embeddings):

        # setup
        var_locs = graph.vars.values()
        var = list(graph.vars.keys())
        layer = self.decoder.var1_score_layer

        # pass forward
        return self.get_probs(var_locs, var, layer, graph, node_embeddings)

    def get_var2_probs(self, graph, node_embeddings):

        # setup
        var_locs = graph.vars.values()
        var = list(graph.vars.keys())
        layer = self.decoder.var2_score_layer

        # pass forward
        return self.get_probs(var_locs, var, layer, graph, node_embeddings)
 
    def get_clause_with_prob(self, graph, graph_embedding):
        
        current_beam = ClauseBeam()
        next_beam = ClauseBeam()

        node_embeddings = graph_embedding[0]
        state = self.decoder.gru_cell(graph_embedding[1])
        attr_or_rela_iter = self.get_attr_or_rela_probs(graph, node_embeddings)
    
        for attr_or_rela_prob, sel_attr_or_rela, rela_or_attr_embedding in attr_or_rela_iter:
            current_beam.insert(attr_or_rela_prob, [sel_attr_or_rela], state, node_embeddings, rela_or_attr_embedding)

        for partial_prob, partial_prog, state, node_embeddings, cur_embedding in current_beam:
            state_var1 = self.decoder.gru_cell(cur_embedding, state)
            node_embeddings_var1 = torch.stack([self.decoder.gru_cell(node_embedding.unsqueeze(0), state_var1) for node_embedding in node_embeddings]).reshape(node_embeddings.shape)
            var1_iter = self.get_var1_probs(graph, node_embeddings_var1)

            for var1_prob, sel_var1, var1_embedding in var1_iter:
                next_beam.insert(partial_prob * var1_prob, partial_prog + [sel_var1], state_var1, node_embeddings_var1, var1_embedding)
        
        current_beam = next_beam 
        next_beam = ClauseBeam()

        for partial_prob, partial_prog, state_var1, node_embeddings_var1, cur_embedding in current_beam:
            if self.decoder.is_attr(graph, partial_prog[0]):
                next_beam.insert([partial_prob], partial_prog, None, None, None)
            else:
                state_var2 = self.decoder.gru_cell(cur_embedding, state_var1)
                node_embeddings_var2 = torch.stack([self.decoder.gru_cell(node_embedding.unsqueeze(0), state_var2) for node_embedding in node_embeddings_var1]).reshape(node_embeddings_var1.shape)
                var2_iter = self.get_var2_probs(graph, node_embeddings_var2)
                for var2_prob, sel_var2, var2_embedding in var2_iter:
                    next_beam.insert(partial_prob * var2_prob, partial_prog + [sel_var2] ,None, None, None )
        
        return next_beam

class SearchProg():
    
    def __init__(self, refrl, max_length = cmd_args.episode_length):
        self.gnn = refrl.gnn
        self.config = refrl.config
        self.decoder = refrl.policy.decoder
        self.encoder = refrl.encoder
        self.search_clause = SearchClause(refrl)
        self.max_length = max_length

    def get_one_clause(self, env_prob, env):
        refs = get_refs(env.clauses)
        graph_embedding = self.gnn(env.data)
        clause_beam = self.search_clause.get_clause_with_prob(env.graph, graph_embedding)

        for prob, c, _, _, _ in clause_beam:

            new_clause = transfer_to_clause(self.decoder, c)
            if cmd_args.hard_constraint:
                if not (new_clause[1] in refs or new_clause[2] in refs):
                    continue

                if new_clause in env.clauses:
                    continue

            new_env = deepcopy(env)
            new_env.clauses.append(new_clause)
            # new_env.step()
            # if new_env.possible:
            
            yield env_prob * prob, new_env

    def get_one_prog(self, graph, data_point):
        env = Env(data_point, graph, self.config, self.encoder)
        current_beam = ProgBeam()
        next_beam = ProgBeam()
        current_beam.insert(1.0, env)
        finished_count = 0
        progs_done = False

        while len(current_beam) > 0:
            
            if progs_done:
                break 

            for env_prob, env in current_beam:
                
                # Stop at a upper bound length
                if len(env.clauses) > self.max_length:
                    progs_done = True

                env.step()
                if env.success:
                    next_beam.insert(env_prob, env)
                    finished_count += 1
                elif env.possible: 
                    prob_env_iter = self.get_one_clause(env_prob, env)
                    prob_env_iter = list(prob_env_iter)
                    random.shuffle(prob_env_iter)
                    for new_env_prob, new_env in prob_env_iter:
                        next_beam.insert(new_env_prob, new_env)

                # all of progs in the beam is already expanded
                if finished_count == len(current_beam):
                    progs_done = True
                    
            current_beam = next_beam
            next_beam = ProgBeam()
            finished_count = 0

        return current_beam

if __name__ == "__main__":
    # arrange all the directories
    data_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../../data"))
    model_dir = os.path.abspath(os.path.join(data_dir, "oak_model/model"))
    log_dir = os.path.abspath(os.path.join(data_dir, "log"))

    scene_file_name = "img_test_500.json"
    graph_file_name = "img_test_500.pkl"
    datafile_name = "img_test_500.pt"
    search_file_name = "beam_search_500.json"

    raw_path = os.path.abspath(os.path.join(data_dir, "./processed_dataset/raw"))
    scenes_path = os.path.abspath(os.path.join(raw_path, scene_file_name))
    graphs_path = os.path.join(raw_path, graph_file_name)
    success_path = os.path.join(data_dir, f"./eval_result/{search_file_name}")

    graphs, scene_dataset = create_dataset(data_dir, scenes_path, graphs_path)

    # update the cmd_args corresponding to the info we have 
    cmd_args.graph_file_name = graph_file_name

    lr4_10_model_path = os.path.join(model_dir, "refrl-0.0001-penyes-no_intermediate-norno-img_test_30.pkl")
    refrl = torch.load(lr4_10_model_path)
    refrl.policy.eval()
    search_clause = SearchClause(refrl)
    search_prog = SearchProg(refrl)

    data_loader = DataLoader(scene_dataset)
    for ct, data_point in enumerate(data_loader):
        with torch.no_grad():
            if ct < 2390:
                continue
            logging.info(ct)
            graph = graphs[data_point.graph_id]
            beam = search_prog.get_one_prog(graph, data_point)

            # graph_embedding = refrl.gnn(data_point)
            # beam = search_space.get_clause_with_prob(graph, graph_embedding)
            # for prob, prog, _,_,_ in beam:
            #     print (prog, prob)
            # break

            for prob, env in beam:
                logging.info( f" prob: {prob}, clause: {env.clauses}, suc: {env.success}")
