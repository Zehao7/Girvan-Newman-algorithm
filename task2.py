from pyspark import SparkContext
import sys, time
from itertools import combinations
from collections import defaultdict, deque


# spark-submit task2.py 7 "/Users/leoli/Desktop/ub_sample_data.csv" "/Users/leoli/Desktop/task2_betweenness_output.txt" "/Users/leoli/Desktop/task2_community_output.txt"
# spark-submit –-class task2 hw4.jar 7 "/Users/leoli/Desktop/ub_sample_data.csv" "/Users/leoli/Desktop/task2_betweenness_output.txt" "/Users/leoli/Desktop/task2_community_output.txt"


class TreeNode:
    def __init__(self, id):
        self.id = id
        self.depth = 0
        self.parents = {}
        self.children = {}
        self.credit = 1
        self.edge_credit = 0



def girvan_newman_algorithm(graph):
    all_edges = get_all_edges(graph)
    degrees = compute_degrees(graph)
    best_modularity, best_communities = -1, []

    while all_edges:
        betweenness = compute_betweenness(graph)
        edges_to_remove = find_edges_with_highest_betweenness(betweenness)
        remove_edges(graph, edges_to_remove)
        all_edges -= edges_to_remove

        modularity, communities = compute_modularity(graph, all_edges, degrees)
        if modularity > best_modularity:
            best_modularity, best_communities = modularity, communities

    return best_communities



def get_all_edges(graph):
    edges = set()
    for node, neighbors in graph.items():
        for neighbor in neighbors:
            edges.add(frozenset({node, neighbor}))
    return edges



def compute_degrees(graph):
    return {node: len(neighbors) for node, neighbors in graph.items()}



def compute_modularity(graph, all_edges, degrees):
    double_m = sum(degrees.values())
    modularity = 0
    communities = find_communities(graph)

    for community in communities:
        for node_i in community:
            degree_i = degrees[node_i]
            for node_j in community:
                degree_j = degrees[node_j]
                modularity -= degree_i * degree_j / double_m

                if frozenset({node_i, node_j}) in all_edges:
                    modularity += 1

    return modularity / double_m, communities



def find_communities(graph):
    communities = []
    visited = set()

    for node in graph:
        if node not in visited:
            community = explore_community(graph, node)
            communities.append(community)
            visited |= community

    return communities



def explore_community(graph, start_node):
    seen = {start_node}
    queue = deque([start_node])

    while queue:
        node = queue.popleft()
        for neighbor in graph[node]:
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append(neighbor)

    return seen



def find_edges_with_highest_betweenness(betweenness):
    max_value = max(betweenness.values())
    return {edge for edge, value in betweenness.items() if value == max_value}



def remove_edges(graph, edges_to_remove):
    for edge in edges_to_remove:
        node1, node2 = edge
        graph[node1].remove(node2)
        graph[node2].remove(node1)



def compute_betweenness(graph):
    betweenness = defaultdict(float)

    for root in graph:
        tree = construct_bfs_tree(graph, root)
        max_depth = max(tree.keys())

        for depth in range(max_depth, 0, -1):
            for node in tree[depth]:
                total_credit = 1 + node.edge_credit
                for parent_id, parent_node in node.parents.items():
                    edge_betweenness = total_credit * parent_node.credit / node.credit
                    parent_node.edge_credit += edge_betweenness
                    edge = frozenset({node.id, parent_id})
                    betweenness[edge] += edge_betweenness

    for edge in betweenness:
        betweenness[edge] /= 2

    return betweenness



def construct_bfs_tree(graph, root_id):
    root = TreeNode(root_id)
    tree = defaultdict(set)
    tree[0].add(root)
    created_nodes = {root_id: root}
    queue = deque([root_id])

    while queue:
        current_node_id = queue.popleft()
        for neighbor_id in graph[current_node_id]:
            if neighbor_id not in created_nodes:
                new_node = TreeNode(neighbor_id)
                new_node.depth = created_nodes[current_node_id].depth + 1
                new_node.parents[current_node_id] = created_nodes[current_node_id]
                new_node.credit = created_nodes[current_node_id].credit
                tree[new_node.depth].add(new_node)
                created_nodes[current_node_id].children[neighbor_id] = new_node
                created_nodes[neighbor_id] = new_node
                queue.append(neighbor_id)
            else:
                neighbor_node = created_nodes[neighbor_id]
                if neighbor_node.depth > created_nodes[current_node_id].depth:
                    neighbor_node.credit += created_nodes[current_node_id].credit
                    created_nodes[current_node_id].children[neighbor_id] = neighbor_node
                    neighbor_node.parents[current_node_id] = created_nodes[current_node_id]
    return tree



if __name__ == "__main__":

    start_time = time.time()

    threshold = sys.argv[1]
    input_file_path = sys.argv[2]
    betweenness_output_file_path = sys.argv[3]
    community_output_file_path = sys.argv[4]


    sc = SparkContext.getOrCreate()
    sc.setLogLevel("ERROR")



    # print("----------------- read data ------------------")
    data_rdd = sc.textFile(input_file_path).filter(lambda x: "user_id" not in x)
    data_rdd = data_rdd.map(lambda x: x.split(",")).map(lambda x: (x[0], x[1]))

    user_group_businesses_rdd = data_rdd.groupByKey().mapValues(set)
    user_group_businesses_dict = user_group_businesses_rdd.collectAsMap()

    user_index_rdd = data_rdd.map(lambda x: x[0]).distinct().zipWithIndex()
    business_index_rdd = data_rdd.map(lambda x: x[1]).distinct().zipWithIndex()

    user_index_dict = user_index_rdd.collectAsMap()
    # business_index_dict = business_index_rdd.collectAsMap()



    # print("----------------- generate pairs ------------------")
    pairs_rdd = sc.parallelize(list(combinations(user_index_dict.keys(), 2)))

    def find_edges(pair):
        user1_businesses = user_group_businesses_dict[pair[0]]
        user2_businesses = user_group_businesses_dict[pair[1]]
        common_businesses = user1_businesses.intersection(user2_businesses)
        return len(common_businesses) >= int(threshold)
    
    edges_rdd = pairs_rdd.filter(find_edges).flatMap(lambda x: [x, (x[1], x[0])])
    # print(edges_rdd.take(5))

    graph_rdd = edges_rdd.groupByKey().mapValues(set)
    graph = graph_rdd.collectAsMap()
    # print(graph_rdd.take(5))


    print("----------------- Task 2.1 -----------------")
    edge_betweenness_frozenset = compute_betweenness(graph)
    
    edge_betweenness = {}
    for edge, betweenness in edge_betweenness_frozenset.items():
        edge_betweenness[tuple(sorted(edge))] = round(betweenness, 5)
        
    print("time: {0:.5f}".format(time.time() - start_time))


    print("----------------- wrtie betweenness_output_file -----------------")
    with open(betweenness_output_file_path, 'w', newline='') as output_file:
        for community in sorted(list(edge_betweenness.items()), key = lambda x: (-x[1], x[0])):
            output_file.write(str(community)[1:-1] + "\n")

    print("time: {0:.5f}".format(time.time() - start_time))


    print("----------------- Task 2.2 -----------------")
    communities = girvan_newman_algorithm(graph)
    communities = [sorted(community) for community in communities]
    communities = sorted(communities, key=lambda x: (len(x), x))
    # print(communities)
    print("time: {0:.5f}".format(time.time() - start_time))


    print("----------------- wrtie community_output_file -----------------")
    with open(community_output_file_path, 'w', newline='') as output_file:
        for community in communities:
            output_file.write(str(community)[1:-1] + "\n")

    print("time: {0:.5f}".format(time.time() - start_time))
