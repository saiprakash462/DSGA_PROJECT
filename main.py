"""
E0 251o Project (2026)
Property-Based Testing for NetworkX using Hypothesis

Student-1 Name: Srinivas Shavukapu Kattegummula (SR No: 13-19-02-19-52-25-1-26388 )
Student-2 Name: Polavarapu Hema Naga Sai Prakash (SR No: 13-19-01-19-52-25-1-26498 )
Course: E0 251o

Algorithms and Graph Properties Tested:
1. Shortest Path Symmetry
2. Shortest Path Non-Negative Distance
3. Minimum Spanning Tree (MST) Edge Count
4. MST Node Preservation
5. MST Idempotence
6. Single Node Graph Boundary Condition
7. Triangle Inequality for Shortest Path
8. MST Connectivity
9. MST Acyclicity
10. Self Distance Zero
11. Connected Components Validation
12. Isolated Node Component Increase
13. Degree Centrality Range
14. Complete Graph Equal Centrality
15. BFS Tree Edge Count
16. DFS Tree Edge Count
17. Graph Diameter Non-Negativity
18. Clustering Coefficient Range
19. Tree Edge-Node Invariant
20. Shortest Path Upper Bound

Description:
This file contains property-based tests for validating important mathematical
properties of graph algorithms in NetworkX.
"""
import inspect
import sys

import networkx as nx
from hypothesis import given, strategies as st, settings

from functools import wraps


# --------------------------------------------------
# PROPERTY TEST RESULT LOGGER DECORATOR
# --------------------------------------------------

def property_result_logger(func):
    """
    Logs PASS / FAIL exactly once per property test.

    This decorator wraps the complete Hypothesis property test
    and prints the final result only once after all generated
    examples have been executed.

    PASS  -> all generated examples satisfy the property
    FAIL  -> at least one generated example violates the property
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            func(*args, **kwargs)
            print(f"\nPASSED: {func.__name__}")
        except Exception:
            print(f"\nFAILED: {func.__name__}")
            raise
    return wrapper

# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def create_connected_graph(n: int) -> nx.Graph:
    """
    Creates a connected undirected graph with n nodes.

    Strategy:
    First creates a path graph to guarantee connectivity,
    then adds a few extra edges.
    """
    G = nx.path_graph(n)

    # Add additional edges for richer structure
    for i in range(n - 2):
        G.add_edge(i, i + 2)

    return G


# --------------------------------------------------
# PROPERTY TEST 1: SHORTEST PATH SYMMETRY
# --------------------------------------------------

@property_result_logger
@given(st.integers(min_value=2, max_value=20))
@settings(max_examples=30)

def test_shortest_path_symmetry(n):
    """
    Property:
    In an undirected graph, shortest path distance is symmetric.

    Mathematical reasoning:
    For any two vertices u and v in an undirected graph,
    distance(u, v) = distance(v, u)

    Why this matters:
    If this fails, shortest path algorithm may be computing
    inconsistent distances.
    """
    G = create_connected_graph(n)

    dist1 = nx.shortest_path_length(G, 0, n - 1)
    dist2 = nx.shortest_path_length(G, n - 1, 0)

    assert dist1 == dist2


# --------------------------------------------------
# PROPERTY TEST 2: NON-NEGATIVE DISTANCE
# --------------------------------------------------

@property_result_logger
@given(st.integers(min_value=2, max_value=20))
@settings(max_examples=30)
def test_shortest_path_non_negative(n):
    """
    Property:
    Shortest path length can never be negative.

    Mathematical reasoning:
    Path length counts number of edges traversed.
    Hence it must always be >= 0.

    Defect indication:
    Negative distance indicates severe logic issue.
    """
    G = create_connected_graph(n)

    dist = nx.shortest_path_length(G, 0, n - 1)

    assert dist >= 0


# --------------------------------------------------
# PROPERTY TEST 3: MST EDGE COUNT
# --------------------------------------------------

@property_result_logger
@given(st.integers(min_value=2, max_value=20))
@settings(max_examples=30)
def test_mst_edge_count(n):
    """
    Property:
    A spanning tree with n nodes must contain exactly n-1 edges.

    Mathematical basis:
    Any tree with n vertices contains exactly n-1 edges.

    Why this matters:
    More edges -> cycle exists
    Fewer edges -> graph disconnected
    """
    G = create_connected_graph(n)
    mst = nx.minimum_spanning_tree(G)

    assert mst.number_of_edges() == n - 1


# --------------------------------------------------
# PROPERTY TEST 4: MST NODE PRESERVATION
# --------------------------------------------------

@property_result_logger
@given(st.integers(min_value=2, max_value=20))
@settings(max_examples=30)
def test_mst_contains_all_nodes(n):
    """
    Property:
    Minimum spanning tree must preserve all graph nodes.

    Mathematical reasoning:
    A spanning tree must span every vertex.

    Defect indication:
    Missing nodes means algorithm failed to span graph.
    """
    G = create_connected_graph(n)
    mst = nx.minimum_spanning_tree(G)

    assert mst.number_of_nodes() == G.number_of_nodes()


# --------------------------------------------------
# PROPERTY TEST 5: IDEMPOTENCE
# --------------------------------------------------

@property_result_logger
@given(st.integers(min_value=2, max_value=20))
@settings(max_examples=30)
def test_mst_idempotence(n):
    """
    Property:
    Applying MST twice should yield same result.

    Mathematical reasoning:
    Once a graph is already a tree,
    reapplying MST should not change it.

    This is an idempotence property.
    """
    G = create_connected_graph(n)

    mst1 = nx.minimum_spanning_tree(G)
    mst2 = nx.minimum_spanning_tree(mst1)

    assert set(mst1.edges()) == set(mst2.edges())


# --------------------------------------------------
# PROPERTY TEST 6: BOUNDARY CONDITION
# --------------------------------------------------

@property_result_logger
def test_single_node_graph():
    """
    Boundary condition:
    Graph with one node must have zero edges in MST.

    Why important:
    Edge cases often reveal hidden defects.
    """
    G = nx.Graph()
    G.add_node(1)

    mst = nx.minimum_spanning_tree(G)

    assert mst.number_of_nodes() == 1
    assert mst.number_of_edges() == 0


# --------------------------------------------------
# PROPERTY TEST 7: TRIANGLE INEQUALITY
# --------------------------------------------------

@property_result_logger
@given(st.integers(min_value=3, max_value=20))
@settings(max_examples=30)
def test_shortest_path_triangle_inequality(n):
    """
    Property:
    Shortest path distances must satisfy triangle inequality.

    Mathematical reasoning:
    For any vertices a, b, c:
    d(a,c) <= d(a,b) + d(b,c)

    Why this matters:
    Violation indicates incorrect distance computation.
    """
    G = create_connected_graph(n)

    a, b, c = 0, n // 2, n - 1
    dab = nx.shortest_path_length(G, a, b)
    dbc = nx.shortest_path_length(G, b, c)
    dac = nx.shortest_path_length(G, a, c)

    assert dac <= dab + dbc


# --------------------------------------------------
# PROPERTY TEST 8: MST IS CONNECTED
# --------------------------------------------------

@property_result_logger
@given(st.integers(min_value=2, max_value=20))
@settings(max_examples=30)
def test_mst_is_connected(n):
    """
    Property:
    The minimum spanning tree (MST) generated from any connected graph must itself remain fully connected.

    Mathematical reasoning:
    A spanning tree is defined as a subgraph that includes all vertices of the original graph while preserving connectivity. This means there must exist at least one path between every pair of nodes in the MST.

    Test strategy:
    Generate connected graphs of different sizes and verify that the resulting MST is still connected using NetworkX's connectivity check.

    Why this matters:
    If this property fails, the algorithm has produced a disconnected structure, which means it is not a valid spanning tree.
    """
    G = create_connected_graph(n)
    mst = nx.minimum_spanning_tree(G)

    assert nx.is_connected(mst)


# --------------------------------------------------
# PROPERTY TEST 9: MST IS ACYCLIC
# --------------------------------------------------

@property_result_logger
@given(st.integers(min_value=2, max_value=20))
@settings(max_examples=30)
def test_mst_is_acyclic(n):
    """
    Property:
    A valid spanning tree must never contain cycles.

    Mathematical reasoning:
    By graph theory definition, every tree is an acyclic connected graph. Therefore, the MST must contain zero cycles regardless of graph size.

    Test strategy:
    Generate multiple connected graphs and confirm that the MST has no cycles by checking the cycle basis.

    Why this matters:
    If even one cycle exists, the result is no longer a tree and the MST algorithm output is incorrect.

    Mathematical reasoning:
    Trees are connected acyclic graphs.
    """
    G = create_connected_graph(n)
    mst = nx.minimum_spanning_tree(G)

    cycles = list(nx.cycle_basis(mst))
    assert len(cycles) == 0


# --------------------------------------------------
# PROPERTY TEST 10: SELF DISTANCE ZERO
# --------------------------------------------------

@property_result_logger
@given(st.integers(min_value=1, max_value=20))
@settings(max_examples=30)
def test_node_distance_to_itself_zero(n):
    """
    Property:
    The shortest path distance from any node to itself must always be zero.

    Mathematical reasoning:
    No edges need to be traversed to reach the same node, so the path length is always 0. This is a fundamental shortest-path invariant.

    Test strategy:
    For randomly generated connected graphs, verify that the distance from node 0 to itself is always zero.

    Why this matters:
    A non-zero value indicates a severe defect in shortest-path computation.
    """
    G = create_connected_graph(max(2, n))

    assert nx.shortest_path_length(G, 0, 0) == 0




# --------------------------------------------------
# PROPERTY TEST 11: CONNECTED COMPONENTS IN CONNECTED GRAPH
# --------------------------------------------------

@property_result_logger
@given(st.integers(min_value=2, max_value=20))
@settings(max_examples=30)
def test_connected_graph_has_one_component(n):
    """
    Property:
    A connected graph must always have exactly one connected component.
    """
    G = create_connected_graph(n)
    assert nx.number_connected_components(G) == 1


# --------------------------------------------------
# PROPERTY TEST 12: ISOLATED NODE INCREASES COMPONENT COUNT
# --------------------------------------------------

@property_result_logger
@given(st.integers(min_value=2, max_value=20))
@settings(max_examples=30)
def test_adding_isolated_node_increases_components(n):
    """
    Property:
    Adding an isolated node increases connected components by exactly one.
    """
    G = create_connected_graph(n)
    before = nx.number_connected_components(G)
    G.add_node(n + 100)
    after = nx.number_connected_components(G)
    assert after == before + 1


# --------------------------------------------------
# PROPERTY TEST 13: DEGREE CENTRALITY RANGE
# --------------------------------------------------

@property_result_logger
@given(st.integers(min_value=2, max_value=20))
@settings(max_examples=30)
def test_degree_centrality_range(n):
    """
    Property:
    Degree centrality must always be between 0 and 1.
    """
    G = create_connected_graph(n)
    centrality = nx.degree_centrality(G)
    for value in centrality.values():
        assert 0 <= value <= 1


# --------------------------------------------------
# PROPERTY TEST 14: COMPLETE GRAPH EQUAL CENTRALITY
# --------------------------------------------------

@property_result_logger
@given(st.integers(min_value=2, max_value=15))
@settings(max_examples=30)
def test_complete_graph_equal_degree_centrality(n):
    """
    Property:
    In a complete graph, all nodes must have the same degree centrality.
    """
    G = nx.complete_graph(n)
    centrality = nx.degree_centrality(G)
    assert len(set(centrality.values())) == 1

# --------------------------------------------------
# PROPERTY TEST 15: BFS TREE EDGE COUNT
# --------------------------------------------------

@property_result_logger
@given(st.integers(min_value=2, max_value=20))
@settings(max_examples=30)
def test_bfs_tree_edge_count(n):
    """
    Property:
    BFS tree of a connected graph must contain exactly n-1 edges.

    Mathematical reasoning:
    BFS tree is a spanning tree.
    Every spanning tree with n nodes contains n-1 edges.
    """
    G = create_connected_graph(n)
    bfs_tree = nx.bfs_tree(G, source=0)

    assert bfs_tree.number_of_edges() == n - 1


# --------------------------------------------------
# PROPERTY TEST 16: DFS TREE EDGE COUNT
# --------------------------------------------------

@property_result_logger
@given(st.integers(min_value=2, max_value=20))
@settings(max_examples=30)
def test_dfs_tree_edge_count(n):
    """
    Property:
    DFS tree of a connected graph must contain exactly n-1 edges.

    Mathematical reasoning:
    DFS traversal over connected graph produces spanning tree.
    """
    G = create_connected_graph(n)
    dfs_tree = nx.dfs_tree(G, source=0)

    assert dfs_tree.number_of_edges() == n - 1


# --------------------------------------------------
# PROPERTY TEST 17: GRAPH DIAMETER NON-NEGATIVE
# --------------------------------------------------

@property_result_logger
@given(st.integers(min_value=2, max_value=20))
@settings(max_examples=30)
def test_graph_diameter_non_negative(n):
    """
    Property:
    Diameter of any connected graph must be >= 0.

    Mathematical reasoning:
    Diameter is maximum shortest path distance.
    Distances cannot be negative.
    """
    G = create_connected_graph(n)
    diameter = nx.diameter(G)

    assert diameter >= 0


# --------------------------------------------------
# PROPERTY TEST 18: CLUSTERING COEFFICIENT RANGE
# --------------------------------------------------

@property_result_logger
@given(st.integers(min_value=3, max_value=20))
@settings(max_examples=30)
def test_clustering_coefficient_range(n):
    """
    Property:
    Clustering coefficient must always lie in [0,1].

    Mathematical reasoning:
    It is a normalized probability-like measure.
    """
    G = create_connected_graph(n)
    clustering = nx.clustering(G)

    for value in clustering.values():
        assert 0 <= value <= 1


# --------------------------------------------------
# PROPERTY TEST 19: EULER CHARACTERISTIC FOR TREE
# --------------------------------------------------

@property_result_logger
@given(st.integers(min_value=2, max_value=20))
@settings(max_examples=30)
def test_tree_edges_equals_nodes_minus_one(n):
    """
    Property:
    For any tree:
    edges = nodes - 1

    Mathematical reasoning:
    Fundamental theorem of trees.
    """
    G = nx.path_graph(n)

    assert G.number_of_edges() == G.number_of_nodes() - 1


# --------------------------------------------------
# PROPERTY TEST 20: SHORTEST PATH UPPER BOUND
# --------------------------------------------------

@property_result_logger
@given(st.integers(min_value=2, max_value=20))
@settings(max_examples=30)
def test_shortest_path_upper_bound(n):
    """
    Property:
    In connected graph, shortest path between two nodes
    cannot exceed n-1.

    Mathematical reasoning:
    Longest simple path in n nodes has at most n-1 edges.
    """
    G = create_connected_graph(n)
    dist = nx.shortest_path_length(G, 0, n - 1)

    assert dist <= n - 1

if __name__ == "__main__":
    pass
    print("Property Based NetworkX Testing is completed")
    """
    Dynamically find and run all test functions
    """
    current_module = sys.modules[__name__]
    for name, obj in inspect.getmembers(current_module):
        if inspect.isfunction(obj) and name.startswith('test_'):
            print(f"\nRunning test: {name}")
            try:
                obj()
            except Exception as e:
                print(f"Error running {name}: {e}")
    print("\nProperty Based NetworkX Testing is completed")
