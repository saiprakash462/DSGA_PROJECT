"""
E0 251o Project (2026)
Property-Based Testing for NetworkX using Hypothesis

Student-1 Name: Srinivas Shavukapu Kattegummula
Student-2 Name: Polavarapu Hema Naga Sai Prakash
Course: E0 251o


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
