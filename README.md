# Graph Partitioning using Multistart and Genetic Algorithms

This project implements a metaheuristic algorithm for graph partitioning optimization problem using a local serach, multistart and genetic algorithms. 

## Optimization Problem 
Solving the optimization problem: Given a weighted graph G=(V,E) where V is a set of n nodes and E is a set of edges, let wi ≥ 0 be the weight of node i ∈ V and let cij ({i, j} ∈ E) be the edge weight between nodes i and j (cij =0, if {i, j} ∉ E). A clustering/grouping problem is to partition V into a given number p (p ≤ n) of disjoint clusters or groups such that the sum of node weights in each cluster is constrained by an upper and a lower capacity limit, while maximizing the sum of edge weights whose two associated endpoints belong to the same cluster.

## Getting Started

To run the code in this repository, you'll need to have Python 3 and the following packages installed:

- numpy
- networkx
- matplotlib

You can install these packages using pip:

```
pip install numpy networkx matplotlib
```

Once you have the necessary packages installed, you can run the code using the following command:

```
python main.py <graph_file> <output_dir>
```

Where `<graph_file>` is the path to the input graph file (in the format described below) and `<output_dir>` is the path to the output directory where the results will be saved.

## Input Graph Format

The input graph should be a text file with the following format:

```
<num_vertices> <num_edges>
<edge_1_start> <edge_1_end>
<edge_2_start> <edge_2_end>
...
<edge_n_start> <edge_n_end>
```

Where `<num_vertices>` is the number of vertices in the graph, `<num_edges>` is the number of edges in the graph, and each `<edge_i_start>` and `<edge_i_end>` is a pair of integers representing the endpoints of the i-th edge.

## Output Format

The output of the algorithm is a partition of the input graph into two subgraphs, represented as two lists of vertices. The output is saved as two text files in the output directory, with the following format:

```
<vertex_1>
<vertex_2>
...
<vertex_n>
```

Where each `<vertex_i>` is an integer representing a vertex in the subgraph.
