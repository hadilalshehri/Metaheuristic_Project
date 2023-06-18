# Graph Partitioning using Multistart and Genetic Algorithms

This project implements a metaheuristic algorithm for graph partitioning optimization problem using a local serach, multistart and genetic algorithms. 

## Optimization Problem 
Solving the optimization problem: Given a weighted graph G=(V,E) where V is a set of n nodes and E is a set of edges, let wi ≥ 0 be the weight of node i ∈ V and let cij ({i, j} ∈ E) be the edge weight between nodes i and j (cij =0, if {i, j} ∉ E). A clustering/grouping problem is to partition V into a given number p (p ≤ n) of disjoint clusters or groups such that the sum of node weights in each cluster is constrained by an upper and a lower capacity limit, while maximizing the sum of edge weights whose two associated endpoints belong to the same cluster.

## Getting Started

To run the code in this repository, you'll need to have Python 3 and the following packages installed:

- matplotlib
- random
-  time
-  statistics
-  pandas
-  itertools 
You can install these packages using pip:

```
pip install tplotlib
pip install random 
pip install time
pip install statistics
pip install pandas
pip install itertools

```

Once you have the necessary packages installed, you can run the code using the following command:

```
python main.py <graph_file> <output_dir>
```

Where `<graph_file>` is the path to the input graph file (in the format described below) and `<output_dir>` is the path to the output directory where the results will be saved.

