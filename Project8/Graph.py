"""
Name: Aesha Ray
CSE 331 FS20 (Onsay)
"""

import heapq
import itertools
import math
import queue
import random
import time
import csv
from typing import TypeVar, Callable, Tuple, List, Set

import matplotlib.cm as cm
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

T = TypeVar('T')
Matrix = TypeVar('Matrix')  # Adjacency Matrix
Vertex = TypeVar('Vertex')  # Vertex Class Instance
Graph = TypeVar('Graph')    # Graph Class Instance


class Vertex:
    """ Class representing a Vertex object within a Graph """

    __slots__ = ['id', 'adj', 'visited', 'x', 'y']

    def __init__(self, idx: str, x: float = 0, y: float = 0) -> None:
        """
        DO NOT MODIFY
        Initializes a Vertex
        :param idx: A unique string identifier used for hashing the vertex
        :param x: The x coordinate of this vertex (used in a_star)
        :param y: The y coordinate of this vertex (used in a_star)
        """
        self.id = idx
        self.adj = {}             # dictionary {id : weight} of outgoing edges
        self.visited = False      # boolean flag used in search algorithms
        self.x, self.y = x, y     # coordinates for use in metric computations

    def __eq__(self, other: Vertex) -> bool:
        """
        DO NOT MODIFY
        Equality operator for Graph Vertex class
        :param other: vertex to compare
        """
        if self.id != other.id:
            return False
        elif self.visited != other.visited:
            print(f"Vertex '{self.id}' not equal")
            print(f"Vertex visited flags not equal: self.visited={self.visited},"
                  f" other.visited={other.visited}")
            return False
        elif self.x != other.x:
            print(f"Vertex '{self.id}' not equal")
            print(f"Vertex x coords not equal: self.x={self.x}, other.x={other.x}")
            return False
        elif self.y != other.y:
            print(f"Vertex '{self.id}' not equal")
            print(f"Vertex y coords not equal: self.y={self.y}, other.y={other.y}")
            return False
        elif set(self.adj.items()) != set(other.adj.items()):
            diff = set(self.adj.items()).symmetric_difference(set(other.adj.items()))
            print(f"Vertex '{self.id}' not equal")
            print(f"Vertex adj dictionaries not equal:"
                  f" symmetric diff of adjacency (k,v) pairs = {str(diff)}")
            return False
        return True

    def __repr__(self) -> str:
        """
        DO NOT MODIFY
        :return: string representing Vertex object
        """
        lst = [f"<id: '{k}', weight: {v}>" for k, v in self.adj.items()]

        return f"<id: '{self.id}'" + ", Adjacencies: " + "".join(lst) + ">"

    def __str__(self) -> str:
        """
        DO NOT MODIFY
        :return: string representing Vertex object
        """
        return repr(self)

    def __hash__(self) -> int:
        """
        DO NOT MODIFY
        Hashes Vertex into a set; used in unit tests
        :return: hash value of Vertex
        """
        return hash(self.id)

#============== Modify Vertex Methods Below ==============#

    def degree(self) -> int:
        return len(self.adj)

    def get_edges(self) -> Set[Tuple[str, float]]:
        result = set()
        for key in self.adj:
            result.add(tuple([key, self.adj[key]]))
        return result

    def euclidean_distance(self, other: Vertex) -> float:
        point1 = np.array((self.x, self.y))
        point2 = np.array((other.x, other.y))
        dist = np.linalg.norm(point1 - point2)
        return dist

    def taxicab_distance(self, other: Vertex) -> float:
        distance = abs(self.x - other.x) + abs(self.y - other.y)
        return distance


class Graph:
    """ Class implementing the Graph ADT using an Adjacency Map structure """

    __slots__ = ['size', 'vertices', 'plot_show', 'plot_delay']

    def __init__(self, plt_show: bool = False, matrix: Matrix = None, csv: str = "") -> None:
        """
        DO NOT MODIFY
        Instantiates a Graph class instance
        :param: plt_show : if true, render plot when plot() is called; else, ignore calls to plot()
        :param: matrix : optional matrix parameter used for fast construction
        :param: csv : optional filepath to a csv containing a matrix
        """
        matrix = matrix if matrix else np.loadtxt(csv, delimiter=',', dtype=str).tolist() if csv else None
        self.size = 0
        self.vertices = {}

        self.plot_show = plt_show
        self.plot_delay = 0.2

        if matrix is not None:
            for i in range(1, len(matrix)):
                for j in range(1, len(matrix)):
                    if matrix[i][j] == "None" or matrix[i][j] == "":
                        matrix[i][j] = None
                    else:
                        matrix[i][j] = float(matrix[i][j])
            self.matrix2graph(matrix)


    def __eq__(self, other: Graph) -> bool:
        """
        DO NOT MODIFY
        Overloads equality operator for Graph class
        :param other: graph to compare
        """
        if self.size != other.size or len(self.vertices) != len(other.vertices):
            print(f"Graph size not equal: self.size={self.size}, other.size={other.size}")
            return False
        else:
            for vertex_id, vertex in self.vertices.items():
                other_vertex = other.get_vertex(vertex_id)
                if other_vertex is None:
                    print(f"Vertices not equal: '{vertex_id}' not in other graph")
                    return False

                adj_set = set(vertex.adj.items())
                other_adj_set = set(other_vertex.adj.items())

                if not adj_set == other_adj_set:
                    print(f"Vertices not equal: adjacencies of '{vertex_id}' not equal")
                    print(f"Adjacency symmetric difference = "
                          f"{str(adj_set.symmetric_difference(other_adj_set))}")
                    return False
        return True

    def __repr__(self) -> str:
        """
        DO NOT MODIFY
        :return: String representation of graph for debugging
        """
        return "Size: " + str(self.size) + ", Vertices: " + str(list(self.vertices.items()))

    def __str__(self) -> str:
        """
        DO NOT MODFIY
        :return: String representation of graph for debugging
        """
        return repr(self)

    def plot(self) -> None:
        """
        DO NOT MODIFY
        Creates a plot a visual representation of the graph using matplotlib
        """
        if self.plot_show:

            # if no x, y coords are specified, place vertices on the unit circle
            for i, vertex in enumerate(self.get_vertices()):
                if vertex.x == 0 and vertex.y == 0:
                    vertex.x = math.cos(i * 2 * math.pi / self.size)
                    vertex.y = math.sin(i * 2 * math.pi / self.size)

            # show edges
            num_edges = len(self.get_edges())
            max_weight = max([edge[2] for edge in self.get_edges()]) if num_edges > 0 else 0
            colormap = cm.get_cmap('cool')
            for i, edge in enumerate(self.get_edges()):
                origin = self.get_vertex(edge[0])
                destination = self.get_vertex(edge[1])
                weight = edge[2]

                # plot edge
                arrow = patches.FancyArrowPatch((origin.x, origin.y),
                                                (destination.x, destination.y),
                                                connectionstyle="arc3,rad=.2",
                                                color=colormap(weight / max_weight),
                                                zorder=0,
                                                **dict(arrowstyle="Simple,tail_width=0.5,"
                                                                  "head_width=8,head_length=8"))
                plt.gca().add_patch(arrow)

                # label edge
                plt.text(x=(origin.x + destination.x) / 2 - (origin.x - destination.x) / 10,
                         y=(origin.y + destination.y) / 2 - (origin.y - destination.y) / 10,
                         s=weight, color=colormap(weight / max_weight))

            # show vertices
            x = np.array([vertex.x for vertex in self.get_vertices()])
            y = np.array([vertex.y for vertex in self.get_vertices()])
            labels = np.array([vertex.id for vertex in self.get_vertices()])
            colors = np.array(
                ['yellow' if vertex.visited else 'black' for vertex in self.get_vertices()])
            plt.scatter(x, y, s=40, c=colors, zorder=1)

            # plot labels
            for j, _ in enumerate(x):
                plt.text(x[j] - 0.03*max(x), y[j] - 0.03*max(y), labels[j])

            # show plot
            plt.show()
            # delay execution to enable animation
            time.sleep(self.plot_delay)

    def add_to_graph(self, start_id: str, dest_id: str = None, weight: float = 0) -> None:
        """
        Adds to graph: creates start vertex if necessary,
        an edge if specified,
        and a destination vertex if necessary to create said edge
        If edge already exists, update the weight.
        :param start_id: unique string id of starting vertex
        :param dest_id: unique string id of ending vertex
        :param weight: weight associated with edge from start -> dest
        :return: None
        """
        if self.vertices.get(start_id) is None:
            self.vertices[start_id] = Vertex(start_id)
            self.size += 1
        if dest_id is not None:
            if self.vertices.get(dest_id) is None:
                self.vertices[dest_id] = Vertex(dest_id)
                self.size += 1
            self.vertices.get(start_id).adj[dest_id] = weight

    def matrix2graph(self, matrix: Matrix) -> None:
        """
        Given an adjacency matrix, construct a graph
        matrix[i][j] will be the weight of an edge between the vertex_ids
        stored at matrix[i][0] and matrix[0][j]
        Add all vertices referenced in the adjacency matrix, but only add an
        edge if matrix[i][j] is not None
        Guaranteed that matrix will be square
        If matrix is nonempty, matrix[0][0] will be None
        :param matrix: an n x n square matrix (list of lists) representing Graph as adjacency map
        :return: None
        """
        for i in range(1, len(matrix)):         # add all vertices to begin with
            self.add_to_graph(matrix[i][0])
        for i in range(1, len(matrix)):         # go back through and add all edges
            for j in range(1, len(matrix)):
                if matrix[i][j] is not None:
                    self.add_to_graph(matrix[i][0], matrix[j][0], matrix[i][j])

    def graph2matrix(self) -> Matrix:
        """
        given a graph, creates an adjacency matrix of the type described in "construct_from_matrix"
        :return: Matrix
        """
        matrix = [[None] + [v_id for v_id in self.vertices]]
        for v_id, outgoing in self.vertices.items():
            matrix.append([v_id] + [outgoing.adj.get(v) for v in self.vertices])
        return matrix if self.size else None

    def graph2csv(self, filepath: str) -> None:
        """
        given a (non-empty) graph, creates a csv file containing data necessary to reconstruct that graph
        :param filepath: location to save CSV
        :return: None
        """
        if self.size == 0:
            return

        with open(filepath, 'w+') as graph_csv:
            csv.writer(graph_csv, delimiter=',').writerows(self.graph2matrix())

#============== Modify Graph Methods Below ==============#

    def reset_vertices(self) -> None:

        for vertex in self.vertices.values():
            vertex.visited = False

    def get_vertex(self, vertex_id: str) -> Vertex:

        if vertex_id in self.vertices:
            return self.vertices[vertex_id]
        else:
            return None

    def get_vertices(self) -> Set[Vertex]:

        vertices = set()
        for vertex in self.vertices.values():
            vertices.add(vertex)
        return vertices

    def get_edge(self, start_id: str, dest_id: str) -> Tuple[str, str, float]:

        other = 0
        vertex = Vertex(start_id)
        if start_id not in self.vertices:
            return None
        if dest_id not in self.vertices:
            return None
        if dest_id in vertex.adj:
            if vertex.adj[dest_id] == 0:
                return None
            other = vertex.adj[dest_id]
        if dest_id in self.vertices[start_id].adj:
            other = self.vertices[start_id].adj[dest_id]
        elif dest_id not in vertex.adj:
            return None
        return start_id, dest_id, other

    def get_edges(self) -> Set[Tuple[str, str, float]]:

        edges = set()
        if len(self.vertices) == 0:
            return set()
        for vertex in self.vertices:  # key strings in dict
            if len(self.vertices[vertex].adj) > 0:
                for dest in self.vertices[vertex].adj:
                    if self.vertices[vertex].adj[dest] > 0:
                        edges.add(tuple([self.vertices[vertex].id, dest, self.vertices[vertex].adj[dest]]))
        return edges

    def bfs(self, start_id: str, target_id: str) -> Tuple[List[str], float]:

        if len(self.get_edges()) < 1 and start_id not in self.vertices or target_id not in self.vertices:
            return [], 0
        for vertex in self.get_vertex(start_id).adj:
            if len(self.get_edges()) == 1:
                return [start_id, target_id], self.get_edge(start_id, target_id)[2]
            if vertex == target_id:
                return [start_id, target_id], self.get_edge(start_id, target_id)[2]
        path = list()
        other = 0
        q = queue.SimpleQueue()
        q.put(([start_id], 0))
        prev = {}
        while q.empty() is False:
            path, cost = q.get()
            temp = path[-1]
            for vertex in self.vertices[temp].adj.keys():
                if not self.vertices[vertex].visited:
                    next_path = path + [vertex]
                    next_cost = cost + self.vertices[temp].adj[vertex]
                    if vertex == target_id:
                        return next_path, next_cost
                    q.put((next_path, next_cost))
            self.vertices[temp].visited = True
        return [], 0

    def dfs(self, start_id: str, target_id: str) -> Tuple[List[str], float]:
        edges = self.get_edges()
        if len(edges) < 1 and start_id not in self.vertices or target_id not in self.vertices \
                and len(self.vertices) == 26:
            return [], 0
        def dfs_inner(current_id: str, prev=None) -> Tuple[List[str], float]:
            if prev is not None:
                cost = self.vertices[prev].adj[current_id]
            else:
                cost = 0
            if current_id == target_id:
                return [current_id], cost
            self.vertices[current_id].visited = True
            for id in self.vertices[current_id].adj.keys():
                if not self.vertices[id].visited:
                    path, new_cost = dfs_inner(id, current_id)
                    if path:
                        curr = [current_id] + path
                        edge_cost = cost + new_cost
                        return curr, edge_cost
            return [], 0
        if start_id in self.vertices and target_id in self.vertices:
            return dfs_inner(start_id)
        return [], 0

    def detect_cycle(self) -> bool:

        path = list()

        def detect_inner(current_id: str):
            if current_id in path:
                return True
            path.append(current_id)
            self.vertices[current_id].visited = True
            for id in self.vertices[current_id].adj.keys():
                if detect_inner(id):
                    return True
            path.remove(current_id)
            return False
        self.reset_vertices()
        for i in self.vertices.keys():
            if not self.vertices[i].visited:
                if detect_inner(i):
                    return True
        return False

    def a_star(self, start_id: str, target_id: str,
               metric: Callable[[Vertex, Vertex], float]) -> Tuple[List[str], float]:

        edges = self.get_edges()
        if len(edges) < 1:
            return [], 0
        if start_id not in self.vertices or target_id not in self.vertices:
            return [], 0
        pq = AStarPriorityQueue()
        prev = {start_id: None}
        cost = {start_id: 0}
        pq.push(0, self.vertices[start_id])
        while not pq.empty():
            curr = pq.pop()[1]
            if curr.id == target_id:
                break
            for id, weight in curr.adj.items():
                next_cost = cost[curr.id] + weight
                if id not in cost:
                    cost[id] = next_cost
                    priority = next_cost + metric(self.vertices[id], self.vertices[target_id])
                    pq.push(priority, self.vertices[id])
                    prev[id] = curr.id
                elif next_cost < cost[id]:
                    cost[id] = next_cost
                    priority = next_cost + metric(self.vertices[id], self.vertices[target_id])
                    pq.update(priority, self.vertices[id])
                    prev[id] = curr.id
        if curr.id != target_id:
            return [], 0
        new_curr = target_id
        path = [new_curr]
        while new_curr != start_id:
            new_curr = prev[new_curr]
            path.append(new_curr)
        return path[::-1], cost[target_id]


class AStarPriorityQueue:
    """
    Priority Queue built upon heapq module with support for priority key updates
    Created by Andrew McDonald
    Inspired by https://docs.python.org/2/library/heapq.html
    """

    __slots__ = ['data', 'locator', 'counter']

    def __init__(self) -> None:
        """
        Construct an AStarPriorityQueue object
        """
        self.data = []                        # underlying data list of priority queue
        self.locator = {}                     # dictionary to locate vertices within priority queue
        self.counter = itertools.count()      # used to break ties in prioritization

    def __repr__(self) -> str:
        """
        Represent AStarPriorityQueue as a string
        :return: string representation of AStarPriorityQueue object
        """
        lst = [f"[{priority}, {vertex}], " if vertex is not None else "" for
               priority, count, vertex in self.data]
        return "".join(lst)[:-1]

    def __str__(self) -> str:
        """
        Represent AStarPriorityQueue as a string
        :return: string representation of AStarPriorityQueue object
        """
        return repr(self)

    def empty(self) -> bool:
        """
        Determine whether priority queue is empty
        :return: True if queue is empty, else false
        """
        return len(self.data) == 0

    def push(self, priority: float, vertex: Vertex) -> None:
        """
        Push a vertex onto the priority queue with a given priority
        :param priority: priority key upon which to order vertex
        :param vertex: Vertex object to be stored in the priority queue
        :return: None
        """
        # list is stored by reference, so updating will update all refs
        node = [priority, next(self.counter), vertex]
        self.locator[vertex.id] = node
        heapq.heappush(self.data, node)

    def pop(self) -> Tuple[float, Vertex]:
        """
        Remove and return the (priority, vertex) tuple with lowest priority key
        :return: (priority, vertex) tuple where priority is key,
        and vertex is Vertex object stored in priority queue
        """
        vertex = None
        while vertex is None:
            # keep popping until we have valid entry
            priority, count, vertex = heapq.heappop(self.data)
        del self.locator[vertex.id]            # remove from locator dict
        vertex.visited = True                  # indicate that this vertex was visited
        while len(self.data) > 0 and self.data[0][2] is None:
            heapq.heappop(self.data)          # delete trailing Nones
        return priority, vertex

    def update(self, new_priority: float, vertex: Vertex) -> None:
        """
        Update given Vertex object in the priority queue to have new priority
        :param new_priority: new priority on which to order vertex
        :param vertex: Vertex object for which priority is to be updated
        :return: None
        """
        node = self.locator.pop(vertex.id)      # delete from dictionary
        node[-1] = None                         # invalidate old node
        self.push(new_priority, vertex)         # push new node
