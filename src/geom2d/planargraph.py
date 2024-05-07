"""Simple planar graph data structure."""

from __future__ import annotations

import enum
import random
from typing import TYPE_CHECKING

from . import polygon
from .box import Box
from .line import Line, TLine
from .point import P, TPoint
from .util import normalize_angle

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from typing_extensions import TypeAlias


class GraphNode:
    """Graph node.

    A node has a vertex and a list of outgoing nodes that
    define the outgoing edges that connect to other nodes
    in the graph.
    """

    # Node vertex point
    vertex: P
    # Connected edges
    edge_nodes: list[GraphNode]
    # Cached edge angles for quick online insertion sort
    edge_angles: list[float]

    def __init__(
        self, vertex: P, edge_nodes: Iterable[GraphNode] | None = None
    ) -> None:
        """Graph node.

        Args:
            vertex: The vertex associated with this node.
            edge_nodes: Endpoints of outgoing edges.
        """
        self.vertex = vertex
        self.edge_nodes = []
        self.edge_angles = []
        if edge_nodes is not None:
            for node in edge_nodes:
                self.add_edge_node(node)

    def degree(self) -> int:
        """Number of incident graph edges.

        I.e. number of edges that share this node's vertex.

        See:
            http://mathworld.wolfram.com/VertexOrder.html
        """
        return len(self.edge_nodes)

    def add_edge_node(self, edge_node: GraphNode) -> None:
        """Add an outgoing edge node.

        Args:
            edge_node: Endpoint of outgoing edge.
        """
        i = 0
        ref_point = P(self.vertex.x + 1, self.vertex.y)
        ccw_angle = normalize_angle(
            self.vertex.angle2(ref_point, edge_node.vertex)
        )
        # Perform an online insertion sort
        if self.edge_angles:
            for i, angle in enumerate(self.edge_angles):
                if angle >= ccw_angle:
                    self.edge_nodes.insert(i, edge_node)
                    self.edge_angles.insert(i, ccw_angle)
                    return
        self.edge_nodes.append(edge_node)
        self.edge_angles.append(ccw_angle)

    def remove_edge_node(self, edge_node: GraphNode) -> None:
        """Remove an outgoing edge node."""
        self.edge_nodes.remove(edge_node)

    def sort_edges(self) -> None:
        """Sort outgoing edges in CCW order."""
        ref_point = P(self.vertex.x + 1, self.vertex.y)

        def sortkey(edge_node: GraphNode) -> float:
            ccw_angle = self.vertex.angle2(ref_point, edge_node.vertex)
            return normalize_angle(ccw_angle)

        self.edge_nodes.sort(key=sortkey)

    def ccw_edge_node(
        self, ref_node: GraphNode, skip_spikes: bool = True
    ) -> GraphNode:
        """The most CCW edge node.

        Starting at the reference edge defined by ref_node->this_node.

        Args:
            ref_node: The reference edge node.
            skip_spikes: Skip over edges that connect to nodes of order one.

        Returns:
            The counter-clockwise edge node closest to the reference node
            by angular distance. If all edges nodes are dead ends the
            reference node will be returned.
        """
        # Assume the edges have been already sorted in CCW order.
        node_index = self.edge_nodes.index(ref_node) - 1
        node = self.edge_nodes[node_index]
        while skip_spikes and node.degree() == 1 and node != ref_node:
            node_index -= 1
            node = self.edge_nodes[node_index]

        return node

    def __eq__(self, other: object) -> bool:
        """Compare for equality.

        GraphNodes are considered equal if their vertices are equal.
        This doesn't check if the outgoing edges are the same...
        """
        return isinstance(other, GraphNode) and other.vertex == self.vertex

    def __hash__(self) -> int:
        """Calculate a hash value for the GraphNode vertex."""
        return hash(self.vertex)

    def __str__(self) -> str:
        """For debug output..."""
        return f'{self.vertex} [{len(self.edge_nodes):d}]'


TNodeMap: TypeAlias = dict[P, GraphNode]


class Graph:
    """Simple connected undirected 2D planar graph."""

    edges: set[Line]
    nodemap: TNodeMap
    _bottom_node: GraphNode
    _modified: bool

    def __init__(self, edges: Iterable[TLine] | None = None) -> None:
        """Create a new planar Graph.

        Args:
            edges: An iterable collection of line segments that
                define the graph edges. Each edge connects two nodes.
                An edge being a 2-tuple of endpoints of the form:
                ((x1, y1), (x2, y2)).
        """
        #: Set of graph edges
        self.edges = set()
        #: Map of vertex points to graph nodes.
        self.nodemap = {}
        # Node at the lowest Y axis value.
        self._bottom_node = GraphNode(P.max_point())
        # Graph has been modified - i.e. nodes added or removed.
        self._modified = False

        if edges is not None:
            for edge in edges:
                self.add_edge(edge)

    def add_edge(self, edge: TLine) -> None:
        """Add an edge to this graph.

        Args:
            edge: A line segment that defines a graph edge.
                An edge being a 2-tuple of endpoints of the form:
                ((x1, y1), (x2, y2)).
        """
        edge_p1 = P(edge[0])
        edge_p2 = P(edge[1])
        # Check for degenerate edge...
        if edge_p1 == edge_p2:
            return
        edge = Line(edge_p1, edge_p2)
        if edge not in self.edges:
            # self._check_modified(modify=True)
            self.edges.add(edge)
            # Build the node graph
            node1 = self.nodemap.get(edge_p1)
            if node1 is None:
                node1 = self._create_node(edge_p1)
            node2 = self.nodemap.get(edge_p2)
            if node2 is None:
                node2 = self._create_node(edge_p2)
            node1.add_edge_node(node2)
            node2.add_edge_node(node1)
            # Update bottom node
            if edge_p1.y < self._bottom_node.vertex.y:
                self._bottom_node = node1
            if edge_p2.y < self._bottom_node.vertex.y:
                self._bottom_node = node2

    def remove_edge(self, edge: TLine) -> None:
        """Remove and unlink the specified edge from the graph.

        Args:
            edge: A line segment that defines a graph edge
                connecting two nodes.
                An edge being a 2-tuple of endpoints of the form:
                ((x1, y1), (x2, y2)).
        """
        # self._check_modified(modify=True)
        p1 = P(edge[0])
        p2 = P(edge[1])
        node1 = self.nodemap[p1]
        node2 = self.nodemap[p2]
        node1.remove_edge_node(node2)
        node2.remove_edge_node(node1)
        if node1.degree() == 0:
            del self.nodemap[p1]
        if node2.degree() == 0:
            del self.nodemap[p2]
        self.edges.remove(Line(edge))

    def add_poly(
        self, vertices: Sequence[TPoint], close_poly: bool = True
    ) -> None:
        """Add polygon edges to this graph.

        Args:
            vertices: A list of polyline/polygon vertices as 2-tuples (x, y).
            close_poly: If True a closing segment will
                be automatically added if absent. Default is True.
        """
        p1 = vertices[0]
        for p2 in vertices[1:]:
            self.add_edge((p1, p2))  # Line(P(p1), P(p2)))
            p1 = p2
        if close_poly and vertices[0] != vertices[-1]:
            self.add_edge((vertices[-1], vertices[0]))
            # self.add_edge(Line(P(vertices[-1]), P(vertices[0])))

    def order(self) -> int:
        """Number of graph nodes (vertices.)."""
        return len(self.nodemap)

    def size(self) -> int:
        """Number of edges."""
        return len(self.edges)

    def vertices(self) -> Iterable[P]:
        """Graph edge vertices."""
        return self.nodemap.keys()

    def bounding_box(self) -> Box:
        """Get the bounding rectangle for this graph.

        Returns:
            A tuple containing two points ((x0, y0), (x1, y1))
            specifying bottom left and top right corners.
        """
        mm = [(min(xy), max(xy)) for xy in zip(*self.vertices())]
        return Box((mm[0][0], mm[1][0]), (mm[0][1], mm[1][1]))

    def boundary_polygon(self) -> list[P]:
        """A polygon defining the outer edges of this segment graph."""
        # self._check_modified(modify=False)
        return self._build_boundary_polygon(self._bottom_node, self.order())

    def peel_boundary_polygon(self, boundary_polygon: Sequence[P]) -> list:
        """Similar to convex hull peeling but with non-convex boundary polygons.

        Args:
            boundary_polygon: The initial graph polygon hull to peel.

        Returns:
            A list of peeled inner polygons. Possibly empty.
        """
        # self._check_modified(modify=False)
        # Make a copy of the graph node map so that pruning won't
        # mutate this graph.
        nodemap = self._copy_nodemap()
        # Peel back the nodes outside and on the boundary
        self._prune_nodes(nodemap, boundary_polygon)
        poly_list = []
        while len(nodemap) > 3 and len(boundary_polygon) > 3:
            # Find the bottom-most node to start polygon march
            bottom_node = self._find_bottom_node(nodemap.values())
            # Get a new boundary polygon from peeled nodes
            boundary_polygon = self._build_boundary_polygon(
                bottom_node, len(nodemap)
            )
            if len(boundary_polygon) > 2:
                poly_list.append(boundary_polygon)
            # Peel the next layer
            self._prune_nodes(nodemap, boundary_polygon)
        return poly_list

    def cull_open_edges(self) -> None:
        """Remove edges that have one or two disconnected endpoints."""
        while True:
            open_edges = [
                edge
                for edge in self.edges
                if (
                    self.nodemap[edge.p1].degree() == 1
                    or self.nodemap[edge.p2].degree() == 1
                )
            ]
            if not open_edges:
                break
            for edge in open_edges:
                self.remove_edge(edge)
        self._bottom_node = self._find_bottom_node(self.nodemap.values())

    def get_face_polygons(self) -> list[list[P]]:
        """Graph face polygons.

        Returns:
            A list of face polygons.
        """
        # self._check_modified(modify=False)
        return _make_face_polygons(self.edges, self.nodemap)

    def _find_bottom_node(self, nodes: Iterable[GraphNode]) -> GraphNode:
        """Find the node that has the minimum Y value.

        Args:
            nodes: An iterable collection of at least one GraphNodes.

        Returns:
            The bottom-most node.
        """
        bottom_node: GraphNode | None = None
        for node in nodes:
            if not bottom_node or node.vertex.y < bottom_node.vertex.y:
                bottom_node = node
        assert bottom_node
        return bottom_node

    def _remove_node(self, nodemap: TNodeMap, node: GraphNode) -> None:
        """Remove and unlink a node from the node map."""
        if node.vertex in nodemap:
            # Remove edges connected to this node.
            for edge_node in node.edge_nodes:
                if edge_node.vertex in nodemap:
                    edge_node.remove_edge_node(node)
                    if edge_node.degree() == 0:
                        # The connected node is now orphaned so remove it also.
                        del nodemap[edge_node.vertex]
            del nodemap[node.vertex]

    def _create_node(self, vertex_point: P) -> GraphNode:
        """Create a graph node and insert it into the graph."""
        node = GraphNode(vertex_point)
        self.nodemap[vertex_point] = node
        return node

    #    def _check_modified(self, modify=False):
    #        """If the graph has been modified by adding or removing
    #        nodes then re-compute graph properties and sort the nodes.
    #        """
    #        if not modify and self._modified:
    #            for node in self.nodemap.values():
    #                node.sort_edges()
    #        self._modified = modify

    def _copy_nodemap(self) -> TNodeMap:
        """Make a copy of the node map and edge connections of this graph."""
        nodemap_copy: TNodeMap = {}
        # Copy the vertex->node mapping
        for vertex in self.nodemap:
            nodemap_copy[vertex] = GraphNode(vertex)
        # Copy edge connections
        for node in nodemap_copy.values():
            # TODO: hoist this into GraphNode
            srcnode = self.nodemap[node.vertex]
            for edge_node in srcnode.edge_nodes:
                node.edge_nodes.append(nodemap_copy[edge_node.vertex])
        return nodemap_copy

    def _prune_nodes(
        self, nodemap: TNodeMap, boundary_polygon: Sequence[P]
    ) -> None:
        """Prune a layer of graph nodes.

        Prune the nodes corresponding to the list of points
        on or outside the specified boundary polygon.
        """
        if boundary_polygon:
            # Delete all the nodes outside the initial polygon.
            deleted_nodes = [
                node
                for node in nodemap.values()
                if not polygon.point_inside(boundary_polygon, node.vertex)
            ]
            for node in deleted_nodes:
                self._remove_node(nodemap, node)
            # Delete the nodes that correspond to the vertices of the polygon
            for vertex in boundary_polygon:
                vnode = nodemap.get(vertex)
                if vnode:
                    self._remove_node(nodemap, vnode)
        # Remove any dead-end spike nodes
        while True:
            # List of vertices that will be deleted from the node map.
            deleted_nodes = [
                node for node in nodemap.values() if node.degree() < 2
            ]
            if not deleted_nodes:
                break
            for node in deleted_nodes:
                self._remove_node(nodemap, node)

    def _build_boundary_polygon(
        self, start_node: GraphNode, num_nodes: int, prune_spikes: bool = True
    ) -> list[P]:
        """Return a polygon defining the outer edges of a graph.

        Args:
            start_node: Should be the bottom-most node in the graph.
            num_nodes: The number of nodes in the graph.
            prune_spikes: Prune dangling edges from the boundary
                polygon. These would be edges that are connected to
                one node. Default is True.
        """
        if not start_node:
            return []
        # debug.draw_point(start_node.vertex, color='#ff0000')
        # If the start node only has one outgoing edge then
        # traverse it until a node with at least two edges is found.
        if prune_spikes and start_node.degree() == 1:
            start_node = start_node.edge_nodes[0]
            num_nodes -= 1
            while start_node.degree() == 2:
                # start_node = start_node.ccw_edge_node()
                start_node = start_node.edge_nodes[0]
            if start_node.degree() == 1:
                # The graph is just a polyline...
                return []
        # debug.draw_point(start_node.vertex, color='#ff0000')
        # Perform a counter-clockwise walk around the outer edges
        # of the graph.
        boundary_polygon = [
            start_node.vertex,
        ]
        curr_node = start_node
        prev_node = start_node.edge_nodes[0]
        while num_nodes > 0:
            next_node = curr_node.ccw_edge_node(prev_node)
            # debug.draw_point(next_node.vertex, color='#00ff00')
            if not prune_spikes or next_node.degree() > 1:
                boundary_polygon.append(next_node.vertex)
            prev_node = curr_node
            curr_node = next_node
            num_nodes -= 1
            if curr_node == start_node:
                break
        return boundary_polygon


class PathStrategy(enum.IntEnum):
    """Path building strategy."""

    STRAIGHTEST = enum.auto()
    SQUIGGLY = enum.auto()
    RANDOM = enum.auto()
    RANDOM2 = enum.auto()


class GraphPathBuilder:
    """Create paths from connected graph edges.

    Given a Graph, build a set of paths made of connected graph edges.
    """

    graph: Graph
    _random: random.Random

    def __init__(self, graph: Graph) -> None:
        self.graph = graph
        self._random = random.Random()
        self._random.seed()

    def build_paths(
        self,
        start_edge: Line | None = None,
        path_strategy: PathStrategy = PathStrategy.STRAIGHTEST,
    ) -> list[list[P]]:
        """Build edge paths.

        Given the starting edge, find the set of edge paths that
        completely fill the graph...

        Args:
            start_edge: The graph edge that starts the path.
            path_strategy: How paths will be constructed. Possible
                path strategies are:

                    STRAIGHTEST, SQUIGGLY,
                    RANDOM, and RANDOM2

        Returns:
            A list of paths sorted by descending order of path length.
        """
        if start_edge is None:
            start_edge = self._random.choice(list(self.graph.edges))
        paths = []
        free_edges = set(self.graph.edges)
        visited_edges: set[Line] = set()
        while free_edges:
            path = self._build_path(start_edge, visited_edges, path_strategy)
            paths.append(path)
            free_edges -= visited_edges
            if free_edges:
                start_edge = self._random.choice(list(free_edges))
        # paths.sort(key=len, reverse=True)
        return paths

    def build_longest_paths(
        self,
        path_strategy: PathStrategy = PathStrategy.STRAIGHTEST,
    ) -> list[list[P]]:
        """Find the longest paths in this graph."""
        path_list = []
        for start_edge in self.graph.edges:
            visited_edges: set[Line] = set()
            path = self._build_path(start_edge, visited_edges, path_strategy)
            path_list.append(path)
        path_list.sort(key=len, reverse=True)
        return self._dedupe_paths(path_list)

    def _build_path(
        self,
        start_edge: Line,
        visited_edges: set[Line],
        path_strategy: PathStrategy,
    ) -> list[P]:
        """Build a path from the starting edge.

        Try both directions and glue the paths together.
        """
        node_a = self.graph.nodemap[start_edge[0]]
        node_b = self.graph.nodemap[start_edge[1]]
        path = self._build_path_forward(
            node_a, node_b, visited_edges, path_strategy
        )
        path_rev = self._build_path_forward(
            node_b, node_a, visited_edges, path_strategy
        )
        if len(path_rev) > 2:
            path.reverse()
            path.extend(path_rev[2:])
        return path

    def _build_path_forward(
        self,
        prev_node: GraphNode,
        curr_node: GraphNode,
        visited_edges: set[Line],
        path_strategy: PathStrategy,
    ) -> list[P]:
        """Build a forward path.

        Starting at the specified node, follow outgoing edges until
        its no longer possible. Sort of a half-assed Euler tour...
        """
        path = [prev_node.vertex]
        next_node: GraphNode | None = curr_node
        while next_node:
            path.append(next_node.vertex)
            curr_node = next_node
            next_node = self._get_exit_edge_node(
                prev_node, curr_node, visited_edges, path_strategy
            )
            edge = Line(prev_node.vertex, curr_node.vertex)
            visited_edges.add(edge)
            prev_node = curr_node
        return path

    def _get_exit_edge_node(
        self,
        prev_node: GraphNode,
        curr_node: GraphNode,
        visited_edges: set[Line],
        path_strategy: PathStrategy,
    ) -> GraphNode | None:
        """Find an exit node that satisfies the path strategy.

        If all exit nodes define edges that have been already visited
        then return None.
        """
        if curr_node.degree() == 1:
            # End of the line...
            return None

        # List of potential exit nodes from the current node.
        exit_node_list = []
        for exit_node in curr_node.edge_nodes:
            if exit_node != prev_node:
                edge = Line(curr_node.vertex, exit_node.vertex)
                if edge not in visited_edges:
                    exit_node_list.append(exit_node)

        if exit_node_list:
            # Sort the exit nodes in order of angular distance
            # from incoming edge.
            def sortkey(node: GraphNode) -> float:
                return abs(
                    curr_node.vertex.angle2(prev_node.vertex, node.vertex)
                )

            exit_node_list.sort(key=sortkey, reverse=True)
            if path_strategy == PathStrategy.SQUIGGLY:
                exit_node = exit_node_list[-1]
            elif path_strategy == PathStrategy.RANDOM:
                exit_node = self._random.choice(exit_node_list)
            elif path_strategy == PathStrategy.RANDOM2:
                # A random choice weighted towards straighter paths
                exit_node_list.insert(0, exit_node_list[0])
                exit_node = self._random.choice(exit_node_list[0:3])
            else:
                exit_node = exit_node_list[0]
            return exit_node

        return None

    def _dedupe_paths(
        self, path_list: list[list[P]], min_difference: int = 2
    ) -> list[list[P]]:
        """Remove similar paths from a list of paths."""
        deduped_path_list = [
            path_list[0],
        ]
        prev_path = path_list[0]
        for path in path_list[1:]:
            pathset = frozenset(prev_path)
            if len(pathset.difference(path)) > min_difference:
                deduped_path_list.append(path)
            prev_path = path
        return deduped_path_list


class MarkedEdge:
    """A graph edge that is used to keep track of graph traversal direction."""

    edge: Line
    # True if traversed in direction p2->p1, CCW winding
    visited_p1: bool = False
    # True if traversed in direction p1->p2, CCW winding
    visited_p2: bool = False

    def __init__(self, edge: Line) -> None:
        """Create a MarkedEdge.

        Args:
            edge: A graph edge (a Line segment).
        """
        self.edge = edge

    def visited_left(self, dest_vertex: P) -> bool:
        """True if this edge has been visited with a CCW winding.

        The edge will be marked as visited.

        Args:
            dest_vertex: The destination vertex.
                Determines which side of the edge has been visited
                (i.e. direction).

        Returns:
            True if this edge has been visited during a counter-clockwise
            traversal. I.e. the left side given the direction.
            Otherwise False.
        """
        if dest_vertex == self.edge.p1:
            is_visited = self.visited_p1
            self.visited_p1 = True
        elif dest_vertex == self.edge.p2:
            is_visited = self.visited_p2
            self.visited_p2 = True
        else:
            raise AssertionError
        return is_visited


class MarkedEdgeMap:
    """Map of edges to marked edges."""

    _edgemap: dict[Line, MarkedEdge]

    def __init__(self, edges: Iterable[Line]) -> None:
        """Create a MarkedEdgeMap."""
        self._edgemap = {edge: MarkedEdge(edge) for edge in edges}
        # for edge in edges:
        #    self._edgemap[edge] = MarkedEdge(edge)

    def lookup(self, p1: P, p2: P) -> MarkedEdge:
        """Find a MarkedEdge by edge endpoints."""
        return self._edgemap[Line(p1, p2)]

    def mark_edge(self, p1: P, p2: P) -> None:
        """Mark an edge segment as visited."""
        marked_edge = self.lookup(p1, p2)
        marked_edge.visited_left(p2)


def _make_face_polygons(
    edges: Iterable[Line], nodemap: TNodeMap
) -> list[list[P]]:
    """Given a graph, make polygons from graph faces delineated by edges.

    Args:
        edges: Graph edges.
        nodemap: A mapping of edges to nodes.
    """
    # First mark the outside edges
    edgemap = MarkedEdgeMap(edges)
    faces = []
    for start_node in nodemap.values():
        # Find a free outgoing edge to start the walk
        next_node = find_free_edge_node(edgemap, start_node)
        while next_node:
            face = _make_face(edgemap, start_node, next_node)
            if face and len(face) > 2:
                faces.append(face)
            # Keep going while there are free outgoing edges....
            next_node = find_free_edge_node(edgemap, start_node)
    return faces


def _make_face(
    edgemap: MarkedEdgeMap, start_node: GraphNode, next_node: GraphNode
) -> list[P] | None:
    """Create face polygon.

    Args:
        edgemap: Map of marked visited edges.
        start_node: Graph node to start face polygon.
        next_node: First outgoing face vertex.
    """
    # Start the counterclockwise walk
    face = [
        start_node.vertex,
        next_node.vertex,
    ]
    prev_node = start_node
    curr_node = next_node
    while next_node != start_node and next_node.degree() > 1:
        next_node = curr_node.ccw_edge_node(prev_node, skip_spikes=False)
        if next_node.degree() == 1:
            edgemap.mark_edge(curr_node.vertex, next_node.vertex)
        else:
            face.append(next_node.vertex)
        edgemap.mark_edge(curr_node.vertex, next_node.vertex)
        prev_node = curr_node
        curr_node = next_node
    # Discard open, inside-out (clockwise wound), or unbounded faces.
    if next_node != start_node or polygon.area(face) > 0:
        return None
    return face


def find_free_edge_node(
    edgemap: MarkedEdgeMap, start_node: GraphNode
) -> GraphNode | None:
    """Find a free outgoing edge that is unmarked for CCW edge traversal.

    Args:
        edgemap: Map of marked visited edges.
        start_node: Graph node to start face polygon.
    """
    for edge_node in start_node.edge_nodes:
        edge = edgemap.lookup(start_node.vertex, edge_node.vertex)
        if not edge.visited_left(edge_node.vertex) and edge_node.degree() > 1:
            return edge_node
    return None
