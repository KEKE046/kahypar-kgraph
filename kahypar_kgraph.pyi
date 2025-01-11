from typing import List, Iterator, Any, Union

class Hypergraph:
    def __init__(self, num_nodes: int, num_edges: int, index_vector: List[int], edge_vector: List[int], k: int) -> None:
        """
        Construct an unweighted hypergraph.

        :param num_nodes: Number of nodes
        :param num_edges: Number of hyperedges
        :param index_vector: Starting indices for each hyperedge
        :param edge_vector: Vector containing all hyperedges
        :param k: Number of blocks in which the hypergraph should be partitioned
        """
        ...

    def __init__(self, num_nodes: int, num_edges: int, index_vector: List[int], edge_vector: List[int], k: int, edge_weights: List[float], node_weights: List[float]) -> None:
        """
        Construct a hypergraph with node and edge weights.

        If only one type of weights is required, the other argument has to be an empty list.

        :param num_nodes: Number of nodes
        :param num_edges: Number of hyperedges
        :param index_vector: Starting indices for each hyperedge
        :param edge_vector: Vector containing all hyperedges
        :param k: Number of blocks in which the hypergraph should be partitioned
        :param edge_weights: Weights of all hyperedges
        :param node_weights: Weights of all hypernodes
        """
        ...

    def printGraphState(self) -> None:
        """
        Print the hypergraph state (for debugging purposes).
        """
        ...

    def nodeDegree(self, node: int) -> int:
        """
        Get the degree of the node.

        :param node: The node to get the degree of.
        :return: The degree of the node.
        """
        ...

    def edgeSize(self, hyperedge: int) -> int:
        """
        Get the size of the hyperedge.

        :param hyperedge: The hyperedge to get the size of.
        :return: The size of the hyperedge.
        """
        ...

    def nodeWeight(self, node: int) -> float:
        """
        Get the weight of the node.

        :param node: The node to get the weight of.
        :return: The weight of the node.
        """
        ...

    def edgeWeight(self, hyperedge: int) -> float:
        """
        Get the weight of the hyperedge.

        :param hyperedge: The hyperedge to get the weight of.
        :return: The weight of the hyperedge.
        """
        ...

    def blockID(self, node: int) -> int:
        """
        Get the block of the node in the current hypergraph partition (before partitioning: -1).

        :param node: The node to get the block ID of.
        :return: The block ID of the node.
        """
        ...

    def numNodes(self) -> int:
        """
        Get the number of nodes.

        :return: The number of nodes.
        """
        ...

    def numEdges(self) -> int:
        """
        Get the number of hyperedges.

        :return: The number of hyperedges.
        """
        ...

    def numPins(self) -> int:
        """
        Get the number of pins.

        :return: The number of pins.
        """
        ...

    def numBlocks(self) -> int:
        """
        Get the number of blocks.

        :return: The number of blocks.
        """
        ...

    def numPinsInBlock(self, hyperedge: int, block: int) -> int:
        """
        Get the number of pins of the hyperedge that are assigned to corresponding block.

        :param hyperedge: The hyperedge to check.
        :param block: The block to check.
        :return: The number of pins in the block.
        """
        ...

    def connectivity(self, hyperedge: int) -> int:
        """
        Get the connectivity of the hyperedge (i.e., the number of blocks which contain at least one pin).

        :param hyperedge: The hyperedge to check.
        :return: The connectivity of the hyperedge.
        """
        ...

    def connectivitySet(self, hyperedge: int) -> 'ConnectivitySet':
        """
        Get the connectivity set of the hyperedge.

        :param hyperedge: The hyperedge to check.
        :return: The connectivity set of the hyperedge.
        """
        ...

    def communities(self) -> Any:
        """
        Get the community structure.

        :return: The community structure.
        """
        ...

    def blockWeight(self, block: int) -> float:
        """
        Get the weight of the block.

        :param block: The block to get the weight of.
        :return: The weight of the block.
        """
        ...

    def blockSize(self, block: int) -> int:
        """
        Get the number of vertices in the block.

        :param block: The block to get the size of.
        :return: The number of vertices in the block.
        """
        ...

    def reset(self) -> None:
        """
        Reset the hypergraph to its initial state.
        """
        ...

    def fixNodeToBlock(self, node: int, block: int) -> None:
        """
        Fix node to the corresponding block.

        :param node: The node to fix.
        :param block: The block to fix the node to.
        """
        ...

    def numFixedNodes(self) -> int:
        """
        Get the number of fixed nodes in the hypergraph.

        :return: The number of fixed nodes.
        """
        ...

    def containsFixedNodes(self) -> bool:
        """
        Return true if the hypergraph contains nodes fixed to a specific block.

        :return: True if the hypergraph contains fixed nodes.
        """
        ...

    def isFixedNode(self, node: int) -> bool:
        """
        Return true if the node is fixed to a block.

        :param node: The node to check.
        :return: True if the node is fixed.
        """
        ...

    def nodes(self) -> Iterator[int]:
        """
        Iterate over all nodes.

        :return: An iterator over all nodes.
        """
        ...

    def edges(self) -> Iterator[int]:
        """
        Iterate over all hyperedges.

        :return: An iterator over all hyperedges.
        """
        ...

    def pins(self, hyperedge: int) -> Iterator[int]:
        """
        Iterate over all pins of the hyperedge.

        :param hyperedge: The hyperedge to iterate over.
        :return: An iterator over all pins of the hyperedge.
        """
        ...

    def incidentEdges(self, node: int) -> Iterator[int]:
        """
        Iterate over all incident hyperedges of the node.

        :param node: The node to iterate over.
        :return: An iterator over all incident hyperedges.
        """
        ...

class ConnectivitySet:
    def contains(self, block: int) -> bool:
        """
        Check if the block is contained in the connectivity set of the hyperedge.

        :param block: The block to check.
        :return: True if the block is contained in the connectivity set.
        """
        ...

    def __iter__(self) -> Iterator[int]:
        """
        Iterate over all blocks contained in the connectivity set of the hyperedge.

        :return: An iterator over all blocks in the connectivity set.
        """
        ...

def createHypergraphFromFile(filename: str, k: int) -> Hypergraph:
    """
    Construct a hypergraph from a file in hMETIS format.

    :param filename: The path to the file.
    :param k: The number of blocks.
    :return: A hypergraph constructed from the file.
    """
    ...

def partition(hypergraph: Hypergraph, context: 'Context') -> None:
    """
    Compute a k-way partition of the hypergraph.

    :param hypergraph: The hypergraph to partition.
    :param context: The context for partitioning.
    """
    ...

def cut(hypergraph: Hypergraph) -> float:
    """
    Compute the cut-net metric for the partitioned hypergraph.

    :param hypergraph: The hypergraph to compute the metric for.
    :return: The cut-net metric.
    """
    ...

def soed(hypergraph: Hypergraph) -> float:
    """
    Compute the sum-of-external-degrees metric for the partitioned hypergraph.

    :param hypergraph: The hypergraph to compute the metric for.
    :return: The sum-of-external-degrees metric.
    """
    ...

def connectivityMinusOne(hypergraph: Hypergraph) -> float:
    """
    Compute the connectivity metric for the partitioned hypergraph.

    :param hypergraph: The hypergraph to compute the metric for.
    :return: The connectivity metric.
    """
    ...

def imbalance(hypergraph: Hypergraph, context: 'Context') -> float:
    """
    Compute the imbalance of the hypergraph partition.

    :param hypergraph: The hypergraph to compute the imbalance for.
    :param context: The context for partitioning.
    :return: The imbalance of the partition.
    """
    ...

class Context:
    def __init__(self) -> None:
        """
        Initialize a new Context object.
        """
        ...

    def setK(self, k: int) -> None:
        """
        Set the number of blocks the hypergraph should be partitioned into.

        :param k: The number of blocks.
        """
        ...

    def setEpsilon(self, eps: float) -> None:
        """
        Set the allowed imbalance epsilon.

        :param eps: The imbalance parameter epsilon.
        """
        ...

    def setCustomTargetBlockWeights(self, custom_target_weights: List[float]) -> None:
        """
        Assigns each block of the partition an individual maximum allowed block weight.

        :param custom_target_weights: The list of custom target block weights.
        """
        ...

    def setSeed(self, seed: int) -> None:
        """
        Set the seed for the random number generator.

        :param seed: The seed value.
        """
        ...

    def writePartitionFile(self, decision: bool) -> None:
        """
        Set whether to write the computed partition to the file specified with setPartitionFileName.

        :param decision: Whether to write the partition file.
        """
        ...

    def setPartitionFileName(self, file_name: str) -> None:
        """
        Set the filename of the computed partition.

        :param file_name: The filename to set.
        """
        ...

    def suppressOutput(self, decision: bool) -> None:
        """
        Suppress partitioning output.

        :param decision: Whether to suppress output.
        """
        ...

    def loadINIconfiguration(self, path: str) -> None:
        """
        Read KaHyPar configuration from file.

        :param path: The path to the configuration file.
        """
        ...

__version__: str