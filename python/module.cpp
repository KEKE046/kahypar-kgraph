
/*******************************************************************************
 * This file is part of KaHyPar.
 *
 * Copyright (C) 2019 Sebastian Schlag <sebastian.schlag@kit.edu>
 *
 * KaHyPar is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * KaHyPar is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with KaHyPar.  If not, see <http://www.gnu.org/licenses/>.
 *
 ******************************************************************************/

#include <pybind11/pybind11.h>

#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "kahypar-resources/definitions.h"
#include "kahypar/definitions.h"

#include "kahypar/partition/context.h"
#include "kahypar/io/hypergraph_io.h"
#include "kahypar/partitioner_facade.h"
#include "kahypar/application/command_line_options.h"
#include "kahypar/datastructure/connectivity_sets.h"
#include "kahypar/partition/metrics.h"

struct ContextWrapper {
  kahypar::Context * ctx;
};
void partition(kahypar::Hypergraph& hypergraph,
               ContextWrapper context) {
  kahypar::PartitionerFacade().partition(hypergraph, *context.ctx);
}

namespace py = pybind11;

namespace bind {
  using kahypar::Hypergraph;
  using kahypar::HypernodeID;
  using kahypar::HyperedgeID;
  using kahypar::HyperedgeIndexVector;
  using kahypar::HyperedgeVector;
  using kahypar::HyperedgeWeightVector;
  using kahypar::HypernodeWeightVector;
  using kahypar::PartitionID;

Hypergraph createHypergraphFromFile(const std::string& filename, const PartitionID num_parts) {
  return kahypar::io::createHypergraphFromFile(filename, num_parts, VALIDATE_INPUT, PROMOTE_WARNINGS_TO_ERRORS);
}

Hypergraph createWeightedHypergraph(const HypernodeID num_nodes, const HyperedgeID num_edges,
                                    const HyperedgeIndexVector& index_vector, const HyperedgeVector& edge_vector,
                                    const PartitionID k, const HyperedgeWeightVector& edge_weights,
                                    const HypernodeWeightVector& node_weights) {
  if (VALIDATE_INPUT) {
    std::vector<HyperedgeID> ignored_hes;
    std::vector<size_t> ignored_pins;
    kahypar::io::validateAndPrintErrors(num_nodes, num_edges, index_vector.data(), edge_vector.data(),
                                        edge_weights.empty() ? nullptr : edge_weights.data(),
                                        node_weights.empty() ? nullptr : node_weights.data(),
                                        {}, ignored_hes, ignored_pins, PROMOTE_WARNINGS_TO_ERRORS);
    return Hypergraph(num_nodes, num_edges, index_vector, edge_vector, k,
                      edge_weights, node_weights, ignored_hes, ignored_pins);
  }
  return Hypergraph(num_nodes, num_edges, index_vector, edge_vector, k, edge_weights, node_weights);
}

Hypergraph createUnweightedHypergraph(const HypernodeID num_nodes, const HyperedgeID num_edges,
                                      const HyperedgeIndexVector& index_vector, const HyperedgeVector& edge_vector,
                                      const PartitionID k) {
  return createWeightedHypergraph(num_nodes, num_edges, index_vector, edge_vector, k, {}, {});
}

template<typename T>
std::vector<T> toVector(const py::array_t<T, py::array::c_style | py::array::forcecast>& array) {
  if(array.ndim() != 1) {
    throw std::runtime_error("Array dimension must be 1");
  }
  std::vector<T> result;
  size_t size = array.shape()[0];
  result.reserve(size);
  for(size_t i = 0; i < size; i++) {
    result.push_back(array.at(i));
  }
  return result;
}

template<typename T>
py::array_t<T> toNumpy(const std::vector<T> & vec) {
  auto res = py::array_t<T>(vec.size());
  for(size_t i = 0, e = vec.size(); i < e; i++) {
    res.mutable_at(i) = vec[i];
  }
  return res;
}

Hypergraph createUnweightedHypergraphFromNumpy(
  const HypernodeID num_nodes, const HyperedgeID num_edges,
  const py::array_t<size_t, py::array::c_style | py::array::forcecast>& index_vector,
  const py::array_t<kahypar::HypernodeID, py::array::c_style | py::array::forcecast>& edge_vector,
  const PartitionID k
) {
  return createUnweightedHypergraph(
    num_nodes, num_edges,
    toVector(index_vector),
    toVector(edge_vector),
    k
  );
}

Hypergraph createWeightedHypergraphFromNumpy(
  const HypernodeID num_nodes, const HyperedgeID num_edges,
  const py::array_t<size_t, py::array::c_style | py::array::forcecast>& index_vector,
  const py::array_t<kahypar::HypernodeID, py::array::c_style | py::array::forcecast>& edge_vector,
  const PartitionID k,
  const py::array_t<kahypar::HyperedgeWeight, py::array::c_style | py::array::forcecast>& edge_weights,
  const py::array_t<kahypar::HypernodeWeight, py::array::c_style | py::array::forcecast>& node_weights
) {
  return createWeightedHypergraph(
    num_nodes, num_edges,
    toVector(index_vector),
    toVector(edge_vector),
    k,
    toVector(edge_weights),
    toVector(node_weights)
  );
}

std::string loadConfigFile(std::string name) {
    py::module importlib_resources = py::module::import("importlib.resources");
    auto res = importlib_resources.attr("files")("kahypar_kgraph")
      .attr("joinpath")("config")
      .attr("joinpath")(name)
      .attr("read_text")();
    return py::str(res);
}

void loadContextPreset(ContextWrapper &c, std::string name) {
  auto content = bind::loadConfigFile(name + ".ini");
  std::stringstream ss;
  ss << content;
  kahypar::parseIniContent(*c.ctx, ss);
}

ContextWrapper createContext(PartitionID k, double epsilon, std::optional<std::string> preset, std::optional<std::string> ini_file, std::optional<std::string> ini_content, bool verbose) {
  if(!preset && !ini_file && !ini_content) {
    throw std::runtime_error("Need `preset` or `ini_file` or `ini_content`");
  }
  auto * res = new kahypar::Context();
  if(preset) {
    auto content = bind::loadConfigFile(*preset + ".ini");
    std::stringstream ss;
    ss << content;
    kahypar::parseIniContent(*res, ss);
  }
  if(ini_file) {
    parseIniToContext(*res, *ini_file);
  }
  if(ini_content) {
    std::stringstream ss;
    ss << *ini_content;
    kahypar::parseIniContent(*res, ss);
  }
  res->partition.k = k;
  res->partition.epsilon = epsilon;
  if(verbose) {
    res->partition.quiet_mode = false;
  } else {
    res->partition.quiet_mode = true;
  }
  return ContextWrapper{res};
}

} // namespace bind


namespace py = pybind11;

PYBIND11_MODULE(kahypar_kgraph, m) {
  using kahypar::Hypergraph;
  using kahypar::HypernodeID;
  using kahypar::HyperedgeID;
  using kahypar::HyperedgeIndexVector;
  using kahypar::HyperedgeVector;
  using kahypar::HyperedgeWeightVector;
  using kahypar::HypernodeWeightVector;
  using kahypar::PartitionID;
  using ConnectivitySet = typename kahypar::ds::ConnectivitySets<PartitionID, HyperedgeID>::ConnectivitySet;

  py::class_<Hypergraph>(
      m, "Hypergraph")
      .def(py::init(&bind::createUnweightedHypergraphFromNumpy),R"pbdoc(
Construct an unweighted hypergraph.

:param HypernodeID num_nodes: Number of nodes
:param HyperedgeID num_edges: Number of hyperedges
:param HyperedgeIndexVector index_vector: Starting indices for each hyperedge
:param HyperedgeVector edge_vector: Vector containing all hyperedges
:param PartitionID k: Number of blocks in which the hypergraph should be partitioned

          )pbdoc",
           py::arg("num_nodes"),
           py::arg("num_edges"),
           py::arg("index_vector"),
           py::arg("edge_vector"),
           py::arg("k"))
      .def(py::init(&bind::createWeightedHypergraphFromNumpy),R"pbdoc(
Construct a hypergraph with node and edge weights.

If only one type of weights is required, the other argument has to be an empty list.

:param HypernodeID num_nodes: Number of nodes
:param HyperedgeID num_edges: Number of hyperedges
:param HyperedgeIndexVector index_vector: Starting indices for each hyperedge
:param HyperedgeVector edge_vector: Vector containing all hyperedges
:param PartitionID k: Number of blocks in which the hypergraph should be partitioned
:param HyperedgeWeightVector edge_weights: Weights of all hyperedges
:param HypernodeWeightVector node_weights: Weights of all hypernodes

          )pbdoc",
           py::arg("num_nodes"),
           py::arg("num_edges"),
           py::arg("index_vector"),
           py::arg("edge_vector"),
           py::arg("k"),
           py::arg("edge_weights"),
           py::arg("node_weights"))
      .def("printGraphState", &Hypergraph::printGraphState,
           "Print the hypergraph state (for debugging purposes)")
      .def("nodeDegree", &Hypergraph::nodeDegree,
           "Get the degree of the node",
           py::arg("node"))
      .def("edgeSize", &Hypergraph::edgeSize,
           "Get the size of the hyperedge",
           py::arg("hyperedge"))
      .def("nodeWeight", &Hypergraph::nodeWeight,
           "Get the weight of the node",
           py::arg("node"))
      .def("edgeWeight", &Hypergraph::edgeWeight,
           "Get the weight of the hyperedge",
           py::arg("hyperedge"))
      .def("blockID", &Hypergraph::partID,
           "Get the block of the node in the current hypergraph partition (before partitioning: -1)",
           py::arg("node"))
      .def("numNodes", &Hypergraph::initialNumNodes,
           "Get the number of nodes")
      .def("numEdges", &Hypergraph::initialNumEdges,
           "Get the number of hyperedges")
      .def("numPins", &Hypergraph::initialNumPins,
           "Get the number of pins")
      .def("numBlocks", &Hypergraph::k,
           "Get the number of blocks")
      .def("numPinsInBlock", &Hypergraph::pinCountInPart,
           "Get the number of pins of the hyperedge that are assigned to corresponding block",
           py::arg("hyperedge"),
           py::arg("block"))
      .def("connectivity", &Hypergraph::connectivity,
           "Get the connecivity of the hyperedge (i.e., the number of blocks which contain at least one pin)",
           py::arg("hyperedge"))
      .def("connectivitySet", &Hypergraph::connectivitySet, py::return_value_policy::reference_internal,
           "Get the connectivity set of the hyperedge",
           py::arg("hyperedge"))
      .def("communities", &Hypergraph::communities,
           "Get the community structure")
      .def("blockWeight", &Hypergraph::partWeight,
      "Get the weight of the block",
           py::arg("block"))
      .def("blockSize", &Hypergraph::partSize,
           "Get the number of vertices in the block",
           py::arg("block"))
      .def("reset", &Hypergraph::reset,
           "Reset the hypergraph to its initial state")
      .def("fixNodeToBlock", &Hypergraph::setFixedVertex,
        "Fix node to the cooresponding block",
        py::arg("node"), py::arg("block"))
      .def("setFixedBlockIds", [](Hypergraph &h, py::array_t<PartitionID, py::array::c_style | py::array::forcecast> _ids) {
        auto ids = bind::toVector(_ids);
        for(size_t i = 0, e = ids.size(); i < e; i++) {
          if(ids[i] != -1) h.setFixedVertex(i, ids[i]);
        }
      })
      .def("numFixedNodes", &Hypergraph::numFixedVertices,
        "Get the number of fixed nodes in the hypergraph")
      .def("containsFixedNodex", &Hypergraph::containsFixedVertices,
        "Return true if the hypergraph contains nodes fixed to a specific block")
      .def("isFixedNode", &Hypergraph::isFixedVertex,
        "Return true if the node is fixed to a block",
        py::arg("node"))
      // .def("nodes", [](Hypergraph &h) {
      //     return py::make_iterator(h.nodes().first,h.nodes().second);}, py::keep_alive<0, 1>(),
      //   "Iterate over all nodes")
      // .def("edges", [](Hypergraph &h) {
      //     return py::make_iterator(h.edges().first,h.edges().second);}, py::keep_alive<0, 1>(),
      //   "Iterate over all hyperedges")
      .def("nodes", [](Hypergraph &h) {
        auto [begin, end] = h.nodes();
        std::vector res(begin, end);
        return bind::toNumpy(res);
      }, "Array of node ids")
      .def("edges", [](Hypergraph &h) {
        auto [begin, end] = h.edges();
        std::vector res(begin, end);
        return bind::toNumpy(res);
      }, "Array of edge ids")
      .def("partIds", [](Hypergraph & h) {
        auto [begin, end] = h.nodes();
        std::vector<HypernodeID> res;
        for(auto i = begin; i != end; i++) {
          res.push_back(h.partID(*i));
        }
        return bind::toNumpy(res);
      }, "Array of part ids")
      .def("pins", [](Hypergraph &h, HyperedgeID he) {
          return py::make_iterator(h.pins(he).first,h.pins(he).second);}, py::keep_alive<0, 1>(),
        "Iterate over all pins of the hyperedge",
        py::arg("hyperedge"))
      .def("incidentEdges", [](Hypergraph &h, HypernodeID hn) {
          return py::make_iterator(h.incidentEdges(hn).first,h.incidentEdges(hn).second);}, py::keep_alive<0, 1>(),
        "Iterate over all incident hyperedges of the node",
        py::arg("node"))
      .def("__repr__", [](Hypergraph &c){
        std::stringstream ss;
        ss << "Hypergraph(n_nodes=" << c.initialNumNodes() << ", n_edges=" << c.initialNumEdges() << ", n_fixed=" << c.numFixedVertices() << ", k=" << c.k() <<")";
        return ss.str();
      })
      .def("__str__", [](Hypergraph &c){
        std::stringstream ss;
        ss << "Hypergraph(n_nodes=" << c.initialNumNodes() << ", n_edges=" << c.initialNumEdges() << ", n_fixed=" << c.numFixedVertices() << ", k=" << c.k() <<")";
        return ss.str();
      });


  py::class_<ConnectivitySet>(m,
                              "Connectivity Set")
      .def("contains",&ConnectivitySet::contains,
           "Check if the block is contained in the connectivity set of the hyperedge",
           py::arg("block"))
      .def("__iter__",
           [](const ConnectivitySet& con) {
             return py::make_iterator(con.begin(),con.end());
           },py::keep_alive<0, 1>(),
           "Iterate over all blocks contained in the connectivity set of the hyperedge");

  m.def(
      "createHypergraphFromFile", &bind::createHypergraphFromFile,
      "Construct a hypergraph from a file in hMETIS format",
      py::arg("filename"), py::arg("k"));


  m.def(
      "partition", &partition,
      "Compute a k-way partition of the hypergraph",
      py::arg("hypergraph"), py::arg("context"));

  m.def(
      "cut", &kahypar::metrics::hyperedgeCut,
      "Compute the cut-net metric for the partitioned hypergraph",
      py::arg("hypergraph"));

  m.def(
      "soed", &kahypar::metrics::soed,
      "Compute the sum-of-extrnal-degrees metric for the partitioned hypergraph",
      py::arg("hypergraph"));

  m.def(
      "connectivityMinusOne", &kahypar::metrics::km1,
      "Compute the connecivity metric for the partitioned hypergraph",
      py::arg("hypergraph"));

  m.def(
      "imbalance", &kahypar::metrics::imbalance,
      "Compute the imbalance of the hypergraph partition",
    py::arg("hypergraph"),py::arg("context"));


  py::class_<ContextWrapper>(m, "Context")
      .def(py::init(&bind::createContext), R"pbdoc(
Construct a KaHyPar Context.

:param str | None preset: KaHyPar presets
:param str | None ini_file: ini file to load
:param str | None ini_content: ini content to load

          )pbdoc",
        py::arg("k"),
        py::arg("epsilon"),
        py::arg("preset")=std::nullopt,
        py::arg("ini_file")=std::nullopt,
        py::arg("ini_content")=std::nullopt,
        py::arg("verbose")=false
      )
      .def_property("k",
        [](ContextWrapper &c) {return c.ctx->partition.k;},
        [](ContextWrapper &c, PartitionID k) {c.ctx->partition.k = k;}
      )
      .def_property("epsilon",
        [](ContextWrapper &c) {return c.ctx->partition.epsilon;},
        [](ContextWrapper &c, double epsilon) {c.ctx->partition.epsilon = epsilon;}
      )
      .def_property("verbose",
        [](ContextWrapper &c) {return !c.ctx->partition.quiet_mode;},
        [](ContextWrapper &c, bool verbose) {c.ctx->partition.quiet_mode = !verbose;}
      )
      .def("__del__", [](ContextWrapper &c) {
        delete c.ctx;
      })
      .def("__repr__", [](ContextWrapper &c){
        std::stringstream ss;
        ss << *c.ctx;
        return ss.str();
      })
      .def("__str__", [](ContextWrapper &c){
        std::stringstream ss;
        ss << *c.ctx;
        return ss.str();
      })
      // .def("debug", [](Context & c) {
      //   std::cout << "debug _parent=" << c.stats._parent << std::endl;
      // })
      .def("setK",[](ContextWrapper &c, const PartitionID k) {
          c.ctx->partition.k = k;
        },
        "Number of blocks the hypergraph should be partitioned into",
        py::arg("k"))
      .def("setEpsilon",[](ContextWrapper &c, const double eps) {
          c.ctx->partition.epsilon = eps;
        },
        "Allowed imbalance epsilon",
        py::arg("imbalance parameter epsilon"))
      .def("setCustomTargetBlockWeights",
        [](ContextWrapper c, const std::vector<kahypar::HypernodeWeight>& custom_target_weights) {
          c.ctx->partition.use_individual_part_weights = true;
          c.ctx->partition.max_part_weights.clear();
          for ( size_t block = 0; block < custom_target_weights.size(); ++block ) {
            c.ctx->partition.max_part_weights.push_back(custom_target_weights[block]);
          }
        },
        "Assigns each block of the partition an individual maximum allowed block weight",
        py::arg("custom target block weights"))
      .def("setSeed",[](ContextWrapper &c, const int seed) {
          c.ctx->partition.seed = seed;
        },
        "Seed for the random number generator",
        py::arg("seed"))
       .def("writePartitionFile",[](ContextWrapper &c, const bool decision) {
          c.ctx->partition.write_partition_file = decision;
        },
        "Write the computed partition to the file specified with setPartitionFileName",
        py::arg("bool"))
      .def("setPartitionFileName",[](ContextWrapper &c, const std::string& file_name) {
          c.ctx->partition.graph_partition_filename = file_name;
        },
        "Set the filename of the computed partition",
        py::arg("file_name"))
      .def("suppressOutput",[](ContextWrapper &c, const bool decision) {
          c.ctx->partition.quiet_mode = decision;
        },
        "Suppress partitioning output",
        py::arg("bool"))
      .def("loadPreset", &bind::loadContextPreset, "Read KaHyPar configuration from preset", py::arg("preset"))
      .def("loadINTContent", [](ContextWrapper &c, const std::string& content) {
        std::stringstream ss;
        ss << content;
        kahypar::parseIniContent(*c.ctx, ss);
      }, "Read KaHyPar configuration from ini string", py::arg("ini-content"))
      .def("loadINIconfiguration",
           [](ContextWrapper c, const std::string& path) {
             parseIniToContext(*c.ctx, path);
           },
           "Read KaHyPar configuration from file",
           py::arg("path-to-file")
      );

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
