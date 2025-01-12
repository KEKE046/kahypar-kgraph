import kahypar_kgraph as kahypar
import numpy as np
ctx = kahypar.Context(2, 0.01, 'cut_kKaHyPar_sea20')
g = kahypar.createHypergraphFromFile('tests/end_to_end/test_instances/bundle1.mtx.hgr', 2)
# idx_vector = np.array([0, 2, 4, 6])
# edge_vector = np.array([0, 1, 2, 3, 4, 3])
# g = kahypar.Hypergraph(5, 3, idx_vector, edge_vector, 2)

kahypar.partition(g, ctx)
print(g.partIds())