import graph_nets
# import ml_collections
import sonnet as snt
import tensorflow as tf

# from . import modules
# from .. import enums, types
import graph_nets.graphs as types


class NodesAggregator(snt.Module):
    """Aggregates neighboring nodes based on sent and received nodes."""

    def __init__(self,
                 reducer=tf.math.unsorted_segment_sum,
                 name='nodes_aggregator'):
        super(NodesAggregator, self).__init__(name=name)
        self.reducer = reducer

    def __call__(self, graph: types.GraphsTuple) -> tf.Tensor:
        num_nodes = tf.reduce_sum(graph.n_node)
        adjacent_nodes = tf.gather(graph.nodes, graph.senders)
        return self.reducer(adjacent_nodes, graph.receivers, num_nodes)


class NodeLayer(graph_nets.blocks.NodeBlock):
    """GNN layer that only updates nodes, but uses edges."""

    def __init__(self, *args, **kwargs):
        super(NodeLayer, self).__init__(*args, use_globals=False, **kwargs)


class GCNLayer(graph_nets.blocks.NodeBlock):
    """GNN layer that only updates nodes using neighboring nodes and edges."""

    def __init__(self, *args, **kwargs):
        super(GCNLayer, self).__init__(*args, use_globals=False, **kwargs)
        self.gather_nodes = NodesAggregator()

    def __call__(self, graph: types.GraphsTuple) -> types.GraphsTuple:
        """Collect nodes, adjacent nodes, edges and update to get new nodes."""
        nodes_to_collect = []
        if self._use_sent_edges:
            nodes_to_collect.append(self._sent_edges_aggregator(graph))
        if self._use_received_edges:
            nodes_to_collect.append(self._received_edges_aggregator(graph))
        if self._use_nodes:
            nodes_to_collect.append(graph.nodes)

        nodes_to_collect.append(self.gather_nodes(graph))

        if self._use_globals:
            nodes_to_collect.append(graph_nets.blocks.broadcast_globals_to_nodes(graph))

        collected_nodes = tf.concat(nodes_to_collect, axis=-1)
        updated_nodes = self._node_model(collected_nodes)
        return graph.replace(nodes=updated_nodes)


class NodeEdgeLayer(snt.Module):
    """GNN layer that only updates nodes and edges."""

    def __init__(self, node_model_fn, edge_model_fn, name='NodeEdgeLayer'):
        super(NodeEdgeLayer, self).__init__(name=name)
        self.edge_block = graph_nets.blocks.EdgeBlock(
            edge_model_fn=edge_model_fn, use_globals=False)
        self.node_block = graph_nets.blocks.NodeBlock(
            node_model_fn=node_model_fn, use_globals=False)

    def __call__(self, graph: types.GraphsTuple) -> types.GraphsTuple:
        return self.node_block(self.edge_block(graph))

def cast_activation(act):
    """Map string to activation, or just pass the activation function."""
    activations = {
        'relu': tf.nn.relu,
        'tanh': tf.nn.tanh,
        'sigmoid': tf.nn.sigmoid,
        'softmax': tf.nn.softmax,
        'identity': tf.identity
    }
    if callable(act):
        return act
    else:
        return activations[act]

def get_mlp_fn(
        layer_sizes,
        act = 'relu'):
    """Instantiates a new MLP, followed by LayerNorm."""

    def make_mlp():
        return snt.Sequential([
            snt.nets.MLP(
                layer_sizes, activate_final=True, activation=cast_activation(act)),
            snt.LayerNorm(axis=-1, create_offset=True, create_scale=True)
        ])
    return make_mlp

def get_graph_block(block_type, node_size: int,
                    edge_size: int, global_size: int, index: int):
    """Gets a GNN block based on enum and sizes."""
    name = f'{block_type}_{index + 1}'
    if block_type == 'gcn':
        return GCNLayer(get_mlp_fn([node_size] * 2), name=name)
    elif block_type == 'mpnn':
        return NodeEdgeLayer(
            get_mlp_fn([node_size] * 2),
            get_mlp_fn([edge_size] * 2),
            name=name)
    elif block_type == 'graphnet':
        use_globals = index != 0
        return graph_nets.modules.GraphNetwork(
            node_model_fn=get_mlp_fn([node_size] * 2),
            edge_model_fn=get_mlp_fn([edge_size] * 2),
            global_model_fn=get_mlp_fn([global_size] * 2),
            edge_block_opt={'use_globals': use_globals},
            node_block_opt={'use_globals': use_globals},
            global_block_opt={'use_globals': use_globals},
            name=name)
    else:
        raise ValueError(f'block_type={block_type} not implemented')


class GNNEmbedder(snt.Module):
    """A general graph neural network for graph property prediction."""

    def __init__(self,
                 node_size: int,
                 edge_size: int,
                 global_size: int,
                 block_type: str,
                 n_layers: int = 3, **kwargs):
        super(GNNEmbedder, self).__init__(name=block_type)

        # Graph encoding step, basic linear mapping.
        self.encode = graph_nets.modules.GraphIndependent(
            node_model_fn=lambda: snt.Linear(node_size),
            edge_model_fn=lambda: snt.Linear(edge_size))
        # Message passing steps or GNN blocks.
        gnn_layers = [
            get_graph_block(
                block_type,
                node_size,
                edge_size,
                global_size,
                index)
            for index in range(0, n_layers)
        ]
        self.gnn = snt.Sequential(gnn_layers)
        # self.pred_layer = snt.Linear(output_dim) # modules.get_pred_layer(output_dim, output_act)

    def embed(self, x: types.GraphsTuple) -> tf.Tensor:
        return self.gnn(self.encode(x)).globals

    def __call__(self, x: types.GraphsTuple) -> tf.Tensor:
        return self.embed(x)

    @classmethod
    def from_hparams(cls, hp):
        return cls(**hp)

class GNNPredictor(snt.Module):
    def __init__(
        self,
        donor_gnn,
        acceptor_gnn,
        output_dim,
        name='predictor'
    ):
        super(GNNPredictor, self).__init__(name=name)
        self.donor_gnn = donor_gnn
        self.acceptor_gnn = acceptor_gnn
        self.pred_layer = snt.Linear(output_dim)
    
    def __call__(self, x_donor, x_acceptor):
        # x_donor, x_acceptor = inputs
        embed_donor = self.donor_gnn(x_donor)
        embed_acceptor = self.acceptor_gnn(x_acceptor)
        output = self.pred_layer(tf.concat([embed_donor, embed_acceptor], axis=-1))
        return output

    def embed_donor(self, x):
        return self.donor_gnn(x)

    def embed_acceptor(self, x):
        return self.acceptor_gnn(x)


def default_hp(output_dim: int):
    # hp = ml_collections.ConfigDict()
    hp = {}
    hp['node_size'] = 50
    hp['edge_size'] = 20
    hp['global_size'] = 150
    hp['block_type'] = 'graphnet'
    hp['n_layers'] = 3
    # hp.task = str(enums.TaskType(task))
    # hp['output_act'] = tf.relu # modules.task_to_activation_str(hp['task)
    hp['output_dim'] = output_dim
    # hp.lr = 1e-3
    # hp.epochs = 2000
    # hp.batch_size = 256
    # hp.patience = 200
    return hp
