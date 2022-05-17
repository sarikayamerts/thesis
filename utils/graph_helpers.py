import typing

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def generate_adj_matrix(df, threshold):
    adjacency_matrix = pd.pivot_table(df[["rt_plant_id", "production"]],
                                      columns="rt_plant_id",
                                      index="forecast_dt").corr()
    adjacency_matrix = adjacency_matrix[adjacency_matrix > threshold].fillna(0)
    return adjacency_matrix

class GraphInfo:
    def __init__(self, edges, num_nodes, edge_weights):
        self.edges = edges
        self.num_nodes = num_nodes
        self.edge_weights = edge_weights

def create_graph(adjacency_matrix, weight=True):
    node_indices, neighbor_indices = np.where(adjacency_matrix.values > 0)
    edges = (node_indices.tolist(), neighbor_indices.tolist())
    if weight:
        edge_weights = adjacency_matrix.values[adjacency_matrix.values > 0]
    else:
        edge_weights = np.ones(len(edges[1]))
    graph = GraphInfo(edges, adjacency_matrix.shape[0], edge_weights=edge_weights)
    print(f"number of nodes: {graph.num_nodes}, number of edges: {len(graph.edges[0])}")
    return graph



class GraphConv(layers.Layer):
    def __init__(
        self,
        in_feat,
        out_feat,
        graph_info: GraphInfo,
        aggregation_type="mean",
        combination_type="concat",
        activation: typing.Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.graph_info = graph_info
        self.aggregation_type = aggregation_type
        self.combination_type = combination_type
        self.weight = tf.Variable(
            initial_value=keras.initializers.glorot_uniform()(
                shape=(in_feat, out_feat), dtype="float32"
            ),
            trainable=True,
        )
        self.activation = layers.Activation(activation)

    def aggregate(self, neighbour_representations: tf.Tensor):
        aggregation_func = {
            "sum": tf.math.unsorted_segment_sum,
            "mean": tf.math.unsorted_segment_mean,
            "max": tf.math.unsorted_segment_max,
        }.get(self.aggregation_type)

        if aggregation_func:
            return aggregation_func(
                neighbour_representations,
                self.graph_info.edges[0],
                num_segments=self.graph_info.num_nodes,
            )

        raise ValueError(f"Invalid aggregation type: {self.aggregation_type}")

    def compute_nodes_representation(self, features: tf.Tensor):
        # print("Computing nodes representation")
        # print(features.shape)
        # print(self.weight.shape)
        return tf.matmul(features, self.weight)

    def compute_aggregated_messages(self, features: tf.Tensor):
        neighbour_representations = tf.gather(features, self.graph_info.edges[1])
        # print(neighbour_representations.shape)
        neighbour_representations = tf.transpose(neighbour_representations, [3, 1, 2, 0])
        neighbour_representations *= tf.convert_to_tensor(np.array(self.graph_info.edge_weights).astype("float32"))
        neighbour_representations = tf.transpose(neighbour_representations, [3, 1, 2, 0])
        aggregated_messages = self.aggregate(neighbour_representations)
        # print("Computing aggregated messages")
        # print(aggregated_messages.shape)
        # print(self.weight.shape)
        return tf.matmul(aggregated_messages, self.weight)

    def update(self, nodes_representation: tf.Tensor, aggregated_messages: tf.Tensor):
        if self.combination_type == "concat":
            h = tf.concat([nodes_representation, aggregated_messages], axis=-1)
        elif self.combination_type == "add":
            h = nodes_representation + aggregated_messages
        else:
            raise ValueError(f"Invalid combination type: {self.combination_type}.")

        return self.activation(h)

    def call(self, features: tf.Tensor):
        """Forward pass.

        Args:
            features: tensor of shape `(num_nodes, batch_size, input_seq_len, in_feat)`

        Returns:
            A tensor of shape `(num_nodes, batch_size, input_seq_len, out_feat)`
        """
        nodes_representation = self.compute_nodes_representation(features)
        aggregated_messages = self.compute_aggregated_messages(features)
        return self.update(nodes_representation, aggregated_messages)

def calculate_wmape_tf(y_true, y_pred):
    total_abs_diff = tf.reduce_sum(tf.abs(tf.subtract(y_true, y_pred)))
    total = tf.reduce_sum(y_true)
    return tf.realdiv(total_abs_diff, total)

class LSTMGC(layers.Layer):
    """Layer comprising a convolution layer followed by LSTM and dense layers."""

    def __init__(
        self,
        in_feat,
        out_feat,
        lstm_units: int,
        input_seq_len: int,
        output_seq_len: int,
        graph_info: GraphInfo,
        graph_conv_params: typing.Optional[dict] = None,
        keras_params: typing.Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # graph conv layer
        if graph_conv_params is None:
            graph_conv_params = {
                "aggregation_type": "mean",
                "combination_type": "concat",
                "activation": None,
            }
        self.graph_conv = GraphConv(in_feat, out_feat, graph_info, **graph_conv_params)

        self.lstm = layers.LSTM(lstm_units, activation=keras_params.get("lstm_activation", "tanh"))
        self.dense = layers.Dense(output_seq_len, activation=keras_params.get("dense_activation", "sigmoid"))
        self.input_seq_len, self.output_seq_len = input_seq_len, output_seq_len

    def get_config(self):
        return super().get_config()


    def call(self, inputs):
        """Forward pass.

        Args:
            inputs: tf.Tensor of shape `(batch_size, input_seq_len, num_nodes, in_feat)`

        Returns:
            A tensor of shape `(batch_size, output_seq_len, num_nodes)`.
        """
        # convert shape to  (num_nodes, batch_size, input_seq_len, in_feat)
        inputs = tf.transpose(inputs, [2, 0, 1, 3])

        gcn_out = self.graph_conv(inputs)  # gcn_out has shape: (num_nodes, batch_size, input_seq_len, out_feat)
        shape = tf.shape(gcn_out)
        num_nodes, batch_size, input_seq_len, out_feat = shape[0], shape[1], shape[2], shape[3]

        # LSTM takes only 3D tensors as input
        gcn_out = tf.reshape(gcn_out, (num_nodes * batch_size, input_seq_len, out_feat))
        lstm_out = self.lstm(gcn_out)  # lstm_out has shape: (num_nodes * batch_size, lstm_units)

        dense_output = self.dense(lstm_out)  # dense_output has shape: (num_nodes * batch_size, output_seq_len)
        output = tf.reshape(dense_output, (num_nodes, batch_size, self.output_seq_len, -1))
        return tf.transpose(output, [1, 2, 0, 3])  # returns Tensor of shape (batch_size, output_seq_len, num_nodes)