"""
Copyright 2025 Universitat Politècnica de Catalunya

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import tensorflow as tf


class RouteNetGauss(tf.keras.Model):
    z_scores_fields = {
        "flow_traffic",
        "flow_packets",
    }

    def __init__(
        self,
        z_scores: dict,
        mask_field: str,
        iterations: int = 8,
        flow_state_dim: int = 32,
        link_state_dim: int = 32,
        queue_state_dim: int = 32,
        node_state_dim: int = 32,
        output_dim: int = 1,
        inference_mode: bool = False,
        use_trans_delay: bool = False,
    ):
        """RouteNet-Gauss model

        Parameters
        ----------
        z_scores : dict
            Z-scores of normalized features. Use self.z_scores_fields to check which
            features are needed.
        mask_field : str,
            Field from the input data that is used to mask out windows without packets
            in a given window. Usually has the form of "flow_has_X", where X is the
            perfomance metric to predict (delay requires at least one packet, jitter
            two).
        iterations : int, optional
            Number of iterations in the Message Passing, by default 8.
        flow_state_dim : int, optional
            Dimension of flow embeddings, by default 32.
        link_state_dim : int, optional
            Dimension of link embeddings, by default 32.
        queue_state_dim : int, optional
            Dimension of queue embeddings, by default 32.
        node_state_dim : int, optional
            Dimension of node embeddings, by default 32.
        output_dim : int, optional
            Number of outputs, by default 1.
        inference_mode : bool, optional
            If true, predictions by the model will be forced to be positive. Doing so
            during training results in poorer learning. By default False.
        use_trans_delay : bool, optional
            If true, transmission delay will be obtained from the measured inputs.
            Useful when aiming to predict the delay, as it allows the model to focus
            only on the queueing delay. By default False
        """
        super().__init__()

        self.max_buffer_types = 3

        self.iterations = iterations
        self.flow_state_dim = flow_state_dim
        self.link_state_dim = link_state_dim
        self.queue_state_dim = queue_state_dim
        self.node_state_dim = node_state_dim
        self.output_dim = output_dim

        assert mask_field is not None, "mask_field must be specified"
        self.mask_field = mask_field

        self.z_scores = z_scores
        assert (
            type(z_scores) == dict
            and all(kk in self.z_scores for kk in self.z_scores_fields)
            and all(len(val) == 2 for val in self.z_scores.values())
        ), "overriden z_score dict is not valid!"
        # Calculate the tranmission delay separately (should be used only when
        # predicting the delay)
        self.use_trans_delay = use_trans_delay
        # Force the queuing predictions to be positive
        self.inference_mode = inference_mode

        # GRU Cells used in the Message Passing step
        self.flow_update = tf.keras.layers.GRUCell(
            self.flow_state_dim, name="PathUpdate"
        )
        self.link_update = tf.keras.layers.GRUCell(
            self.link_state_dim, name="LinkUpdate"
        )
        self.queue_update = tf.keras.layers.GRUCell(
            self.queue_state_dim, name="QueueUpdate"
        )
        self.node_update = tf.keras.layers.GRUCell(
            self.node_state_dim, name="NodeUpdate"
        )

        # Embedding functions
        self.flow_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(None, 3)),
                tf.keras.layers.Dense(
                    self.flow_state_dim, activation=tf.keras.activations.relu
                ),
                tf.keras.layers.Dense(
                    self.flow_state_dim, activation=tf.keras.activations.relu
                ),
            ],
            name="PathEmbedding",
        )
        self.queue_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=self.max_buffer_types),
                tf.keras.layers.Dense(
                    self.queue_state_dim, activation=tf.keras.activations.relu
                ),
                tf.keras.layers.Dense(
                    self.queue_state_dim, activation=tf.keras.activations.relu
                ),
            ],
            name="QueueEmbedding",
        )
        self.link_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(None, 1)),
                tf.keras.layers.Dense(
                    self.link_state_dim, activation=tf.keras.activations.relu
                ),
                tf.keras.layers.Dense(
                    self.link_state_dim, activation=tf.keras.activations.relu
                ),
            ],
            name="LinkEmbedding",
        )
        self.node_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=self.queue_state_dim),
                tf.keras.layers.Dense(
                    (self.queue_state_dim + self.node_state_dim) // 2,
                    activation=tf.keras.activations.relu,
                ),
                tf.keras.layers.Dense(
                    self.node_state_dim, activation=tf.keras.activations.relu
                ),
            ]
        )

        self.readout_path = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(None, self.flow_state_dim)),
                tf.keras.layers.Dense(
                    int(self.link_state_dim / 2), activation=tf.keras.activations.relu
                ),
                tf.keras.layers.Dense(
                    int(self.flow_state_dim / 2), activation=tf.keras.activations.relu
                ),
                tf.keras.layers.Dense(self.output_dim),
            ],
            name="PathReadout",
        )

    @tf.function
    def call(self, inputs):
        # Initialize result matrix
        total_delay = tf.zeros((0, self.output_dim))

        seg_num = inputs["seg_num"]
        flow_to_link = flow_to_queue = inputs["path_to_link"]
        node_groupings = inputs["node_groupings"]
        inverse_node_groupings = inputs["node_groupings_inversed"]
        queue_to_link = inputs["queue_to_link"]
        link_to_path = queue_to_path = inputs["link_to_path"]

        # Initial embeddings
        traffic = inputs["flow_traffic"]
        pkt_rate = inputs["flow_packets"]
        pkt_size = inputs["flow_packet_size"]
        length = tf.squeeze(inputs["flow_length"], 1)
        flow_has_traffic = inputs["flow_has_traffic"]
        # We apply the transpose so the first dimension are the segments, the second the
        # flows
        initial_flow_state = tf.transpose(
            self.flow_embedding(
                tf.concat(
                    [
                        (traffic - self.z_scores["flow_traffic"][0])
                        / self.z_scores["flow_traffic"][1],
                        (pkt_rate - self.z_scores["flow_packets"][0])
                        / self.z_scores["flow_packets"][1],
                        tf.expand_dims(tf.cast(flow_has_traffic, tf.float32), 2),
                    ],
                    axis=2,
                ),
            ),
            perm=[1, 0, 2],
        )

        # Calculate load per link per window, including packet size correction due to
        # l1 and l2 headers size
        if "link_capacity" not in inputs:
            capacity = (
                tf.concat(
                    [inputs["link_r_capacity"], inputs["link_s_capacity"]], axis=0
                )
                * 1e9
            )
        else:
            capacity = inputs["link_capacity"] * 1e9
        expanded_capacity = tf.tile(tf.expand_dims(capacity, 1), [1, seg_num, 1])
        if "link_pkt_header_size" not in inputs:
            pkt_size_correction = tf.concat(
                [inputs["link_r_pkt_header_size"], inputs["link_s_pkt_header_size"]],
                axis=0,
            )
        else:
            pkt_size_correction = inputs["link_pkt_header_size"]
        pkt_size_correction = tf.tile(
            tf.expand_dims(pkt_size_correction, 1), [1, seg_num, 1]
        )
        flow_gather_traffic = tf.gather(traffic, flow_to_link[:, :, 0])
        flow_traffic = tf.math.reduce_sum(flow_gather_traffic, axis=1)
        flow_gather_pkt_rate = tf.gather(pkt_rate, flow_to_link[:, :, 0])
        flow_pkt_rate = tf.math.reduce_sum(flow_gather_pkt_rate, axis=1)
        load = (flow_traffic + flow_pkt_rate * pkt_size_correction) / expanded_capacity
        # We apply the transpose so the first dimension are the segments, the second the
        # links
        initial_link_state = tf.transpose(self.link_embedding(load), [1, 0, 2])

        # Queue_state and node states are related to memory buffers, these are the
        # states that are kept between windows
        buffer_type = inputs["buffer_type"]
        queue_state = self.queue_embedding(
            tf.squeeze(tf.one_hot(buffer_type, self.max_buffer_types), 1)
        )
        node_state = self.node_embedding(
            tf.math.reduce_sum(
                tf.gather(queue_state, node_groupings),
                axis=1,
                name="RQueueGrouping-Embedding",
            ),
        )

        # Variables for tf.autograd
        flow_state_sequence = tf.RaggedTensor.from_row_lengths(
            tf.zeros((tf.reduce_sum(length), self.flow_state_dim)), length
        ).with_row_splits_dtype(tf.int64)

        for curr_seg in range(inputs["seg_num"]):
            tf.autograph.experimental.set_loop_options(
                shape_invariants=[
                    (total_delay, tf.TensorShape([None, self.output_dim])),
                    (
                        flow_state_sequence,
                        tf.TensorShape([None, None, self.flow_state_dim]),
                    ),
                ],
            )

            # Initialize segment states for flows and links
            flow_state = initial_flow_state[curr_seg]
            link_state = initial_link_state[curr_seg]

            # Iterate t times doing the message passing
            for it in range(self.iterations):
                ###################
                #  LINK AND QUEUE #
                #     TO PATH     #
                ###################
                queue_gather = tf.gather(queue_state, queue_to_path)
                link_gather = tf.gather(link_state, link_to_path, name="LinkToPath")
                flow_update_rnn = tf.keras.layers.RNN(
                    self.flow_update, return_sequences=True, return_state=True
                )
                previous_flow_state = flow_state

                # flow_state -> state of path after processing sequence
                # flow_state_sequence -> sequence of intermediate states of the path
                # when elements within the sequence are processed
                flow_state_sequence, flow_state = flow_update_rnn(
                    tf.concat([queue_gather, link_gather], axis=2),
                    initial_state=flow_state,
                )
                # We select the element in flow_state_sequence so that it corresponds to
                # the state before the link was considered
                flow_state_sequence = tf.concat(
                    [tf.expand_dims(previous_flow_state, 1), flow_state_sequence],
                    axis=1,
                )

                ###################
                #  PATH AND NODE  #
                #    TO QUEUE     #
                ###################
                flow_gather = tf.gather_nd(flow_state_sequence, flow_to_queue)
                flow_sum = tf.math.reduce_sum(flow_gather, axis=1)
                node_gather = tf.gather(
                    node_state, inverse_node_groupings, name="NodeStateUnfolded"
                )

                queue_state, _ = self.queue_update(
                    tf.concat([flow_sum, node_gather], axis=1), [queue_state]
                )

                ###################
                #  QUEUE TO LINK  #
                ###################
                queue_gather = tf.gather(queue_state, queue_to_link)

                link_gru_rnn = tf.keras.layers.RNN(
                    self.link_update, return_sequences=False
                )
                link_state = link_gru_rnn(queue_gather, initial_state=link_state)

                ###################
                #  QUEUE TO NODE  #
                ###################
                node_state, _ = self.node_update(
                    tf.math.reduce_sum(
                        tf.gather(queue_state, node_groupings),
                        axis=1,
                        name="NodeQueueGrouping",
                    ),
                    states=node_state,
                )

            ###################
            # MESSAGE PASSING #
            #       END       #
            ###################
            # Readout and delay prediction
            capacity_gather = tf.gather(capacity, link_to_path)
            input_tensor = flow_state_sequence[:, 1:].to_tensor()

            occupancy_gather = self.readout_path(input_tensor)
            length = tf.ensure_shape(length, [None])
            occupancy_gather = tf.RaggedTensor.from_tensor(
                occupancy_gather, lengths=length
            )

            queue_delay = tf.math.reduce_sum(occupancy_gather / capacity_gather, axis=1)
            if self.use_trans_delay:
                trans_delay = pkt_size * tf.math.reduce_sum(1 / capacity_gather, axis=1)
                delay = queue_delay + trans_delay
            else:
                delay = queue_delay
            if self.inference_mode:
                delay = tf.keras.activations.relu(delay)

            delay = tf.boolean_mask(delay, inputs[self.mask_field][:, curr_seg])

            total_delay = tf.concat([total_delay, delay], axis=0)

        ##################
        #     WINDOW     #
        # PROCESSING END #
        ##################

        return total_delay
