class ActorNetwork(network.Network):
    def __init__(self, 
                 observation_spec,
                 action_spec,
                 embedding_dim,
                 hidden_dim,
                 items_num,
                 movie_embedding,
                 name):
        
        super(ActorNetwork, self).__init__(input_tensor_spec=observation_spec, state_spec=(), name=name)
        
        self.movie_embedding = movie_embedding
        self.items_num = items_num
        self._action_spec = action_spec
#         self._network_output_spec = None
        self.observation_spec = observation_spec
        
#         self.model = tf.keras.Sequential([
#             tf.keras.layers.InputLayer(name='input_layer', input_shape=(3*embedding_dim,)),
#             tf.keras.layers.Dense(hidden_dim, activation='relu'),
#             tf.keras.layers.Dense(hidden_dim, activation='relu'),
#             tf.keras.layers.Dense(embedding_dim, activation='tanh')
#         ])

        self.model = tf_agents.networks.Sequential([tf.keras.layers.InputLayer(name='input_layer', input_shape=(3*embedding_dim,)),
                                                    tf.keras.layers.Dense(hidden_dim, activation='relu'),
                                                    tf.keras.layers.Dense(hidden_dim, activation='relu'),
                                                    tf.keras.layers.Dense(embedding_dim, activation='tanh')
                                                   ])
                                                   
        
        
        self.recommended_items = []
    
    def create_variables(self, input_tensor_spec=None, **kwargs):
        """Force creation of the network's variables.
        Return output specs.
        Args:
          input_tensor_spec: (Optional).  Override or provide an input tensor spec
            when creating variables.
          **kwargs: Other arguments to `network.call()`, e.g. `training=True`.
        Returns:
          Output specs - a nested spec calculated from the outputs (excluding any
          batch dimensions).  If any of the output elements is a tfp `Distribution`,
          the associated spec entry returned is `None`.
        Raises:
          ValueError: If no `input_tensor_spec` is provided, and the network did
            not provide one during construction.
        """
        if self._network_output_spec is not None:
            if self._network_output_spec.shape != []:
                print(f"_network_output_spec : {self._network_output_spec}")
                return self._network_output_spec
    
        if self._input_tensor_spec is None:
            print(f"_input_tensor_spec : {self._input_tensor_spec}")
            self._input_tensor_spec = input_tensor_spec
        input_tensor_spec = self._input_tensor_spec
        
        if input_tensor_spec is None:
            raise ValueError(
              "Unable to create_variables: no input_tensor_spec provided, and "
              "Network did not define one.")

        random_input = tensor_spec.sample_spec_nest(
            input_tensor_spec, outer_dims=(1,))
        
        initial_state = self.get_initial_state(batch_size=1)
        step_type = tf.fill((1,), time_step.StepType.FIRST)
        outputs = self.__call__(
                    random_input,
                    step_type=step_type,
                    network_state=initial_state,
                    **kwargs)
        print(outputs)
        
#         def _calc_unbatched_spec(x):
#             if isinstance(x, tfp.distributions.Distribution):
#                 parameters = distribution_utils.get_parameters(x)
#                 parameter_specs = _convert_to_spec_and_remove_singleton_batch_dim(parameters, outer_ndim=1)

#                 return distribution_utils.DistributionSpecV2(event_shape=x.event_shape, dtype=x.dtype, parameters=parameter_specs)
#             else:
#                 return nest_utils.remove_singleton_batch_spec_dim(tf.type_spec_from_value(x), outer_ndim=1)

#         self._network_output_spec = tf.nest.map_structure(_calc_unbatched_spec, outputs[0])

        self._network_output_spec = tf.type_spec_from_value(outputs[0])
        return self._network_output_spec        
    
    
    def call(self, observations, step_type = (), network_state = ()):
        action_score, network_state = self.model(observations)
        action_score = tf.reshape(action_score, (1,100))
        
        items_ids = np.array(range(self.items_num))
        
#         items_ebs = self.embedding_network.get_layer('movie_embedding')(items_ids)
        items_ebs = self.movie_embedding
        action_score = tf.transpose(action_score, perm=(1,0))
        
        item_idx = np.argmax(tf.keras.backend.dot(items_ebs, action_score))
        
        action = int(items_ids[item_idx])
        self.recommended_items.append(action)
        
        return tf.nest.pack_sequence_as(self._action_spec, [tf.convert_to_tensor(np.array([action]))]), network_state 
#         return tf.convert_to_tensor(np.array([action])), network_state
