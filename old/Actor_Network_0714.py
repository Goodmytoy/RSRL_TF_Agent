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
        self.observation_spec = observation_spec
        
        self.model = tf_agents.networks.Sequential([tf.keras.layers.InputLayer(name='input_layer', input_shape=(3*embedding_dim,)),
                                                    tf.keras.layers.Dense(hidden_dim, activation='relu'),
                                                    tf.keras.layers.Dense(hidden_dim, activation='relu'),
                                                    tf.keras.layers.Dense(embedding_dim, activation='tanh')
                                                   ])
                                                   
        
        
        self.recommended_items = [] 
    
    
    def call(self, observations, step_type = (), network_state = ()):
        action_score, network_state = self.model(observations)
        action_score = tf.reshape(action_score, (1,100))
        
        items_ids = np.array(range(self.items_num))
        
        items_ebs = self.movie_embedding
        action_score = tf.transpose(action_score, perm=(1,0))
        
        item_idx = np.argmax(tf.keras.backend.dot(items_ebs, action_score))
        
        action = int(items_ids[item_idx])
        self.recommended_items.append(action)
        
        return tf.nest.pack_sequence_as(self._action_spec, [tf.convert_to_tensor(np.array([action]))]), network_state 
