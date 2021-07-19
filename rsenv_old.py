class RS_Env(py_environment.PyEnvironment):
    def __init__(self, ratings_df, embedding_dim, state_size):
        self.users_num = ratings_df["UserID"].max() + 1
        self.items_num = ratings_df["MovieID"].max() + 1
        self.ratings_df = ratings_df
        self.pos_ratings_df = ratings_df.loc[ratings_df["Rating"] >= 4]
        self.embedding_dim = embedding_dim
        self.state_size = state_size
        
            
        self._action_spec = array_spec.ArraySpec(shape = (1, ), dtype = np.float32, name = "action")
        self._observation_spec = array_spec.ArraySpec(shape = (state_size, ), dtype = np.int64, name = "state")
        

#         self.embedding_network = UserMovieEmbedding(users_num, items_num, embedding_dim)
#         self.embedding_network([np.zeros((1,)),np.zeros((1,))])
#         self.embedding_network.load_weights('save_weights/user_movie_embedding_case4.h5')
        
        
        self.valid_users = self._generate_valid_user()
        
        # reset env
        self.target_user = np.random.choice(self.valid_users, size = 1).item()
        
        self.user_df = self.ratings_df.loc[self.ratings_df["UserID"] == self.target_user]
        self.movie_rate_dict = defaultdict(lambda: -0.5, zip(self.user_df["MovieID"], self.user_df["Rating"]))
        self._state = self.user_df.loc[self.user_df["Rating"] >= 4, "MovieID"].head(self.state_size).values
        self.user_items = self.user_df["MovieID"].values
        
        self.recommended_items = set(self._state)
        
        
        
        for x in self.recommended_items:
            self.movie_rate_dict[x] = -0.5
            
        self._state = 0
        self._episode_ended = False
        
        
    def action_spec(self):
        return self._action_spec
    
    def observation_spec(self):
        return self._observation_spec
    
    def _reset(self):        
        
        self.target_user = np.random.choice(self.valid_users, size = 1)
        
        self.user_df = self.ratings_df.loc[self.ratings_df["UserID"] == self.target_user.item()]
        self.movie_rate_dict = defaultdict(lambda: -0.5, zip(self.user_df["MovieID"], self.user_df["Rating"]))
        self._state = self.user_df.loc[self.user_df["Rating"] >= 4, "MovieID"].head(self.state_size).values
        self.user_items = self.user_df["MovieID"].values

        self.recommended_items = set(self._state)
        
        for x in self.recommended_items:
            self.movie_rate_dict[x] = -0.5
        
        self._episode_ended = False
        
        return ts.restart(self._state)
        
    def _generate_valid_user(self):
        temp = self.ratings_df.loc[ratings_df["Rating"] >= 4].groupby(["UserID"])["Rating"].count()
        valid_users = temp.loc[temp >= self.state_size].index
        
        return valid_users
    
    def _step(self, action):
        
        if self._episode_ended:
            return self.reset()
        
        if self.movie_rate_dict[action] > 0:
            reward = (self.movie_rate_dict[action]-3)/2
            if reward > 0:
                self._state = np.append(self._state[1:], values = action)
        else:
            reward = self.movie_rate_dict[action]
        
        # Once item has been recommended, it will return negative reward next time.
        self.movie_rate_dict[action] = 0
        
        self.recommended_items.add(action)
        
        if len(self.recommended_items) > 20:
            self._episode_ended = True
            
        if self._episode_ended:
            return ts.termination(self._state, reward)
        else:
            return ts.transition(self._state, reward, discount = 0.9)