import numpy as np
import torch
import torch.optim as optim


class DQN(object):

    def __init__(self, env, model, target_model, replay_buffer, config, name_agent="dqn"):

        self.name_agent = name_agent

        self.dim_space = env.observation_space.n+1
        self.nb_actions = env.action_space.n
        self.epsilon = config.epsilon_start

        self.epsilon_final = config.epsilon_final
        self.epsilon_start = config.epsilon_start
        self.epsilon_decay = config.epsilon_decay

        self.gamma = config.gamma
        self.replay_buffer = replay_buffer
        self.environment = env
        self.batch_size = config.batch_size
        self.update_nb_iter = config.update_nb_iter

        # q0
        self.model = model
        # q0_barre
        self.target_model = target_model
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)

        #
        self.loss_data = []
        self.rewards = []

    def forward(self, value):
        pred = self.model(value)
        return pred

    def action(self, state):
        """
            Select an action to take with an epsilon greedy strategy.
            Epsilon greedy strategy: with probability ϵ select a random action,
            otherwise select a = argmax Q(s_t,a)
            Params
            ------
            state:
                current state
            Returns
            -------
            action: int
                The action the agent will take.
        """

        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.nb_actions)

        else:
            prediction = self.predict_model(np.atleast_2d(self.preprocess(state)))[0].detach().numpy()
            for i in range(self.nb_actions):
                if state.board[i] != 0:
                    prediction[i] = -1e7
            return int(np.argmax(prediction))
        
    def preprocess(self, state):
        result = state.board[:]
        result.append(state.mark)

        return result
    
    def predict_model(self, X):
        if type(X)==np.ndarray:
            X = torch.from_numpy(X).float()
        return self.model(X)
    
    def predict_target_model(self, X):
        if type(X)==np.ndarray:
            X = torch.from_numpy(X).float()
        return self.target_model(X)
    
    def loss(self):
        """
            the loss is equal to:
                    R_t+1 + γ_t+1 qθ(S_t+1, argmax qθ(S_t+1, a′)) − qθ(S_t,A_t))^2
        """
        states, actions, rewards, next_states, finish = self.replay_buffer.sample()
        actions = actions.long()
        
        states = np.asarray([self.preprocess(state) for state in states])
        next_states = np.asarray([self.preprocess(next_state) for next_state in next_states])
        # gather function is used to take in the torch tensor the proba at the actions' indices

        # squeeze and unsqueeze is used to reshape the tensor at the good dimensions for the computation
        # probabilty next_states (using the current model) considering the action made
        # qθ(St,At)
        q0 = self.predict_model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # argmax qθ_barre(S_t+1, a′)
        max_next_q0 = self.predict_target_model(next_states).max(1)[0] * (1-finish)

        Rt_gamma_max = (rewards + self.gamma * max_next_q0)

        loss = (q0 - Rt_gamma_max).pow(2).sum()

        return loss

    def do_one_step(self):

        loss = self.loss()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.loss_data.append(loss.item())

    def update_epsilon(self, i):
        """
            update the probability of exploration
            params:
                i:int
                    the current iteration of the agent
        """

        self.epsilon = self.epsilon_final + (self.epsilon_start - self.epsilon_final) * np.exp(-1. * i / self.epsilon_decay)

    def update_model(self):
        """
            At every update_nb_iter step, we copy the parameters from our DQN network to
            update the target network.
        """
        self.target_model.load_state_dict(self.model.state_dict())

    def save_data(self, state, action, reward, next_state, finish):
        """
            Store a new experience in the buffer.
            Params
            ------
            state: S_t
            action: A_t
            reward: R_t+1
            next_state: S_t+1
            finish: 1 if the action lead to the end of the episode, 0 otherwise
        """

        self.replay_buffer.append_data(state, action, reward, next_state, finish)

    def train(self, nb_episode):

        state = self.environment.reset()
        sum_reward_game = 0
        
        i_episode = 0
        i = 0
        while i_episode<nb_episode:
            action = self.action(state)
            next_state, reward, finish, info = self.environment.step(action)
            self.update_epsilon(i)
            if finish == 1:
                i_episode += 1
                if reward == 1:
                    reward = 20
                elif reward == 0:
                    reward = -20
                else:
                    reward = 10
            else:
                reward = 0.5
                
            sum_reward_game += reward
            self.save_data(state, action, reward, next_state, finish)

            if finish == 1:
                print(i_episode, sum_reward_game)
                self.environment.render()
                # if the episode is finished, we need to restart the "game"
                state = self.environment.reset()
                # add the sum of rewards of the episode in a list
                # (we can plot it in the end to see the evolution of performance)
                self.rewards.append(sum_reward_game)
                sum_reward_game = 0
                

            if i > self.batch_size:
                # as soon as possible, the agent begin to learn
                self.do_one_step()

            if i % self.update_nb_iter == 0:
                # update the target network
                self.update_model()
                
            i += 1