from algo.DQN import DQN
from model.ModelDuelings import ModelDuelings
from buffer.ReplayBuffer import ReplayBuffer
from env.puissance4 import puissance4
from config import Config




if __name__ == '__main__':
    nb_episode = 500
    
    env = puissance4()
    conf = Config()
    replay_buffer = ReplayBuffer(max_data=10000,
                                 batch_size=conf.batch_size)
    model = ModelDuelings(input_shape=env.observation_space.n+1,
                          output_shape=env.action_space.n)
    target_model = ModelDuelings(input_shape=env.observation_space.n+1,
                          output_shape=env.action_space.n)
    
    dqn = DQN(env=env,
              model=model,
              target_model=target_model,
              replay_buffer=replay_buffer,
              config=conf)
    

    dqn.train(nb_episode=nb_episode)

    #print(dqn.rewards)
