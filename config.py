


class Config(object):
    
    gamma = 0.99
    batch_size = 32
    learning_rate = 1e-4
    update_nb_iter = 200
    
    epsilon_final = 0.01
    epsilon_start = 0.99
    epsilon_decay = 200

    beta_start = 0.4
    beta_final = 1.         
    beta_decay = 200        
    
    w = 0.8             
    
    nb_steps = 3
    
    factorized = True
    sigma = 0.5


    nb_atoms = 51
    Z_min = -10
    Z_max = 10