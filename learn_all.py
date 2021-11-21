import torch
from torch import nn
import numpy as np
from train_models import ModelTrainer

def collect_data(embeddings, train_sets):
    all_data = []
    assert len(train_sets) == len(embeddings)
    for i in range(len(train_sets)):
        for j in range(len(train_sets[i])):
            step = train_sets[i][j]
            #step[0] = np.concatenate([step[0], np.array([1+int(i/10)])])
            step[0] = np.concatenate([step[0], embeddings[i]])
            all_data.append(step)
    return all_data

# TODO Try with RNN
if __name__ == '__main__':
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    dir = ''  
    envs = ['Ant', 'Crawler', 'Dog', 'Spindra']
    train_envs = 10
    train_size = 10
    embeddings = np.load('100K_node_embeddings.npy')
    matrix = np.load('transfer_graph.npy')
    train_sets = [np.load(dir+'data/' + env + '_train_10000_' + str(i) + '.npy', allow_pickle=True)[:train_size] for env in envs for i in range(train_envs)]
    valid_sets = [np.load(dir+'data/' + env + '_valid_2500_' + str(i) + '.npy', allow_pickle=True) for env in envs for i in range(100)]
    train_embeddings = []
    for i in range(len(envs)):
        train_embeddings.extend(embeddings[i*100:i*100+train_envs])
    all_data = collect_data(train_embeddings, train_sets)
    print('Dataset Size: '+str(len(all_data)))
    trainer = ModelTrainer(all_data, hid_size=64)
    model, L = trainer.train(epochs=300, verbose=10)

    ant_valid = collect_data(embeddings[:train_envs], valid_sets[:train_envs])
    crawler_valid = collect_data(embeddings[100:100+train_envs], valid_sets[100:100+train_envs])
    dog_valid = collect_data(embeddings[200:200+train_envs], valid_sets[200:200+train_envs])
    spindra_valid = collect_data(embeddings[300:300+train_envs], valid_sets[300:300+train_envs])

    print('')
    print('Seen Multi Network Validation')
    print('Ant Avg Valid: '+str(trainer.validate(ant_valid)))
    print('Crawler Avg Valid: '+str(trainer.validate(crawler_valid)))
    print('Dog Avg Valid: '+str(trainer.validate(dog_valid)))
    print('Spindra Avg Valid: '+str(trainer.validate(spindra_valid)))

    ant_valid = collect_data(embeddings[train_envs:100], valid_sets[train_envs:100])
    crawler_valid = collect_data(embeddings[100+train_envs:200], valid_sets[100+train_envs:200])
    dog_valid = collect_data(embeddings[200+train_envs:300], valid_sets[200+train_envs:300])
    spindra_valid = collect_data(embeddings[300+train_envs:400], valid_sets[300+train_envs:400])
    print('')
    print('Unseen Multi Network Validation')
    print('Ant Avg Valid: '+str(trainer.validate(ant_valid)))
    print('Crawler Avg Valid: '+str(trainer.validate(crawler_valid)))
    print('Dog Avg Valid: '+str(trainer.validate(dog_valid)))
    print('Spindra Avg Valid: '+str(trainer.validate(spindra_valid)))

    print('')
    print('Single Problem Network Validation: ')
    print('Ant Avg Valid: '+str(np.mean([matrix[0,i] for i in range(0,100)])))
    print('Crawler Avg Valid: '+str(np.mean([matrix[0,i] for i in range(100,200)])))#These are the lines which contained loss on the vanilla model which required changes.
    print('Dog Avg Valid: '+str(np.mean([matrix[0,i] for i in range(200,300)])))#
    print('Spindra Avg Valid: '+str(np.mean([matrix[0,i] for i in range(300,400)])))#
    print('Done')