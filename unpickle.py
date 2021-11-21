import pickle


with open('rand_test_config_0.pkl', 'rb') as f:
    data = pickle.load(f)
    print(data)