import pickle
import os


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    print(name)
    if (os.path.exists(name + ".pkl")):
        with open(name + '.pkl', 'rb') as f:
            return pickle.load(f)
    else:
        raise Exception("The file does not exist")
