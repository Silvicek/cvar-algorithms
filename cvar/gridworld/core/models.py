import pickle


# class Model:
#     """ Container for safe saving and retrieval of VI/Q-learning models."""
#
#     def __init__(self, world, model, **kwargs):
#         self.world = world
#         self.model = model
#         self.info = kwargs

def save(path, world, model, **kwargs):
    pickle.dump((world, model, kwargs), open(path, mode='wb'))


