import torch


class Sensor(object):
    
    def get_noise(x, y):
        pass


class SensingAgent(object):
    def __init__(self, start, ns=1):
        self.x = start[0]
        self.y = start[1]
        self.z = 0
        self.ns = ns
        self.travel = 0

    def get_observations():
        pass

        

class SensorNetwork(object):
    def __init__(self, Ns):
        self.Ns = Ns

    def seed(self):
        pass

    def MITE(self, args=None):
        pass

    '''Optimize the objective function given strategy'''
    def optimize(self, E):
        pass