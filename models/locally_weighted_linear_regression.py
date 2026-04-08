import matplotlib.pyplot as plt
import numpy as np

from .base import Model


class lw_LinearRegression(Model):
    def __init__(self):
        super().__init__()
        self.theta = None
        self.tau = None
        self.X_train = None
        self.y_train = None
        
    def fit(self, X, y, tau):
        self.tau = tau
        self.X_train = X
        self.y_train = y
    
    
    
        
    

        
