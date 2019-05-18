import numpy as np

class MultiVarLinearReg:
    def __init__(self,features,target):
        self.features = np.array(features)
        self.target = np.array(target)

    def normal_eq(self):
        X = (self.features).T # Applied transposition for the sake of following along with the normal equation
        y = self.target

        # left half of normal equation
        left = np.linalg.inv(np.dot(X.T,X))
        #right half of normal equation
        right = np.dot(X.T, y)
        # returns thetha
        return np.dot(left,right)

    def cost_function(self):
        X = (self.features)
        actual = self.target
        theta_vector = self.normal_eq()
        hypothesis = np.dot(theta_vector,X)
        loss = hypothesis - actual
        return hypothesis, loss

# house size, num of bed rooms

theta = np.array([1,2])
X = np.array([1,2000])

                          # Features: House size, num of bed room
model = MultiVarLinearReg(features=[[2104,1496,1534,852],[5,3,3,2]], target = [460,232,315,178])
model.normal_eq()

model.cost_function()
