import numpy as np

class MultiVarLinearReg:
    def __init__(self,features,target):
        self.features = np.array(features)
        self.target = np.array(target)

    def normal_eq(self):
        # Applied transposition for the sake of following along with the normal equation when applying matrix operations
        X = (self.features).T
        y = self.target

        # left half of normal equation
        left = np.linalg.inv(np.dot(X.T,X))
        #right half of normal equation
        right = np.dot(X.T, y)
        # returns a vector of theta in the form of a list
        return np.dot(left,right)

    def predict(self):
        X = (self.features)
        actual = self.target
        theta_vector = self.normal_eq()
        prediction = np.dot(theta_vector,X)
        # loss = prediction - actual
        return prediction





                          # Features: [House size], [# of bed room]   # target: sale price (1000x)
model = MultiVarLinearReg(features=[[2104,1496,1534,852],[5,3,3,2]], target = [460,232,315,178])
model.normal_eq()
print(model.predict())
