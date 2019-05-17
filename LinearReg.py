import numpy as np
import matplotlib.pyplot as plt


class ML_Algo:
    def __init__(self,x_vals,y_vals,theta_0 = 0, theta_1 = 1):
        self.x_vals = x_vals
        self.y_vals = y_vals
        # for cost function and gradient descent
        self.theta_0 = theta_0
        self.theta_1 = theta_1

    def plots(self):
        plt.title("Data point")
        plt.scatter(self.x_vals,self.y_vals)
        plt.show()

    def cost_func(self,plot=True):
        theta_0 = self.theta_0
        theta_1 = self.theta_1
        m = len(self.x_vals)
        error = 0
        predictions = []
        for i in range(m): #
            hypothesis = theta_0 + (theta_1 * (i+1)) # b + mx... i+1 so that i can equal m at the end
            predictions.append(hypothesis)
            actual = self.y_vals[i]
            diff = (hypothesis - actual) ** 2
            error += diff
        squared_error = 1/(2*m) * error
        if plot is True:
            plt.title("Regression prediction vs actual data")
            plt.scatter(self.x_vals,self.y_vals)
            plt.plot(self.x_vals,predictions,color='red')
            plt.show()
        # print("Squared error:", round(squared_error,3))
        return squared_error


    def gradient_descent(self,plot_costFunc=False,alpha=0.1,iterations=10):
        m = len(self.x_vals)
        def cost_func_theta0(theta_0,m=m,plot=False):
            error = 0
            predictions = []
            for i in range(m): #
                hypothesis = theta_0 + (theta_1 * (i+1)) # b + mx... i+1 so that i can equal m at the end
                predictions.append(hypothesis)
                actual = self.y_vals[i]
                diff = (hypothesis - actual)
                error += diff
            slope = 1/(m) * error
            if plot is True:
                plt.title("Regression prediction vs actual data")
                plt.scatter(self.x_vals,self.y_vals)
                plt.plot(self.x_vals,predictions,color='red')
                plt.show()
            return alpha * slope

        def cost_func_theta1(theta_1,m=m,plot=False):
            error = 0
            predictions = []
            for i in range(m): #
                hypothesis = theta_0 + (theta_1 * (i+1)) # b + mx... i+1 so that i can equal m at the end
                predictions.append(hypothesis)
                actual = self.y_vals[i]
                diff = (hypothesis - actual) * (i+1)
                error += diff
            slope = (1/m) * error
            if plot is True:
                plt.title("Regression prediction vs actual data")
                plt.scatter(self.x_vals,self.y_vals)
                plt.plot(self.x_vals,predictions,color='red')
                plt.show()
            return alpha * slope

        theta_0 = self.theta_0
        theta_1 = self.theta_1
        #while( (cost_func_theta0(theta_0) or cost_func_theta1(theta_1)) != 0 )
        print("==="*50)
        print("Running: ",iterations, " iterations of gradient descent")
        sqr_errors = []
        for i in range(iterations):
            tmp_theta_0 = theta_0 - cost_func_theta0(theta_0)
            tmp_theta_1 = theta_1 - cost_func_theta1(theta_1)
            theta_0 = tmp_theta_0
            theta_1 = tmp_theta_1
            self.theta_0 = theta_0
            self.theta_1 = theta_1
            if plot_costFunc is True:
                sqr_errors.append(self.cost_func(plot=False))
        print("Gradient Descent finished!")
        print("Final thetha 0: ",round(theta_0,3)," ","Final theta 1: ", round(theta_1,3))
        if plot_costFunc is True:
            plt.title("Plot of Cost Function in gradient descent")
            plt.plot(sqr_errors)
            plt.show()

# Test trial
Lin_reg = ML_Algo(x_vals=[1,2,3], y_vals = [4,5,6])
Lin_reg.cost_func() # running cost function before gradient descent
Lin_reg.gradient_descent(plot_costFunc=True)
Lin_reg.cost_func() # running cost function AFTER gradient descent
