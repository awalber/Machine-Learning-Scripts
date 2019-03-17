# This code was written by Aaron Walber to display an understanding of machine learning
# topics taught in Andrew Ng's online machine learning course found on coursera. The
# course is taught in Octave, and I wanted to translate what I learned
# to python for my own benefit. The data used in this 
# code is from the third exercise of that course.
from numpy import *
from random import randint
from pylab import xlabel, ylabel, title, plot, show
from scipy.optimize import fmin_cg
from matplotlib import pyplot
from scipy.io import loadmat
import matplotlib.image as mpimg
import os
from scipy.special import expit
num_labels = 10
data = loadmat("D:/Python/Python/Custom_Projects/Machine_Learning/ex3data1.mat")
X = data['X']
y = data['y']
weights = loadmat("D:\Python\Python\Custom_Projects\Machine_Learning\ex3weights.mat")
Theta1 = weights['Theta1']
Theta2 = weights['Theta2']
def plot_image(X,image_num,n):
    min_num = X.min()
    max_num = X.max()
    X_new = X[image_num,:]
    X_new = X_new.reshape(int(sqrt(n)),int(sqrt(n)))
    X_new = X_new.T
    img = pyplot.imshow(X_new, cmap='gray', vmin=min_num, vmax=max_num)
    return img

def Cost(theta,X,y,L):
    X_ones=ones((m,1))
    X = hstack((X_ones,X))
    h = expit(theta@X.T).T
    theta_sqrd = square(theta)
    J = (1/m)*(-log(h).T*y - log(1-h).T*(1-y))+(L/(2*m))*sum(theta_sqrd)
    return J

def gradient(theta,X,y,L):
    X_ones=ones((m,1))
    X = hstack((X_ones,X))
    h = expit(theta@X.T).T
    part1 = (1/m)*(h.T@X - y.T@X)
    part2 = matrix((1/m)*L*theta)
    grad[0,0] = part1[0,0]
    grad[0,1:] = part1[0,1:] + part2[0,1:]
    new_matrix = grad.flatten()
    return new_matrix

def one_vs_all(X,y,num_labels,L):
    i = 1
    initial_theta = zeros((1,n+1))
    initial_theta = initial_theta.flatten()
    all_theta = zeros((num_labels,n+1))
    while i <=10:
        new_y = where(y==i,1,0)
        all_theta[i-1,:] = fmin_cg(Cost,x0=initial_theta,args=(X,new_y,L),fprime=gradient,maxiter=50)
        i+=1
    return all_theta

def predict(X,all_theta):
    X_ones=ones((m,1))
    X = hstack((X_ones,X))
    pred = X*all_theta.T
    pred = pred.argmax(axis=1)
    pred +=1
    result = where(y == pred,1,0)
    result = result.mean()
    return result

def nn_predict(Theta1,Theta2,X):
    m,n = shape(X)
    X_ones=ones((m,1))
    X = hstack((X_ones,X))
    z1 = X*Theta1.T
    a1 = expit(z1)
    a1_ones = ones((m,1))
    a1 = concatenate((a1_ones.T,a1.T)).T
    z2 = Theta2*a1.T
    a3 = expit(z2)
    row_max_index = a3.argmax(axis=0)
    row_max_index +=1
    return row_max_index

# This data is taken from Andrew Ng's machine learning course found on coursera.
# However, this code should work for any data set that can be split into a matrix of
# input features and the resulting output.

# Translate the values into a python matrix from matlab formatting
X = matrix(X)
y = matrix(y)
# Assign the dimensions of X to m and n.
m,n = shape(X)
# Set original theta to a vector of zeros with the same shape as X.
# I have taken the liberty of already adding a column of ones to theta.
theta = zeros((shape(X[1,:])))
p,l = shape(theta)
theta_ones=ones((p,1))
theta = hstack((theta_ones,theta))
grad = zeros((shape(theta)))
# This function finds the minima of the gradient function for various
# values of y (y = 1, 2, 3, ... 10) and outputs the accuracy of the
# training set using a multiclass classification method
all_theta = one_vs_all(X,y,num_labels,.1)
print("The one versus all training set accuracy is: ",predict(X,all_theta))
# This function uses predetermined weights to obtain the accuracy of the neural
# network as well as the predicted value of randomly selected digits.
result = nn_predict(Theta1,Theta2,X)
result = where(y == result.T,1,0)
result = result.mean()
print("The Neural Network's training set accuracy is: ", result)
while True:
    number = randint(0,m)
    img = plot_image(X,number,n)
    result = nn_predict(Theta1,Theta2,X[number,:])
    if result == 10:
        result = 0
    print(result)
    show(img)
    message = input("Type 'q' and press enter to quit.")
    if message == 'q':
        break
