import numpy as np
from matplotlib import pyplot


# sigmoid function
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))
 
# input dataset
X = np.array([[0.4, -0.7],
     [0.3, -0.5],
     [0.6,  0.1],
     [0.2,  0.4],
     [0.1, -0.2]])
 
# output dataset            
y = np.array([[0.1, 0.05, 0.3, 0.25, 0.12]]).T
 
# seed random numbers to make calculation
# deterministic (just a good practice)
# np.random.seed(1)

#pyplot.plot(X)
#pyplot.show()
 
# initialize weights randomly with mean 0
syn_0 = 2*np.random.random((2,1)) - 1
 
for iter in xrange(1000):
    # forward propagation
    layer_0 = X
    layer_1 = nonlin(np.dot(layer_0,syn_0))
 
    # how much did we miss?
    l1_error = y - layer_1
 
    # multiply how much we missed by the 
    # slope of the sigmoid at the values in l1
    l1_delta = l1_error * nonlin(layer_1,True)
 
    # update weights
    syn_0 += np.dot(layer_0.T,l1_delta)
    
print "Output After Training:"
print layer_1
#pyplot.plot(layer_1)
#pyplot.show()

if __name__ == '__main__':
    print 'End'