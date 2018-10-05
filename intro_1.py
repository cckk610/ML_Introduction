import numpy as np
from matplotlib import pyplot


# sigmoid function
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))
 
# input dataset
X=np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
 
# output dataset            
y = np.array([[0.1], [0.2], [0.3], [0.4], [0.5]])
 
# seed random numbers to make calculation
# deterministic (just a good practice)
# np.random.seed(1)

pyplot.plot(X, y)
#pyplot.show()
 
# initialize weights randomly with mean 0
syn_0 = 2*np.random.random((2,10)) - 1
syn_1 = 2*np.random.random((10,1)) - 1
 
for iter in xrange(20000):
    # forward propagation
    layer_0 = X
    layer_1 = nonlin(np.dot(layer_0,syn_0))
    layer_2 = nonlin(np.dot(layer_1,syn_1))
 
    # how much did we miss?
    l2_error = y - layer_2
    l2_delta = l2_error * nonlin(layer_2,True)
    
    if (iter% 1000) == 0:
        print "Error:" + str(np.mean(np.abs(l2_error)))
    
 
    # multiply how much we missed by the 
    # slope of the sigmoid at the values in l1
    l1_error = l2_delta.dot(syn_1.T)
    l1_delta = l1_error * nonlin(layer_1,True)
 
    # update weights
    syn_1 += np.dot(layer_1.T,l2_delta)
    syn_0 += np.dot(layer_0.T,l1_delta)
    
print "Output After Training:"
print layer_2
#print syn_0
#print syn_1
#pyplot.plot(layer_1)
#pyplot.show()

if __name__ == '__main__':
    print 'End'