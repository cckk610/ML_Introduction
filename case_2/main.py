import numpy as np

# 随机初始化参数
h = 100 # 隐层大小
W = 0.01 * np.random.randn(D,h)
b = np.zeros((1,h))
W2 = 0.01 * np.random.randn(h,K)
b2 = np.zeros((1,K))
 
# 手动敲定的几个参数
step_size = 1e-0
reg = 1e-3 # 正则化参数
 
 hidden_layer = np.maximum(0, np.dot(X, W) + b)
 
# 梯度迭代与循环
num_examples = X.shape[0]
for i in xrange(10000):
 
  scores = np.dot(hidden_layer, W2) + b2
 
  # 计算类别概率
  exp_scores = np.exp(scores)
  probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K]
 
  # 计算互熵损失与正则化项
  corect_logprobs = -np.log(probs[range(num_examples),y])
  data_loss = np.sum(corect_logprobs)/num_examples
  reg_loss = 0.5*reg*np.sum(W*W) + 0.5*reg*np.sum(W2*W2)
  loss = data_loss + reg_loss
  if i % 1000 == 0:
    print "iteration %d: loss %f" % (i, loss)
 
  # 计算梯度
  dscores = probs
  dscores[range(num_examples),y] -= 1
  dscores /= num_examples
 
  # 梯度回传
  dW2 = np.dot(hidden_layer.T, dscores)
  db2 = np.sum(dscores, axis=0, keepdims=True)
 
  dhidden = np.dot(dscores, W2.T)
 
  dhidden[hidden_layer <= 0] = 0
  # 拿到最后W,b上的梯度
  dW = np.dot(X.T, dhidden)
  db = np.sum(dhidden, axis=0, keepdims=True)
 
  # 加上正则化梯度部分
  dW2 += reg * W2
  dW += reg * W
 
  # 参数迭代与更新
  W += -step_size * dW
  b += -step_size * db
  W2 += -step_size * dW2
  b2 += -step_size * db2