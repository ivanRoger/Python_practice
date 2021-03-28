import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt    # matplotlib:一個可視化數據的模組

def add_layer(inputs,in_size,activation_function=None):  #  加一個層,None: 默認沒有激勵函數(默認成線性函數)
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))  #  在生成初始的 Weights時,使用隨機變量較佳
    biases = tf.Variable(tf.zeros([1, out_size])+ 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases   #  tf.matmul(inputs, Weights): inputs乘以Weights
    if activation_function is None:
        outputs = Wx_plus_b
    else:  #  activation_function不為None的時候
        outputs = activation_function(Wx_plus_b)
    return outputs

x_data = np.linspace(-1,1,300, dtype=np.float32)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
# noise:噪點(mean為5,平方差為0.05)
y_data = np.square(x_data) - 0.5 + noise  #  np.square(x_data): X data的二次方

xs = tf.placeholder(tf.float32,[None,1])   #  None:無論給多少個sample都ok
ys = tf.placeholder(tf.float32,[None,1])

#  定義隱藏層
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu) # 激勵函數:tf.nn.relu
#  定義輸出層
prediction= add_layer(l1, 10, 1, activation_function=None)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys- prediction),
                     reduction_indices=[1]))
#  tf.reduce_sum:對每個例子求和; tf.reduce_mean:對每個例子求平均值

# 選擇一個優化器Optimizer
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
#  選擇學習效率:0.1 ; minimize(loss):最小化誤差

#  非常重要的步驟!!使用變量時,要先對它進行初始化,缺少此步驟將無法進行後續運算
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)   # 直到此步驟才會開始運算

fig = plt.figure()
ax = fig.add_subplot(1,11)   #  ax:可做一個連續性的畫圖
ax.scatter(x_data,y_data)   #  用點的形式把它plot上來
plt.show()
for i in range(1000):   #  學習1000步(重複1000次)
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
    if i% 50:
        print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))
    
    


