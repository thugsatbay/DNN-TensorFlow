import tensorflow as tf

'''
gdhody 1/25/2017
'''

'''
input > weight > hiddenlayer 1 > weight > hiddenlayer2 > weight > output layer
Each hidden layer has activation function

neural structure
'''

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/",one_hot=True)

hiddenLayerNodes=[500,500,500]

numbers=10
batch_size=100
x=tf.placeholder('float',[None,784])#input data height*width, image flattened in pixels, size is optional useful to check
y=tf.placeholder('float')

#input*data*weights + biases

def neural_network_model(data):
    hidden_1_layer={'weights':tf.Variable(tf.random_normal([784,hiddenLayerNodes[0]])),'biases':tf.Variable(tf.random_normal([hiddenLayerNodes[0]]))}

    hidden_2_layer={'weights':tf.Variable(tf.random_normal([hiddenLayerNodes[0],hiddenLayerNodes[1]])),'biases':tf.Variable(tf.random_normal([hiddenLayerNodes[1]]))}

    hidden_3_layer={'weights':tf.Variable(tf.random_normal([hiddenLayerNodes[1],hiddenLayerNodes[2]])),'biases':tf.Variable(tf.random_normal([hiddenLayerNodes[2]]))}

    hidden_4_layer={'weights':tf.Variable(tf.random_normal([hiddenLayerNodes[2],numbers])),'biases':tf.Variable(tf.random_normal([numbers]))}


    l1=tf.add(tf.matmul(data,hidden_1_layer['weights']),hidden_1_layer['biases'])

    l1=tf.nn.relu(l1)#rectified


    l2=tf.add(tf.matmul(l1,hidden_2_layer['weights']),hidden_2_layer['biases'])

    l2=tf.nn.relu(l2)#rectified



    l3=tf.add(tf.matmul(l2,hidden_3_layer['weights']),hidden_3_layer['biases'])

    l3=tf.nn.relu(l3)#rectified



    output=tf.matmul(l3,hidden_4_layer['weights'])+hidden_4_layer['biases']

    return output



def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction,y))

    optimizer=tf.train.AdamOptimizer().minimize(cost)

    epochs=10

    hm_epochs = 10
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))


train_neural_network(x)
