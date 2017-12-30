import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.ops.math_ops import to_int64
mnist=input_data.read_data_sets('/tmp/data', one_hot=True)

batch_size=100
n_nd1=450
n_nd2=600
n_classes=10

x=tf.placeholder(tf.float32, [None, 784])
y=tf.placeholder(tf.float32)

def node_control(varx):
    one_hl={'weights':tf.Variable(tf.random_uniform([784, n_nd1])),
            'biases':tf.Variable(tf.random_uniform([n_nd1]))
        }
    
    two_hl={'weights':tf.Variable(tf.random_uniform([n_nd1, n_nd2])),
            'biases':tf.Variable(tf.random_uniform([n_nd2]))
        }
    output_lay={'weights':tf.Variable(tf.random_uniform([n_nd2, n_classes])),
                'biases':tf.Variable(tf.random_uniform([n_classes]))
        }

    #working out the neural formula
    
    xone_hl=tf.add(tf.matmul(varx, one_hl['weights']), one_hl['biases'])
    xone_hl=tf.nn.relu(xone_hl)
    
    xtwo_hl=tf.add(tf.matmul(xone_hl, two_hl['weights']), two_hl['biases'])
    xtwo_hl=tf.nn.relu(xtwo_hl)
    
    xoutput_lay=tf.add(tf.matmul(xtwo_hl, output_lay['weights']), output_lay['biases'])
    olay=tf.nn.softmax(xoutput_lay)
    return olay
   
def train_nn(x):
    prediction=node_control(x)
    prediction=tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction)
    cost=tf.reduce_mean(prediction)
    optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)
    hm_epoch=30
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for epoch in range(hm_epoch):
            epoch_loss=0
            total_batch=int(mnist.train.num_examples/batch_size)
            
            for _ in range(total_batch):
                batch_x,batch_y=mnist.train.next_batch(batch_size)
              
                sess.run(optimizer, feed_dict={x:batch_x, y:batch_y})
                epoch_loss+=sess.run(cost, feed_dict={x:batch_x, y:batch_y})/total_batch
                  
            if epoch%2==0:
                print('Epoch: ', epoch, 'completed out of', hm_epoch,'loss:', epoch_loss)
        
        correct=tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
        accuracy=tf.reduce_mean(tf.cast(correct, tf.float32))
        
        print("Accuracy:", accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))
        
                
                
                
                
                
#kickstart the process by feeding the method data(x)                
train_nn(x)
                
                

        
        
        
        
        
        
        
        
        
        
    
    
    
