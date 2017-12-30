import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


mnist=input_data.read_data_sets('/tmp/data', one_hot=True)

batch_size=100

#weight
W=tf.Variable(tf.random_uniform([784,10]))
#bias
b=tf.Variable(tf.random_uniform([10]))

#x,y==placeholders
x=tf.placeholder(tf.float32, [None, 784])
y=tf.placeholder(tf.float32)


#a method to make this look a tad complicated than it is.
def process_nn(x):
    output=tf.add(tf.matmul(x,W),b)
    output=tf.nn.softmax(output)
    return output


#this method receives the output from process_nn() and works on it    
def train_nn(x):
    prediction=process_nn(x)
    cost=-tf.reduce_sum(y*tf.log(prediction))
    
    #Optimizer and a cost minimization process
    optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)
    hm_epoch=30
    
    #Starting a session
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
        
                
                
                
                
                
#kickstart the process passing x(data) to the method                
train_nn(x)
                
                
        
        
        
        
        
        
        
        
        
        
        
    
    
    
