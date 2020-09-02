# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import argparse 
import csv
import math

from numpy import genfromtxt
np.random.seed(1234)
parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, metavar='',help='(initial learning rate for gradient descent based algorithms)',default = .0007)
parser.add_argument('--init', type = int , metavar ='', help = '(weghit initialization method)', default = 1)
parser.add_argument( '--batch_size',type=int,metavar='', help='(the batch size to be used)', default= 128)
parser.add_argument('--epochs',type=int,metavar='', help='(number of passes over the data)',default = 27)
parser.add_argument('--dataAugment', type = int, metavar = '',help ='(data augmentation is used or not)',default = 1)

parser.add_argument( '--save_dir',type=str, metavar='',help='(the directory in which the pickled model should be saved - by model we mean all the weights and biases of the network)', default='/home/ubuntu/my_test_model.ckpt.meta')

parser.add_argument("--train",type=str,help="path to training dataset",default="train.csv")
parser.add_argument("--val",type=str,help="path to validation dataset",default="valid.csv")
parser.add_argument("--test",type=str,help="path to test dataset",default="test.csv")

args = parser.parse_args()

learning_rate =args.lr
batch_size = args.batch_size
initialization = args.init
augment = args.dataAugment
epochs = args.epochs
train = args.train
test = args.test
valid = args.val
directory = args.save_dir

training_iters = epochs
learning_rate = learning_rate
# learning_rate =args.lr
batch_size = batch_size

# MNIST total classes (0-9 digits)
n_classes = 20
init = initialization


my_train_data = genfromtxt(train, delimiter=',')
length1 = len(my_train_data[0])
length_row_train = len(my_train_data)
#length_row_train = 4
#print("length_row_train",length_row_train)
train_data = np.array(my_train_data[1:length_row_train ,1:length1-1])/255.0
train_X = train_data.reshape(-1, 64, 64, 3)
train_Y = np.array(my_train_data[1:length_row_train,length1-1:length1])

print(len(train_data[0]))
# flip starts

def flip_images_left_right(X_imgs):
    X_flip = []
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape = (None,64, 64, 3))
    tf_img1 = tf.image.flip_left_right(X)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        flipped_imgs = sess.run([tf_img1], feed_dict = {X: X_imgs})
        X_flip.extend(flipped_imgs)
    X_flip = np.array(X_flip, dtype = np.float32)
    return X_flip

def flip_images_up_down(X_imgs):
    X_flip = []
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape = (None,64, 64, 3))
    tf_img1 = tf.image.flip_up_down(X)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        flipped_imgs = sess.run([tf_img1], feed_dict = {X: X_imgs})
        X_flip.extend(flipped_imgs)
    X_flip = np.array(X_flip, dtype = np.float32)
    return X_flip



def add_salt_pepper_noise(X_imgs):
    # Need to produce a copy as to not modify the original image
    X_imgs_copy = X_imgs.copy()
    row, col, _ = X_imgs_copy[0].shape
    salt_vs_pepper = 0.2
    amount = 0.004
    num_salt = np.ceil(amount * X_imgs_copy[0].size * salt_vs_pepper)
    num_pepper = np.ceil(amount * X_imgs_copy[0].size * (1.0 - salt_vs_pepper))
    
    for X_img in X_imgs_copy:
      
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in X_img.shape]
        X_img[coords[0], coords[1], :] = 1
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in X_img.shape]
        X_img[coords[0], coords[1], :] = 0
    return X_imgs_copy


def rotate_image(X_imgs):
    x_rotate = []
    tf.reset_default_graph()
    shape = [None, 64, 64, 3]
    y = tf.placeholder(dtype = tf.float32, shape = shape)
    rot_tf_45 = tf.contrib.image.rotate(y, angles=4.71)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        roted_images = sess.run([rot_tf_45], feed_dict = {y: X_imgs})
        x_rotate.extend(roted_images)
    x_rotate = np.array(x_rotate, dtype = np.float32)
    return x_rotate
  

# for i in range(1):
if augment == 1:
    flipped_images = flip_images_left_right(train_X)
    flipped = np.squeeze(flipped_images)
    #print("flipped",flipped.shape)
    flipped_X = np.concatenate((train_X,flipped),axis=0)
    flipped_Y = np.concatenate((train_Y,train_Y),axis=0)
    #print(len(flipped_X))
    #print("combined",flipped_X.shape)

    #salt_pepper_noise_imgs = add_salt_pepper_noise(train_X)
    #rot = np.squeeze(salt_pepper_noise_imgs)
    #print("pepersalt",rot.shape)

    #salt_X = np.concatenate((flipped_X,rot),axis=0)
    #salt_Y = np.concatenate((flipped_Y,train_Y),axis=0)
    #print(salt_X.shape)


    roted_images = flip_images_up_down(train_X)
    rot = np.squeeze(roted_images)
    #print("roted",rot.shape)
    train_X = np.concatenate((flipped_X,rot),axis=0)
    train_Y = np.concatenate((flipped_Y,train_Y),axis=0)


#saltnpepperends
my_valid_data = genfromtxt(valid, delimiter=',')
length2 = len(my_valid_data[0])
length_row_valid = len(my_valid_data)
#length_row_valid = 16
#print("length_row_train",length_row_valid)
valid_data = np.array(my_valid_data[1:length_row_valid,1:length2-1])/255.0
print(len(valid_data[0]))
valid_x =valid_data.reshape(-1, 64, 64, 3)
valid_y = np.array(my_valid_data[1:length_row_valid,length2-1:length2])

def fooling(valid_x,pixel_change):
    
    X_imgs_copy = valid_x.copy()
  

    for i in range(len(valid_x)):
        for z in range(pixel_change):
            r = np.random.randint(0,64,2)
            p = np.random.uniform(0.0,0.1,1)
            s = np.random.randint(0,3,1)
            X_imgs_copy[i][r[0]][r[1]][s] = p
  
  
       
    return X_imgs_copy


def find_accuracy(a,b):
    count = 0
    length=  len(a)
    for c, d in zip(a,b):
        if c==d:
            count = count +1
    return count/(length*1.0)


my_test_data = genfromtxt(test, delimiter=',')
length2 = len(my_test_data[0])
length_row_test=len(my_test_data)
#length_row_test=16
#print("length_row_train",length_row_test)
test_data = np.array(my_test_data[1:length_row_test,1:length2])/255.0
print(len(test_data[0]))
test_x =test_data.reshape(-1, 64, 64, 3)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

tf.reset_default_graph()
def encode_into_prob(x,max_val):
    len=x.size
    y_list=[]
    for i in range(len):
        temp=[]
        for j in range(max_val):
            if(x[i]==j):
                temp.append(1)
            else:
                temp.append(0)
            #print(temp)
        y_list.append(temp)
    return y_list

y_true_train = np.array(encode_into_prob(train_Y, 20))
y_true_valid= np.array(encode_into_prob(valid_y, 20))


#both placeholders are of type float
x = tf.placeholder("float", [None, 64,64,3])
y = tf.placeholder("float", [None, n_classes])
drop_out_value = .5
dropout = tf.placeholder(tf.float32)
isnormalize = tf.placeholder_with_default(False, shape =(),name = 'isnormalize')

def conv2d(x, W, b, strides=1, isnormalize = False):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
   
    x = tf.nn.bias_add(x, b)
    batch_normal1 = tf.layers.batch_normalization(x, training = isnormalize)
    return tf.nn.relu(batch_normal1)
def last_layer(x, W, b, strides=1,isnormalize = False):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='VALID')

    x = tf.nn.bias_add(x, b)
    batch_normal = tf.layers.batch_normalization(x, training = isnormalize)
    return tf.nn.relu(batch_normal)


def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],padding='SAME')
if init == 1:

    weights = {
    'wc1': tf.get_variable('W0', shape=(5,5,3,32), initializer=tf.contrib.layers.xavier_initializer(seed=123))
    ,
    'wc2': tf.get_variable('W1', shape=(5,5,32,32), initializer=tf.contrib.layers.xavier_initializer(seed=123)),
    'wc3': tf.get_variable('W2', shape=(3,3,32,64), initializer=tf.contrib.layers.xavier_initializer(seed=123)),
    'wc4': tf.get_variable('W3', shape=(3,3,64,64), initializer=tf.contrib.layers.xavier_initializer(seed=123)),
    'wc5': tf.get_variable('W4', shape=(3,3,64,64), initializer=tf.contrib.layers.xavier_initializer(seed=123)),
    'wc6': tf.get_variable('W5', shape=(3,3,64,128), initializer=tf.contrib.layers.xavier_initializer(seed=123)),
    'wd1': tf.get_variable('W6', shape=(6272,256), initializer=tf.contrib.layers.xavier_initializer(seed=123)),
    'out': tf.get_variable('W7', shape=(256,n_classes), initializer=tf.contrib.layers.xavier_initializer(seed=123)),
    }

    biases = {
    'bc1': tf.get_variable('B0', shape=(32), initializer=tf.contrib.layers.xavier_initializer(seed=123)),
    'bc2': tf.get_variable('B1', shape=(32), initializer=tf.contrib.layers.xavier_initializer(seed=123)),
    'bc3': tf.get_variable('B2', shape=(64), initializer=tf.contrib.layers.xavier_initializer(seed=123)),
    'bc4': tf.get_variable('B3', shape=(64), initializer=tf.contrib.layers.xavier_initializer(seed=123)),
    'bc5': tf.get_variable('B4', shape=(64), initializer=tf.contrib.layers.xavier_initializer(seed=123)),
    'bc6': tf.get_variable('B5', shape=(128), initializer=tf.contrib.layers.xavier_initializer(seed=123)),
    'bd1': tf.get_variable('B6', shape=(256), initializer=tf.contrib.layers.xavier_initializer(seed=123)),
    'out': tf.get_variable('B7', shape=(20), initializer=tf.contrib.layers.xavier_initializer(seed=123)),
    }
elif init == 2:

    weights = {
    'wc1': tf.get_variable('W0', shape=(5, 5, 3, 32),
                           initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG",seed=123))
    ,
    'wc2': tf.get_variable('W1', shape=(5, 5, 32, 32),
                           initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG", seed=123)),
    'wc3': tf.get_variable('W2', shape=(3, 3, 32, 64),
                           initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG", seed=123)),
    'wc4': tf.get_variable('W3', shape=(3, 3, 64, 64),
                           initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG",seed=123)),
    'wc5': tf.get_variable('W4', shape=(3, 3, 64, 64),
                           initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG",seed=123)),
    'wc6': tf.get_variable('W5', shape=(3, 3, 64, 128),
                           initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG",seed=123)),
    'wd1': tf.get_variable('W6', shape=(6272, 256),
                           initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG",seed=123)),
    'out': tf.get_variable('W7', shape=(256, n_classes),
                           initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG",seed=123)),
}

    biases = {
    'bc1': tf.get_variable('B0', shape=(32),
                           initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG",seed=123)),
    'bc2': tf.get_variable('B1', shape=(32),
                           initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG",seed=123)),
    'bc3': tf.get_variable('B2', shape=(64),
                           initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG",seed=123)),
    'bc4': tf.get_variable('B3', shape=(64),
                           initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG",seed=123)),
    'bc5': tf.get_variable('B4', shape=(64),
                           initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG",seed=123)),
    'bc6': tf.get_variable('B5', shape=(128),
                           initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG",seed=123)),
    'bd1': tf.get_variable('B6', shape=(256),
                           initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG",seed=123)),
    'out': tf.get_variable('B7', shape=(20),
                           initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG",seed=123)),
}
def plotting_conv1_weights(weights, input_channel=1):
   
    w = sess.run(weights)
    w_min = np.min(w)
    w_max = np.max(w)

 
    num_filters = w.shape[3]

  
    num_grids = math.ceil(math.sqrt(num_filters))
    
  
    fig, axes = plt.subplots(4, 8)

    # Plot all the filter-weights.
    for i, ax in enumerate(axes.flat):
        
        if i<num_filters:
          
            img = w[:, :, input_channel, i]

            # Plot image.
            ax.imshow(img, vmin=w_min, vmax=w_max,
                      interpolation='nearest', cmap='seismic')
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.

    plt.show()
    plt.savefig('weights.png')
    plt.close()


def conv_net(x, weights, biases,dropout, isnormalize):
    conv1 = conv2d(x, weights['wc1'], biases['bc1'], isnormalize = True)
    
    conv2 = conv2d(conv1,  weights['wc2'], biases['bc2'] , isnormalize = True)
    
    pool1 = maxpool2d(conv2, k=2)
    conv3 = conv2d(pool1,  weights['wc3'], biases['bc3'], isnormalize = True)
    #pool11 = maxpool2d(conv3, k=2)
    conv4 = conv2d(conv3, weights['wc4'], biases['bc4'], isnormalize = True)
  
    pool2 = maxpool2d(conv4, k=2)
    conv5 = conv2d(pool2,  weights['wc5'], biases['bc5'], isnormalize = True)
    #pool22= maxpool2d(conv5, k=2)
    conv6 = last_layer(conv5,  weights['wc6'], biases['bc6'], isnormalize = True)
    
    pool3 = maxpool2d(conv6, k=2)
    
    fc1 = tf.reshape(pool3, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    drop_out = tf.nn.dropout(fc1,  keep_prob=dropout)
    
   
    out = tf.add(tf.matmul(drop_out, weights['out']), biases['out'])
   
    return  out

pred = conv_net(x, weights, biases,dropout, isnormalize)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))



extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(extra_update_ops):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(cost)

#Here you check whether the index of the maximum value of the predicted image is equal to the actual labelled image. and both will be a column vector.
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))

#calculate accuracy across all the given images and average them out.
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    train_loss = []
    test_loss = []
    train_accuracy = []
    test_accuracy = []
    min_valid_loss = 10000
    x_axis = []
    
    # summary_writer = tf.summary.FileWriter('./Output', sess.graph)
    for i in range(training_iters):
        train_loss_list = []
        for batch in range(len(train_X)//batch_size):
            batch_x = train_X[batch*batch_size:min((batch+1)*batch_size,len(train_X))]
            batch_y = y_true_train[batch*batch_size:min((batch+1)*batch_size,len(y_true_train))]
            # Run optimization op (backprop).
                # Calculate batch loss and accuracy
            opt = sess.run(train_op, feed_dict={x: batch_x,
                                                             y: batch_y,dropout:.75,isnormalize:True})
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                              y: batch_y,dropout:.75,isnormalize:True})
            train_loss_list.append(loss)
        print("Iter " + str(i) + ", Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))
        print("Optimization Finished!")

        # Calculate accuracy for all 10000 mnist test images
        test_acc,valid_loss = sess.run([accuracy,cost], feed_dict={x: valid_x,y : y_true_valid,dropout:1,isnormalize:False})
        train_loss.append(np.mean(train_loss_list))
        test_loss.append(valid_loss)
        train_accuracy.append(acc)
        test_accuracy.append(test_acc)
        print("Testing Accuracy:","{:.5f}".format(test_acc))
        x_axis.append(i)
        if i%5==0:
            if valid_loss<min_valid_loss:
                min_valid_loss = valid_loss
                saver = tf.train.Saver()
                saver.save(sess, directory)
            else:
                break
        
            
    # print(train_loss, len(train_loss))
    # print(test_loss)
    # print(x_axis)
    fig = plt.figure()
    xint = range(0, training_iters)
    plt.xticks(xint)
    plt.plot(x_axis,train_loss,label="training loss")
    plt.plot(x_axis,test_loss, label="validatiion loss")
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Loss vs iteration')
    plt.show()
    plt.savefig('train_valid_loss.png')
    plt.close(fig)

with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph(directory)
    new_saver.restore(sess, tf.train.latest_checkpoint('./'))

    

    accuracy_fianl_list = []
    acuuracy_y_list = []
    for i in range(13):
        test_fool_x = fooling(valid_x,np.power(2,i))
        re = sess.run(pred,feed_dict={x: test_fool_x,dropout:1,isnormalize:False})
        accura = find_accuracy(np.argmax(re, axis=1),valid_y)
        accuracy_fianl_list.append(accura)
        acuuracy_y_list.append(np.power(2,i))
    print("values are")    
    print(accuracy_fianl_list,acuuracy_y_list)
    fig2 = plt.figure()
    plt.plot(acuuracy_y_list,accuracy_fianl_list)
    plt.xlabel('pixel change')
    plt.ylabel('accuracy')
    plt.title('no.of pixel change vs accuracy')
    plt.show()
    plt.savefig('accuracy_pixel_change.png')
    plt.close(fig2)
    
    re = sess.run(pred,feed_dict={x: test_x,dropout:1,isnormalize:False})
    converted_weight = tf.convert_to_tensor(value= sess.run(weights['wc1']))
    plotting_conv1_weights(converted_weight)

    myprediction = np.argmax(re, axis=1)
    final = []
    count = 0
    for i in myprediction:
        element = []
        element.append(count)
        element.append(i)
        count = count + 1
        final.append(element)


    print(np.argmax(re, axis=1))
    final = np.array(final)

    #print("weights",sess.run(weights['wc1']))
    with open("submission.csv", 'w', newline='') as f:
        thewriter = csv.writer(f)
        row, col = final.shape
        thewriter.writerow(['id', 'label'])
        for i in range(row):
            thewriter.writerow(final[i])
    # summary_writer.close()



