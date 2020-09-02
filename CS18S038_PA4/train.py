import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import seaborn as sns
import matplotlib.patheffects as PathEffects
from multiprocessing.dummy import Pool as ThreadPool
from sklearn.preprocessing import Binarizer
from sklearn.manifold import TSNE

data = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
prediction = pd.DataFrame(test, columns=['label'])
df = data
tf = test
del tf['id']
del tf['label']
#print("shape",df.shape)
del df['id']
#print("shape",df.shape)
del df['label']
#print("shape",df.shape)
df= df.values
tf = tf.values
#print(df[0])
# print(df[0])
binarizer = Binarizer(threshold=127.0).fit(df)
binaryX = binarizer.transform(df)
# print(binaryX[0])
df = binaryX
binarizer = Binarizer(threshold=127.0).fit(tf)
binaryX = binarizer.transform(tf)
# print(binaryX[0])
tf = binaryX


class RBM:
    """Restricted Boltzmann Machine."""

    def __init__(self, n_hidden=2, m_observe=784):
        """Initialize model.
        Args:
            n_hidden: int, the number of hidden units
            m_observe: int, the number of visible units
        """
        self.n_hidden = n_hidden
        self.m_visible = m_observe
        print("self.n_hidden ,self.m_visible",self.n_hidden ,self.m_visible)

        self.visible = None
        self.weight = np.random.rand(self.m_visible, self.n_hidden)  # [m, n]
        print("self.weight",self.weight.shape)
        self.a = np.random.rand(self.m_visible, 1)  # [m, 1]
        self.b = np.random.rand(self.n_hidden, 1)  # [n, 1]
        #print("self.a,self.b",self.a,self.b)

        self.alpha = 0.01
        self.avg_energy_record = []
        self.avg_error_list = np.array([])
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def train(self, data, epochs=2):
        """train the RBM
        Args:
            data: numpy ndarray of shape [N, m], representing N sample with m visible units
            epochs: int, the total number of epochs in training
        """

        self.avg_energy_record.clear()
        print("self.avg_energy_record",self.avg_energy_record)
        print("data.shape",data.shape)
        #print("self.m_visible",self.m_visible)
        self.visible = data.reshape(-1, self.m_visible)
        error_list = self.__contrastive_divergence(self.visible, epochs)
        #print("self.visible",self.visible)

        print("training finished")
        return error_list

    def __forward(self, v):
        #print("in forward",v.shape)
        h_dist = self.sigmoid(
            np.matmul(np.transpose(self.weight), v) + self.b)  # [n, 1]
        #print("h_dist",h_dist)
        return self.__sampling(h_dist)  # [n, 1]

    def __backward(self, h):
        v_dist = self.sigmoid(np.matmul(self.weight, h) + self.a)  # [m, 1]
        return self.__sampling(v_dist)  # [m, 1]

    def __sampling(self, distribution):
        dim = distribution.shape[0]
        #print("dim",dim)
        true_idx = np.random.uniform(0, 1, dim).reshape(dim, 1) <= distribution
        #print("true_idx",true_idx)
        sampled = np.zeros((dim, 1))
        sampled[true_idx] = 1  # [n, 1]
        return sampled        

    def __CD_1(self, v_n):
        #print("v_n",v_n)
        v_n = v_n.reshape(-1, 1)
        v_t = v_n
        for i in range(Nsteps):
          h_t = self.__forward(v_t)
          v_t = self.__backward(h_t)
        h_sampled = h_t
        v_sampled = v_t
        h_recon = self.__forward(v_sampled)
        
        self.weight += self.alpha * \
            (np.matmul(v_n, np.transpose(h_sampled)) -
             np.matmul(v_sampled, np.transpose(h_recon)))
        #print("self.weight ",self.weight )    
        self.a += self.alpha * (v_n - v_sampled)
        #print("self.a",self.a)
        self.b += self.alpha * (h_sampled - h_recon)

        self.energy_list.append(self._energy(v_n, h_recon))
        self.error_list.append(self._error(v_n, v_sampled))
        
    def __contrastive_divergence(self, data, max_epoch):
        #print("data",data)
        print("max_epoch",max_epoch)
        train_time = []
        for t in range(max_epoch):
            np.random.shuffle(data)
            self.energy_list = []
            self.error_list = []

            start = time.time()
            #print("start",start)
            pool = ThreadPool(5)
            #print("pool",pool)
            pool.map(self.__CD_1, data)
            end = time.time()
            #print("end",end)

            avg_energy = np.mean(self.energy_list)
            avg_error = np.mean(self.error_list)
            self.avg_error_list = np.append(self.avg_error_list, avg_error)
            print("[epoch {}] , average energy={}, average error={}".format(
                t,  avg_energy, avg_error))
            self.avg_energy_record.append(avg_energy)
            train_time.append(end - start)
        #print("Average Training Time: {:.2f}".format(np.mean(train_time)))
        return self.avg_error_list

    def _energy(self, visible, hidden):
        #print("in test")
        return - np.inner(self.a.flatten(), visible.flatten()) - np.inner(self.b.flatten(), hidden.flatten()) \
            - np.matmul(np.matmul(visible.transpose(), self.weight), hidden)

    def _error(self, v_n, v_sampled):
      reconstruction_err = LA.norm(v_n - v_sampled)
      return reconstruction_err
    
    def energy(self, v):
        v = np.transpose(v)
        hidden = self.__forward(v)
        #print("in this loop")
        a = self._energy(v, hidden)
        return hidden

    def __Gibbs_sampling(self, v_init, num_iter=10):
        v_t = v_init.reshape(-1, 1)
        v_Matrix = np.zeros((64,784))
        print('in Gibbs sample')
        print(v_Matrix.shape)
        p = 0
        for t in range(num_iter):
          h_dist = self.sigmoid(
              np.matmul(np.transpose(self.weight), v_t) + self.b)  # [n, 1]
          h_t = self.__sampling(h_dist)  # [n, 1]

          v_dist = self.sigmoid(
              np.matmul(self.weight, h_t) + self.a)  # [m, 1]
          v_t = self.__sampling(v_dist)  # [m, 1]
          if(t%100 == 0):
            v_Matrix[p,:] = np.ravel(v_t)
            p += 1
        return v_Matrix

    def sample(self, num_iter=10, v_init=None):
        """Sample from trained model.
        Args:
            num_iter: int, the number of iterations used in Gibbs sampling
            v_init: numpy ndarray of shape [m, 1], the initial visible units (default: None)
        Return:
            v: numpy ndarray of shape [m, 1], the visible units reconstructed from RBM.
        """
        print('in sample')
        if v_init is None:
            v_init = np.random.rand(self.m_visible, 1)
        v_Matrix = self.__Gibbs_sampling(v_init, num_iter)
        return v_Matrix


Nsteps = 1 #k
Nhidden = 100 #n
Nepochs = 10
rbm = RBM(Nhidden, 784)
print("Start RBM training.")
    # train rbm model using mnist
error_list = rbm.train(df, epochs=Nepochs)
print("Error_list shape", error_list.shape[0])
print("Finish RBM training.")

fig2 = plt.figure()
plt.plot(list(range(Nepochs)), error_list)
plt.xlabel("Nepochs")
plt.ylabel("Reconstruction error")
plt.title('learning curve')
legend = plt.legend(loc='upper left', shadow=True, fontsize='x-small')
legend.get_frame()
plt.show()
fig2.savefig('learning_curve.png',dpi=300)
print('error_list:',error_list)

import csv

with open('/content/drive/My Drive/error_k'+str(Nsteps)+'_n'+str(Nhidden)+'.csv', 'w') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(error_list)

##########################################################
hidrep = []
for i in range(len(test)):
    hid = rbm.energy(tf[i:i+1,])
    hidrep.append(hid)

hidrep =  np.asarray(hidrep)  
hidrep = np.squeeze(hidrep)
test_data = tf
n_sne = len(test_data)

time_start = time.time()
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(hidrep)
print("(tsne_results",tsne_results)

print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
import seaborn as sns
import matplotlib.patheffects as PathEffects
def fashion_scatter(x, colors):
    # choose a color palette with seaborn.
    num_classes = len(np.unique(colors))
    print(num_classes)
    palette = np.array(sns.color_palette("hls", num_classes))

    # create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40, c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')
    # add the labels for each digit corresponding to the label
    txts = []

    for i in range(num_classes):

        # Position of each label at median of data points.

        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)
    plt.title('t_SNE plot of test images using hidden variables;k: '+str(Nsteps)+'; n: '+str(Nhidden))
    plt.xlabel('t_SNE_feature2')
    plt.ylabel('t_SNE_feature1')
    f.savefig('/content/drive/My Drive/t_SNE_k'+str(Nsteps)+'_n'+str(Nhidden)+'.png',dpi=300)
    
    return f, ax, sc, txts
test2 = pd.read_csv("test.csv")
fashion_scatter(tsne_results, test2.iloc[:,-1].values)        
