	
# coding: utf-8

# In[1]:


#Disclaimer of the project
'''A Neural network for training images in the MNIST dataset for number recognition'''


# In[2]:


#Imports needed
import numpy as np
import cv2
import os,sys

#Code to counter overflow in lambda operator
def activation(inputa):
    shapes = inputa.shape
    for i in range(0,shapes[1]):
        inputa[0,i] = float(1)/(float(1)+np.exp(-1*inputa[0,i]))
    return inputa
# In[3]:


#Class for creating the neural network
class Neural:
    def __init__(self,lr,wh1i,wh2h1,woh2):
        self.lr = lr
        self.wh1i = wh1i
        self.wh2h1 = wh2h1
        self.woh2 = woh2
    def train(self,image,result):
        #Training the neural net
        inputh1 = np.dot(image,(self.wh1i).T)
        inputh1 = inputh1/np.amax(inputh1)
        outputh1 = activation(inputh1)
        inputh2 = np.dot(outputh1,(self.wh2h1).T)
        inputh2 = inputh2/np.amax(inputh2)
        outputh2 = activation(inputh2)
        inputo = np.dot(outputh2,(self.woh2).T)
        inputo = inputo/np.amax(inputo)
        outputo = activation(inputo)
        #Calculating errors and back propagation
        error = result - outputo
        erroh2 = np.dot(error,self.woh2)
        errh2h1 = np.dot(erroh2,self.wh2h1)
        errh1i = np.dot(errh2h1,self.wh1i)
        #Updating the weights of the network along with managing the weights to be confined to 0-1
        self.wh1i+=self.lr*errh1i*np.dot((outputh1*(np.ones([1,100],dtype = np.int)-outputh1)).T,image)
        #Dividing the weights with the largest number in it
        #If this fails comment these weight lines
        #self.wh1i = self.wh1i/max(self.wh1i)
        self.wh2h1+= self.lr*errh2h1*np.dot((outputh2*(np.ones([1,50],dtype = np.int)-outputh2)).T,inputh1)
        #self.wh2h1 = self.wh2h1/max(self.wh2h1)
        self.woh2+= self.lr*erroh2*np.dot((outputo*(np.ones([1,10],dtype = np.int)-outputo)).T,inputh2)
        #self.woh2 = self.woh2/max(self.woh2)
        np.save("wh1i.npy",self.wh1i)
        np.save("wh2h1.npy",self.wh2h1)
        np.save("woh2.npy",self.woh2)

    def test(self,image):
        inputh1 = np.dot(image,(self.wh1i).T)
	inputh1 = inputh1/10000
        outputh1 = activation(inputh1)
        inputh2 = np.dot(outputh1,(self.wh2h1).T)
	inputh2 = inputh2/1000
        outputh2 = activation(inputh2)
        inputo = np.dot(outputh2,(self.woh2).T)
	inputo = inputo/10
        outputo = activation(inputo)
        print outputo


# In[4]:


#Code for controlling the neural net for training and testing
print "Enter your choices from below:"
print "1)Train the network with images"
print "2)Test the network with an image"
choice = input("")
flag=0
#Getting weights from the data file
lis = os.listdir('./')
for one in lis:
    if one == 'wh1i.npy':
        flag=1
    else:
        pass
if flag==1:
    wh1i = np.load('wh1i.npy')
    wh2h1 = np.load('wh2h1.npy')
    woh2 = np.load('woh2.npy')
elif flag==0:
    execfile("init.py")
network = Neural(0.01,wh1i,wh2h1,woh2)
if choice == 1:
    #Code for training all images in MNIST Directory
    count = 0
    for count in range(0,10,1):
        result = np.zeros((1,10),int)
        result[0,count] = 1
        for i in range(1,61,1):
            image = np.array(cv2.imread("./Training/"+str(count)+"/"+str(i)+".jpg"),int)
            image = image[:,:,0]
            #Converting the image to 1*784
            image = np.reshape(image,(1,784),order = 'C')
            network.train(image,result)
    print "Completed training your model"
elif choice == 2:
    #Code for testing a particular image in the directory
    for i in range(1,351,1):
        image = np.array(cv2.imread("./Testing/"+str(i)+".jpg"),int)
        image = image[:,:,0]
        image = np.reshape(image,(1,784),order = 'C')
        network.test(image)
else:
    exit(0)
