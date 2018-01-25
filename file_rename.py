																																																																																																																																																																																																						#Code for converting names of file in a order for training the neural network
import os,sys

count = 1
all = os.listdir(".")
for one in all:
	if one.startswith("img_"):
		os.rename(one,str(count)+".jpg")
		count+=1
	else:
		pass
