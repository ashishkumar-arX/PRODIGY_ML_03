import os 
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import random

# CONVERTING IMAGES TO AN OBJECTS FIILE 
dir = 'task3/PetImages'
categories = ['Cat','Dog']
data = []

for category in categories:
    path = os.path.join(dir,category)
    label = categories.index(category)

    for img in os.listdir(path):
        imgpath = os.path.join(path,img)
        pet_img = cv2.imread(imgpath,0)
        try:
            pet_img = cv2.resize(pet_img,(50,50))
            image = np.array(pet_img).flatten()

            data.append([image,label])
        except Exception as e:
            pass

print(len(data))

pick_in = open('task3/data.pickle','wb')
pickle.dump(data,pick_in)
pick_in.close()

##############################################################################
# BUILDING MODEL ON THE ABOVE DATA
pick_in = open('task3/data.pickle','rb')
data = pickle.load(pick_in)
pick_in.close()

# shuffling the data
random.shuffle(data)
features = []
labels = []

for feature, label in data:
    features.append(feature)
    labels.append(label)
print("features and labels done !")

# splitting the dataset in to train and test data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(features,labels,test_size=0.5)
print("splliting done ! ")

# ##############################################################################
# creating the model
from sklearn.svm import SVC
model = SVC(C=1,kernel='poly',gamma='auto')
model.fit(x_train,y_train)

# saving the model as model.sav file
pick = open('task3/model.sav','wb')
pickle.dump(model,pick)
pick.close()

# ##############################################################################
# MAKING PREDICTIONS
pick = open('task3/model.sav','rb')
model = pickle.load(pick)
pick.close()

prediction = model.predict(x_test)
accuracy = model.score(x_test,y_test)
categories = ['Cat','Dog']

print('Accuracy: ', accuracy)

print('Prediction is : ', categories[prediction[0]])

mypet = x_test[0].reshape(50,50)
plt.imshow(mypet,cmap='gray')
plt.show()
