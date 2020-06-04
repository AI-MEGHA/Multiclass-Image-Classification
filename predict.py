#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


# # Initialising the CNN

# In[2]:


classifier = Sequential()


# # Step 1 - Convolution
# 

# In[3]:


classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))


# In[4]:


classifier.add(MaxPooling2D(pool_size = (2, 2)))


# # Adding a second convolutional layer

# In[5]:


classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))


# # 3RD Layer

# In[6]:


classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))


# In[7]:


classifier.add(Flatten())


# In[8]:



classifier.add(Dense(units = 10, activation = 'relu'))
classifier.add(Dense(units =5, activation = 'sigmoid'))


# In[9]:



classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# In[10]:



from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)


# In[11]:


training_set = train_datagen.flow_from_directory("C:/Users/ADMIN/Downloads/Family",
                                                 target_size = (64, 64),
                                                 batch_size = 2,
                                                 class_mode = 'categorical')


# In[12]:


test_set = test_datagen.flow_from_directory('C:/Users/ADMIN/Downloads/Family',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'categorical')


# In[13]:



classifier.fit_generator(training_set,
                         steps_per_epoch = 2000,
                         epochs =1,
                         validation_data = test_set,
                         validation_steps = 2000)


# In[14]:


from keras.models import Sequential
classifier.summary()


# In[15]:


classifier.save("model.h55")
print("Saved model to disk")


# In[16]:

# MEGHA
import numpy as np
from keras.preprocessing import image

test_image = image.load_img("C:/Users/ADMIN/Downloads/Family/MEGHA/WhatsApp Image 2020-06-03 at 9.40.32 AM (5).jpeg",
                            target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'MEGHA'
    print(prediction)
elif result[0][1] == 1:
    prediction = 'SANCHITA'
    print(prediction)
elif result[0][2] == 1:
    prediction = 'SPARSH'
    print(prediction)
elif result[0][3]:
    prediction = 'YOGESH'
    print(prediction)

# In[17]:


class fam:
    def __init__(self,filename):
        self.filename =filename


    def family(self):
        # load model
        model = load_model('model.h5')

        # summarize model
        #model.summary()
        imagename = self.filename
        test_image = image.load_img(imagename, target_size = (64, 64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = model.predict(test_image)

        if result[0][0] == 1:
            prediction = 'MEGHA'
            print(prediction)
        elif result[0][1] == 1:
            prediction = 'SANCHITA'
            print(prediction)
        elif result[0][2] == 1:
            prediction = 'SPARSH'
            print(prediction)
        elif result[0][3]:
            prediction = 'YOGESH'
            print(prediction)



