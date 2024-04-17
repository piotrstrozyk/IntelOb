
import os
# counting the number of files in train folder
path, dirs, files = next(os.walk('./images'))
file_count = len(files)
print('Number of images: ', file_count)



import numpy as np
from PIL import Image
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
import cv2

# display dog image
img = mpimg.imread('./images/dog.8298.jpg')
imgplt = plt.imshow(img)
plt.show()

# %%
# display cat image
img = mpimg.imread('./images/cat.4352.jpg')
imgplt = plt.imshow(img)
plt.show()

# %%
# original_folder = './images/'
# resized_folder = './resized/'

# for i in range(25000):

#   filename = os.listdir(original_folder)[i]
#   img_path = original_folder+filename

#   img = Image.open(img_path)
#   img = img.resize((224, 224))
#   img = img.convert('RGB')

#   newImgPath = resized_folder+filename
#   img.save(newImgPath)

# %%
# display cat image
img = mpimg.imread('./resized/cat.4352.jpg')
imgplt = plt.imshow(img)
plt.show()

# %%
# creaing a for loop to assign labels
filenames = os.listdir('./resized/')


labels = []

for i in range(25000):

  file_name = filenames[i]
  label = file_name[0:3]

  if label == 'dog':
    labels.append(1)

  else:
    labels.append(0)

# %%
print(filenames[0:5])
print(len(filenames))

# %%
print(labels)
print(len(labels))

# %%
# counting the images of dogs and cats out of 25000 images
values, counts = np.unique(labels, return_counts=True)
print(values)
print(counts)

# %%
import glob
image_directory = './resized/'
image_extension = ['png', 'jpg']

files = []

[files.extend(glob.glob(image_directory + '*.' + e)) for e in image_extension]

dog_cat_images = np.asarray([cv2.imread(file) for file in files])


# %%
print(dog_cat_images)

# %%
type(dog_cat_images)

# %%
print(dog_cat_images.shape)

# %%
X = dog_cat_images
Y = np.asarray(labels)

# %%
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# %%
print(X.shape, X_train.shape, X_test.shape)

# %% [markdown]
# 
# 
# 20000 --> training images
# 
# 5000 --> test images
# 

# %%
# scaling the data
X_train_scaled = X_train/255

X_test_scaled = X_test/255

# %%
print(X_train_scaled)

# %%
import tensorflow as tf
import tensorflow_hub as hub


# %%
mobilenet_model = 'https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4'

pretrained_model = hub.KerasLayer(mobilenet_model, input_shape=(224,224,3), trainable=False)

# %%
num_of_classes = 2

model = tf.keras.Sequential([
    
    pretrained_model,
    tf.keras.layers.Dense(num_of_classes)

])

model.summary()


