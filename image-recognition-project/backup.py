# %% [markdown]
# # Wstęp
# 
# Jako projekt wybrałem klasyfikację na bazie danych obrazkowej, na podstawie zdjęć samolotów militarnych z bazy danych z kaggle pod tytułem "Military Aircraft Detection Dataset". W tym zadaniu staram się wytrenować algorytm do przydzielenia odpowiedniej nazwy do samolotu.
# 

# %%
import cv2
import pandas as pd
import numpy as np
import os
import pathlib
import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory

# %% [markdown]
# ## Baza Danych i Preprocessing
# 
# https://www.kaggle.com/datasets/a2015003713/militaryaircraftdetectiondataset/data
# 
# Zbiór danych zawiera 49 różnych typów pojazdów lotniczych, z niektórymi skupionymi w jednej klasie wraz z ich podtypami. Zdjęć jest łącznie 23654.
# Poniżej dokonuję podziału na datasety treningowe i walidacyjne z tasowaniem. Jednocześnie konwertuję też obrazy na rozmiar 128 x 128 pikseli dla ułatwienia uczenia.

# %%
path = pathlib.Path("./crop/")

train_df = image_dataset_from_directory(path,
                                        image_size = (128, 128),
                                        validation_split = 0.3,
                                        subset = "training",
                                        shuffle = True,
                                        batch_size = 50,
                                        seed = 278)

validation_df = image_dataset_from_directory(path,
                                             image_size = (128, 128),
                                             validation_split = 0.35,
                                             subset = "validation",
                                             shuffle = True,                                         
                                             batch_size = 50,
                                             seed = 278)



# %%
print("Nazwy klas: " + str(train_df.class_names))
print("Długość zbioru treningowego: " + str(len(train_df)))
print("Długość zbioru walidacyjnego: " + str(len(validation_df)))

# %%
classes = train_df.class_names
test_df = validation_df.take(tf.data.experimental.cardinality(validation_df) // 5)

validation_df = validation_df.skip(tf.data.experimental.cardinality(validation_df) // 5)

test_df, validation_df

# %%
import matplotlib.pyplot as plt

# %%
class_names = train_df.class_names

images, labels = next(iter(train_df))

fig, axs = plt.subplots(5, 5, figsize=(20, 20))

for i, ax in enumerate(axs.flatten()):
    ax.imshow(images[i].numpy().astype("uint8"))
    ax.set_title(class_names[labels[i]])
    ax.axis('off')

plt.show()

# %% [markdown]
# Poniżej używam metody <span style="color:orange">prefetch</span> od Tensorflow aby przyspieszyć proces ładowania danych poprzez wczytywanie kolejnej partii danych w tle, podczas gdy trwa trening modelu na bieżącej partii. AUTOTUNE oznacza że Tensorflow dynamicznie dostosowuje liczbę partii do wczytania.

# %%
autotune = tf.data.AUTOTUNE
pf_train = train_df.prefetch(buffer_size = autotune)
pf_test = test_df.prefetch(buffer_size = autotune)
pf_val = validation_df.prefetch(buffer_size = autotune)

# %%
def plot_training(hist):
    tr_acc = hist.history['accuracy']
    tr_loss = hist.history['loss']
    val_acc = hist.history['val_accuracy']
    val_loss = hist.history['val_loss']
    best_loss_epoch = np.argmin(val_loss) + 1
    best_acc_epoch = np.argmax(val_acc) + 1
    epochs = range(1, len(tr_acc) + 1)

    plt.figure(figsize=(20, 8))
    plt.style.use('ggplot')

    plt.subplot(1, 2, 1)
    plt.plot(epochs, tr_loss, 'purple', label='Training loss')
    plt.plot(epochs, val_loss, 'orange', label='Validation loss')
    plt.scatter(best_loss_epoch, val_loss[best_loss_epoch - 1], s=150, c='blue', label=f'Best epoch= {best_loss_epoch}')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, tr_acc, 'purple', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'orange', label='Validation Accuracy')
    plt.scatter(best_acc_epoch, val_acc[best_acc_epoch - 1], s=150, c='blue', label=f'Best epoch= {best_acc_epoch}')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()


# %%
from tensorflow.keras import layers, applications

data_augmentation = tf.keras.Sequential()
data_augmentation.add(layers.RandomRotation(0.3))
data_augmentation.add(layers.RandomFlip("horizontal_and_vertical"))

image_size = (128, 128)
image_shape = image_size + (3,)

preprocess_input = applications.resnet50.preprocess_input
base_model = applications.ResNet50(input_shape = image_shape, include_top = False, weights = 'imagenet')

# %%
base_model.trainable = False
base_model.summary()

# %%
nclass = len(class_names)
global_avg = layers.GlobalAveragePooling2D()
output_layer = layers.Dense(nclass, activation = 'softmax', kernel_regularizer=tf.keras.regularizers.l2(0.01))
dropout_layer = layers.Dropout(0.5)

inputs = tf.keras.Input(shape = image_shape)
x = data_augmentation(inputs)
x = preprocess_input(inputs)
x = base_model(x)
x = global_avg(x)
x = dropout_layer(x)
outputs = output_layer(x)

model = tf.keras.Model(inputs = inputs, outputs = outputs)
model.summary()

# %%
from tensorflow.keras import losses, optimizers, callbacks

base_model.trainable = True
for layer in base_model.layers[:100]:
    layer.trainable = False

early_stopping = callbacks.EarlyStopping(
    min_delta=0.001,  # minimium amount of change to count as an improvement
    patience=20,  # how many epochs to wait before stopping
    restore_best_weights=True,
)
optimizer = optimizers.Adamax(learning_rate = optimizers.schedules.CosineDecay(0.001, 500))
loss = losses.SparseCategoricalCrossentropy()
model.compile(optimizer = optimizer, loss = loss, metrics = ['accuracy'])

# %%
history = model.fit(pf_train, validation_data = (pf_val), epochs = 5, callbacks=[early_stopping])

# %%
plot_training(history)

# %%
ts_length = len(test_df)
test_batch_size = test_batch_size = max(sorted([ts_length // n for n in range(1, ts_length + 1) if ts_length%n == 0 and ts_length/n <= 80]))
test_steps = ts_length // test_batch_size

train_score = model.evaluate(pf_train, steps= test_steps, verbose= 1)
valid_score = model.evaluate(pf_val, steps= test_steps, verbose= 1)
test_score = model.evaluate(pf_test, steps= test_steps, verbose= 1)

print("Train Loss: ", train_score[0])
print("Train Accuracy: ", train_score[1])
print('-' * 20)
print("Validation Loss: ", valid_score[0])
print("Validation Accuracy: ", valid_score[1])
print('-' * 20)
print("Test Loss: ", test_score[0])
print("Test Accuracy: ", test_score[1])

# %%
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Twój kod
image_batch, label_batch = pf_test.as_numpy_iterator().next()
pred_labels = np.argmax(model.predict(image_batch), axis = 1)

# Mapowanie indeksów na nazwy klas
label_names = [classes[i] for i in label_batch]
pred_names = [classes[i] for i in pred_labels]

lab_and_pred = np.transpose(np.vstack((label_names, pred_names)))

print(lab_and_pred)

# Utworzenie macierzy pomyłek
cm = confusion_matrix(label_batch, pred_labels)

# Wizualizacja macierzy pomyłek za pomocą Seaborn
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.title('ResNet50 - Confusion Matrix')
plt.show()

# %% [markdown]
# ### Testing with EfficientNetB3

# %%
img_shape = (128, 128, 3)

base_model = tf.keras.applications.efficientnet.EfficientNetB3(include_top= False, weights= "imagenet", input_shape= img_shape, pooling= 'max')

# %%
data_efficient = tf.keras.Sequential()
data_efficient.add(layers.RandomRotation(0.3))
data_efficient.add(layers.RandomFlip("horizontal_and_vertical"))

preprocess_input = applications.efficientnet.preprocess_input
base_model = applications.EfficientNetB3(input_shape = image_shape, include_top = False, weights = 'imagenet')

# %%
base_model.trainable = False
base_model.summary()

# %%
from tensorflow.keras import regularizers

global_avg = layers.GlobalAveragePooling2D()
batch_norm = layers.BatchNormalization(axis= -1, momentum= 0.99, epsilon= 0.001)
dense = layers.Dense(256, kernel_regularizer= regularizers.l2(l= 0.016), activity_regularizer= regularizers.l1(0.006),
                bias_regularizer= regularizers.l1(0.006), activation= 'relu')
dropout = layers.Dropout(rate= 0.45, seed= 123)
output_layer = layers.Dense(nclass, activation = 'softmax')


inputs = tf.keras.Input(shape = image_shape)
x = data_efficient(inputs)
x = preprocess_input(inputs)
x = base_model(x)
x = global_avg(x)
x = batch_norm(x)
x = dense(x)
x = dropout(x)
outputs = output_layer(x)

model = tf.keras.Model(inputs = inputs, outputs = outputs)
model.summary()
optimizer = optimizers.Adamax(learning_rate = 0.001)
model.compile(optimizer = optimizer, loss = loss, metrics = ['accuracy'])

# %%
base_model.trainable = True
for layer in base_model.layers[:100]:
    layer.trainable = False

# %%
loss = losses.SparseCategoricalCrossentropy()

# %%
optimizer = optimizers.RMSprop(learning_rate = optimizers.schedules.CosineDecay(0.001, 500))
model.compile(optimizer = optimizer, loss = loss, metrics = ['accuracy'])

# %%
history_fine = model.fit(pf_train, validation_data = (pf_val), epochs = 5)

# %%
plot_training(history_fine)

# %%
ts_length = len(test_df)
test_batch_size = test_batch_size = max(sorted([ts_length // n for n in range(1, ts_length + 1) if ts_length%n == 0 and ts_length/n <= 80]))
test_steps = ts_length // test_batch_size

train_score = model.evaluate(pf_train, steps= test_steps, verbose= 1)
valid_score = model.evaluate(pf_val, steps= test_steps, verbose= 1)
test_score = model.evaluate(pf_test, steps= test_steps, verbose= 1)

print("Train Loss: ", train_score[0])
print("Train Accuracy: ", train_score[1])
print('-' * 20)
print("Validation Loss: ", valid_score[0])
print("Validation Accuracy: ", valid_score[1])
print('-' * 20)
print("Test Loss: ", test_score[0])
print("Test Accuracy: ", test_score[1])

# %%
preds = model.predict(pf_test)
y_pred = np.argmax(preds, axis=1)
print(y_pred)



# %%
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Twój kod
image_batch, label_batch = pf_test.as_numpy_iterator().next()
pred_labels = np.argmax(model.predict(image_batch), axis = 1)

# Mapowanie indeksów na nazwy klas
label_names = [classes[i] for i in label_batch]
pred_names = [classes[i] for i in pred_labels]

lab_and_pred = np.transpose(np.vstack((label_names, pred_names)))

print(lab_and_pred)

# Utworzenie macierzy pomyłek
cm = confusion_matrix(label_batch, pred_labels)

# Wizualizacja macierzy pomyłek za pomocą Seaborn
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.title('EfficientNetB3 - Confusion Matrix')
plt.show()

# %%
base_model = tf.keras.applications.Xception(include_top= False, weights= "imagenet", input_shape= img_shape, pooling= 'max')
data_efficient = tf.keras.Sequential()
data_efficient.add(layers.RandomRotation(0.3))
data_efficient.add(layers.RandomFlip("horizontal_and_vertical"))

preprocess_input = applications.xception.preprocess_input
base_model = applications.Xception(input_shape = image_shape, include_top = False, weights = 'imagenet')
base_model.trainable = False

from tensorflow.keras import regularizers

global_avg = layers.GlobalAveragePooling2D()
batch_norm = layers.BatchNormalization(axis= -1, momentum= 0.99, epsilon= 0.001)
dense = layers.Dense(256, kernel_regularizer= regularizers.l2(l= 0.016), activity_regularizer= regularizers.l1(0.006),
                bias_regularizer= regularizers.l1(0.006), activation= 'relu')
dropout = layers.Dropout(rate= 0.45, seed= 123)
output_layer = layers.Dense(nclass, activation = 'softmax')


inputs = tf.keras.Input(shape = image_shape)
x = data_efficient(inputs)
x = preprocess_input(inputs)
x = base_model(x)
x = global_avg(x)
x = batch_norm(x)
x = dense(x)
x = dropout(x)
outputs = output_layer(x)

model = tf.keras.Model(inputs = inputs, outputs = outputs)
model.summary()
optimizer = optimizers.Adamax(learning_rate = 0.001)
model.compile(optimizer = optimizer, loss = loss, metrics = ['accuracy'])

base_model.trainable = True
for layer in base_model.layers[:100]:
    layer.trainable = False
    
loss = losses.SparseCategoricalCrossentropy()

optimizer = optimizers.RMSprop(learning_rate = optimizers.schedules.CosineDecay(0.001, 500))
model.compile(optimizer = optimizer, loss = loss, metrics = ['accuracy'])


# %%
history_xception = model.fit(pf_train, validation_data = (pf_val), epochs = 5)

# %%
plot_training(history_xception)

# %%
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

image_batch, label_batch = pf_test.as_numpy_iterator().next()
pred_labels = np.argmax(model.predict(image_batch), axis = 1)
label_names = [classes[i] for i in label_batch]
pred_names = [classes[i] for i in pred_labels]

lab_and_pred = np.transpose(np.vstack((label_names, pred_names)))

print(lab_and_pred)

cm = confusion_matrix(label_batch, pred_labels)

plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.title('Xception - Confusion Matrix')
plt.show()

# %% [markdown]
# # GAN

# %%
import os
import cv2
import numpy as np


folder_path = "crop"

airplane_images = []

for subdir in os.listdir(folder_path):
    subdir_path = os.path.join(folder_path, subdir)
    if os.path.isdir(subdir_path):  

        for filename in os.listdir(subdir_path):

            image = cv2.imread(os.path.join(subdir_path, filename))

            resized_image = cv2.resize(image, (28, 28))

            normalized_image = resized_image / 255.0

            airplane_images.append(normalized_image)


airplane_images = np.array(airplane_images)

print("Kształt tensora obrazów samolotów:", airplane_images.shape)

# %%
import numpy as np
from skimage import color

# Konwersja obrazów samolotów do skali szarości (jeśli nie są)
airplane_images_gray = np.array([color.rgb2gray(img) for img in airplane_images])

# Rozszerzenie wymiarów o 1 (dla kanału koloru)
airplane_images_gray = np.expand_dims(airplane_images_gray, axis=-1)

# Przekształcenie kształtu obrazów do (batch_size, 28, 28, 1)
airplane_images_resized = tf.image.resize(airplane_images_gray, (28, 28))

# Normalizacja pikseli do zakresu [-1, 1]
airplane_images_normalized = airplane_images_resized / 127.5 - 1.0

# Trenowanie modelu GAN
train_gan(generator, discriminator, gan, airplane_images_normalized, epochs, batch_size, noise_dim)

# %%
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

# Definicja generatora
def build_generator(input_shape):
    model = models.Sequential([
        layers.Dense(256, input_shape=input_shape, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(1024, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(784, activation='tanh'),  # Wielkość obrazu (28x28 = 784 piksele)
        layers.Reshape((28, 28, 1))  # Przekształcenie wektora w obraz
    ])
    return model

# Definicja dyskryminatora
def build_discriminator(input_shape):
    model = models.Sequential([
        layers.Flatten(input_shape=input_shape),  # Spłaszczenie obrazu do wektora
        layers.Dense(512, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # Wyjście binarne (0 - fałszywy, 1 - prawdziwy)
    ])
    return model

# Wymiary wejściowe dla generatora i dyskryminatora (np. 28x28 obrazy w odcieniach szarości)
input_shape = (100,)  # Generator przyjmuje wektor szumu o wymiarze 100

# Zbudowanie i kompilacja generatora
generator = build_generator(input_shape)
generator_optimizer = optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
generator.compile(loss='binary_crossentropy', optimizer=generator_optimizer)

# Zbudowanie i kompilacja dyskryminatora
discriminator = build_discriminator((28, 28, 1))  # Dyskryminator przyjmuje obrazy 28x28x1
discriminator_optimizer = optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
discriminator.compile(loss='binary_crossentropy', optimizer=discriminator_optimizer, metrics=['accuracy'])

# Zbudowanie modelu GAN
z = layers.Input(shape=input_shape)
generated_image = generator(z)
discriminator.trainable = False  # Wyłączenie trenowania dyskryminatora podczas trenowania modelu GAN
validity = discriminator(generated_image)
gan = models.Model(z, validity)
gan_optimizer = optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
gan.compile(loss='binary_crossentropy', optimizer=gan_optimizer)

# Trenowanie modelu GAN
import numpy as np

# Funkcja do generowania szumu jako dane wejściowe dla generatora
def generate_noise(batch_size, noise_dim):
    return np.random.normal(0, 1, (batch_size, noise_dim))

# Funkcja do treningu modelu GAN
def train_gan(generator, discriminator, gan, X_train, epochs, batch_size, noise_dim):
    for epoch in range(epochs):
        # Wybór losowych próbek z rzeczywistego zbioru danych
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        real_images = X_train[idx]
        
        # Generowanie szumu
        noise = generate_noise(batch_size, noise_dim)
        
        # Generowanie fałszywych obrazów przy użyciu generatora
        generated_images = generator.predict(noise)
        
        # Trenowanie dyskryminatora na rzeczywistych i fałszywych danych
        d_loss_real = discriminator.train_on_batch(real_images, np.ones((batch_size, 1)))
        d_loss_fake = discriminator.train_on_batch(generated_images, np.zeros((batch_size, 1)))
        
        # Obliczanie średniej straty dla dyskryminatora
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        # Trenowanie modelu GAN (generatora)
        noise = generate_noise(batch_size, noise_dim)
        g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))
        
        # Wyświetlanie postępu trenowania co 100 epok
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, D Loss: {d_loss[0]}, G Loss: {g_loss}")


# Parametry trenowania
epochs = 10000
batch_size = 64
noise_dim = 100

# Trenowanie modelu GAN
train_gan(generator, discriminator, gan, airplane_images, epochs, batch_size, noise_dim)


