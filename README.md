# Pneumonia Detection Based on Chest X-Ray Images
# Automated Diagnosis with Convolutional Neural Networks (CNN) Classification
Pengujian deep learning dengan menggunakan Convolutional Neural Networks (CNN) untuk mengklasifikasikan gambar Chest X-Ray pada kelompok normal atau terdapat penyakit pneunomia.
Kernel ini menggunakan dataset Chest X-Ray Images (Pneumonia), yang tersusun dalam 3 folder (train, test, val) dan berisi subfolder untuk setiap kategori gambar (Pneumonia / Normal). Ada 5.863 gambar X-Ray (JPEG) dan 2 kategori (Pneumonia / Normal)

## Pendahuluan
Pneumonia adalah peradangan paru-paru yang disebabkan oleh infeksi. Beberapa gejala yang umumnya dialami penderita pneumonia adalah batuk berdahak, demam, dan sesak napas. Pada kondisi pneunomia (paru-paru basah), infeksi menyebabkan peradangan pada kantong-kantong udara (alveoli) di salah satu atau kedua paru-paru. Akibatnya, alveoli bisa dipenuhi cairan atau nanah sehingga menyebabkan penderitanya sulit bernapas.

Pneumonia merupakan salah satu penyebab kematian tertinggi pada anak-anak di seluruh dunia. Badan Kesehatan Dunia (WHO) memperkirakan bahwa 15% kematian anak-anak berusia di bawah 5 tahun disebabkan oleh penyakit ini. WHO juga menyatakan bahwa pada tahun 2017, terdapat lebih dari 800.000 anak-anak meninggal akibat pneumonia. Sayangnya, pneunomia ini sering terjadi di negara berkembang dibandingkan negara maju, petugas klinis yang terlatih untuk menafsirkan foto rontgen dada ini sering kali kurang.

Pada kesempatan kali ini akan dilakukan pengujian deep learning dengan menggunakan Convolutional Neural Networks (CNN) untuk mengklasifikasikan gambar Chest X-Ray pada kelompok normal atau terdapat penyakit pneunomia. Dalam kernel ini akan menggunakan Convolutional Neural Networks yang hampir sama dengan yang ditampilkan pada paper Efficient Deep Network Architectures for Fast Chest X-Ray Tuberculosis Screening and Visualization, penting untuk diingat bahwa kernel dan model ini juga tidak disarankan untuk diagnosis klinis karena hanya diperuntukkan sebagai projek akhir Bootcamp Indocyber dan studi kasus pribadi.

![Alt paper](https://user-images.githubusercontent.com/72612848/96394017-1a3b1b00-11eb-11eb-9354-e0e14f313035.png)

## Persiapan Data

Dataset yang digunakan adalah dataset yang tersedia pada kaggle dengan link sebagai berikut: https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia
untuk memudahkan dalam menggunakan data, pada kesempatan ini data langsung diambil pada kaggle
```
!pip install kaggle
from google.colab import files
files.upload()
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
!unzip chest-xray-pneumonia.zip
```

melakukan import yang diperlukan, selain itu Salah satu praktik terbaik yang harus dilakukan saat melakukan project machine learning adalah dengan menentukan konstanta, sehingga memfasilitasi perubahan lebih lanjut. Mengingat itu, perlu dilakukan penentuan nilai bacth size, tinggi dan lebar gambar, dan learning rate.

```
import os
import cv2
import seaborn as sns
import numpy as np 
import matplotlib.pyplot as plt
import tensorflow as tf
import datetime
from PIL import Image
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras.layers import add
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

np.random.seed(777)
tf.random.set_seed(777)
```

```
BATCH_SIZE = 32
IMG_HEIGHT = 240
IMG_WIDTH = 240
ALPHA = 1e-4
```

## Load Data
Kernel ini menggunakan dataset Chest X-Ray Images (Pneumonia), yang disusun menjadi 3 folder (train, test, val) dan berisi subfolder untuk setiap kategori gambar (Pneumonia / Normal). Ada 5.863 gambar X-Ray (JPEG) dan 2 kategori (Pneumonia / Normal).

Untuk analisis gambar Chest X-Ray, semua radiografi dada pada awalnya diskrining untuk kontrol kualitas dengan menghapus semua pindaian berkualitas rendah atau tidak terbaca.
```
data_dir = '/content/chest_xray'
```
```
labels = ['NORMAL','PNEUMONIA']
def get_data(data_dir):
    data = [] 
    for label in labels: 
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE) 
                resized_arr = cv2.resize(img_arr, (IMG_WIDTH, IMG_HEIGHT))
                data.append([resized_arr, class_num])
            except Exception as e:
                pass

    return np.array(data)
 ```
 
kemudian definisikan sebuah fungsi untuk mengembalikan np.array dengan semua gambar yang terletak di direktori tertentu dan menggunakannya untuk memuat data training, validation dan test data
```
train = get_data('/content/chest_xray/train/')
test = get_data('/content/chest_xray/test/')
val = get_data('/content/chest_xray/val/')
```

kemudian perlu juga untuk melihat berapa banyak gambar yang dimiliki dari setiap class di training set. Selain itu, mari kita lihat bagaimana gambar didistribusikan diantara training, validation dan test data

```
print(f"{[y for _, y in train].count(0)} NORMAL IMAGES IN TRAINING SET")
print(f"{[y for _, y in train].count(1)} PNEUMONIA IMAGES IN TRAINING SET")
```
```
  1341 NORMAL IMAGES IN TRAINING SET
  3875 PNEUMONIA IMAGES IN TRAINING SET
```
```
print(f'Images in TRAINING SET: {train.shape[0]}')
print(f'Images in VALIDATION SET: {val.shape[0]}')
print(f'Images in TEST SET: {test.shape[0]}')
```
```
  Images in TRAINING SET: 5216
  Images in VALIDATION SET: 16
  Images in TEST SET: 624
```
Seperti yang dapat dilihat pada output dari dua cell sebelumnya, terdapat masalah data yang tidak seimbang dan dengan proporsi yang agak aneh antara training set dan validation set. masalah tersebut akan coba diatasi dalam tahap pemrosesan data, untuk saat ini yang dilakukan hanya akan menggabungkan dataset train dan val kemudian melakukan pemisahan (split) lagi.
```
train = np.append(train, val, axis=0)
train, val = train_test_split(train, test_size=.20, random_state=777)
```
Untuk mengakhiri bagian ini, akan ditampilkan beberapa contoh dalam dataset yang dimiliki.
```
plt.figure(figsize=(10, 10))
for k, i in np.ndenumerate(np.random.randint(train.shape[0], size=9)):
    ax = plt.subplot(3, 3, k[0] + 1)
    plt.imshow(train[i][0], cmap='gray')
    plt.title(labels[train[i][1]])
    plt.axis("off")
```
![Alt sample_data](https://user-images.githubusercontent.com/72612848/96394761-fc6eb580-11ec-11eb-9bc1-be2f713e9ce8.png)

## Processing Data
pada tahap awal ini akan membuat dan menggunakan fungsi yang disebut prepared_data () yang akan menormalkan gambar (membagi setiap piksel dengan 255) dan me-reshape array menjadi sesuai bentuk. Setelah itu, fungsi akan mengembalikan array x dan y secara terpisah dari dataset.
```
def prepare_data(data):
    x = []
    y = []
    
    for feature, label in data:
        x.append(feature)
        y.append(label)
        
    x = (np.array(x) / 255).reshape(-1,IMG_WIDTH, IMG_HEIGHT, 1)
    y = np.array(y)
        
    return x, y

x_train, y_train = prepare_data(train)
x_val, y_val = prepare_data(val)
x_test, y_test = prepare_data(test)
```
Untuk mencari performa terbaik dari model yang dilakukan, penting untuk menambah jumlah sampel dalam dataset dan untuk itu perlu melakukan proses data augumentasi. Perhatikan bahwa di sini tidak akan menggunakan flip untuk menghasilkan gambar baru karena paru-paru tidak simetris secara horizontal dan vertikal sehingga tidak diperlukan. kemudian juga ditampilkan hasil dari data augmentasi yang dilakukan.
```
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range = 20, 
    zoom_range = 0.2, 
    width_shift_range=0.15,  
    height_shift_range=0.15,
    horizontal_flip = False,  
    vertical_flip=False)


datagen.fit(x_train)

# pick an image to transform
image_path = '/content/NORMAL2-IM-1427-0001.jpeg'
img = image.load_img(image_path)


img=image.img_to_array(img)
img=img.reshape((1,) + img.shape)

i = 0

for batch in datagen.flow(img, save_prefix='test', save_format='jpeg'):  # this loops runs forever until we break, saving images to current directory with specified prefix
    plt.figure(i)
    plot = plt.imshow(image.img_to_array(batch[0]).astype(np.uint8))
    i += 1
    if i > 4: 
        break

plt.show()
```
Nah, untuk menyelesaikan masalah ketidakseimbangan data yang disebutkan sebelumnya. Ada beberapa kemungkinan pendekatan untuk diambil tetapi kami akan memilih untuk memberikan weight (bobot/pembebanan) yang berbeda ke kelas.

weight ini akan digunakan di pada proses lanjutan sebagai parameter yang sesuai dengan model, dan sebagaimana dijelaskan dalam dokumentasi resmi Keras, fungsi compute_class_weight() dapat berguna untuk memberi tahu model agar "lebih memperhatikan" sampel dari under-represented class.
```
weights = compute_class_weight('balanced', np.unique(y_train), y_train)
weights = {0: weights[0], 1: weights[1]}
print(weights)
```
```
  {0: 1.9339186691312384, 1: 0.6743474057363842}
```

## Membuat Model
Akhirnya sampai pada bagian yang paling menarik bagi sebagian orang: tahap pembuatan model. Seperti disebutkan di awal, kesempatan kali ini akan menggunakan CNN yang hampir sama dengan yang ada dari paper Efficient Deep Network Architectures for Fast Chest X-Ray Tuberculosis Screening and Visualization. Model yang disajikan memiliki arsitektur berikut:

dikatakan hampir sama karena tidak sepenuhnya sama karena model kali ini menggunakan aktivasi sigmoid (pada paper menggunakan aktivasi Softmax) di lapisan terakhir dari model. Namun secara umum tidak mengubah apa pun, karena fungsi Softmax adalah generalisasi fungsi sigmoid tetapi berlaku untuk masalah multilabel.

![Alt CNN](https://user-images.githubusercontent.com/72612848/96395000-a4847e80-11ed-11eb-9756-97076c7e6226.png)

```
def block(inputs, filters, stride):
    conv_0 = layers.Conv2D(filters=filters, kernel_size=(3, 3), strides=(stride, stride), padding='same', activation='relu')(inputs)
    conv_1 = layers.Conv2D(filters=filters, kernel_size=(3, 3), strides=(stride, stride), padding='same', activation='relu')(conv_0)
    
    skip = layers.Conv2D(input_shape=input_size, filters=filters, kernel_size=(1, 1), strides=(stride**2, stride**2), padding='same', activation='relu')(inputs)
    
    pool = layers.MaxPool2D(pool_size=(3, 3), strides=(2,2), padding='same')(add([conv_1, skip]))
    
    return pool
```
```
input_size = (IMG_HEIGHT, IMG_WIDTH, 1)

inputs = tf.keras.Input(shape=input_size, name='input')

y_0 = block(inputs, 16, 2)
y_1 = block(y_0, 32, 1)
y_2 = block(y_1, 48, 1)
y_3 = block(y_2, 64, 1)
y_4 = block(y_3, 80, 1)

gap = layers.GlobalMaxPooling2D()(y_4)
dense = layers.Dense(2, activation='relu')(gap)

outputs = layers.Dense(1, activation='sigmoid')(dense)
```
```
model = tf.keras.Model(inputs=inputs, outputs=outputs, name="pneumonia_model")
model.summary()
```
```
Model: "pneumonia_wnet"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input (InputLayer)              [(None, 240, 240, 1) 0                                            
__________________________________________________________________________________________________
conv2d_24 (Conv2D)              (None, 120, 120, 16) 160         input[0][0]                      
__________________________________________________________________________________________________
conv2d_25 (Conv2D)              (None, 60, 60, 16)   2320        conv2d_24[0][0]                  
__________________________________________________________________________________________________
conv2d_26 (Conv2D)              (None, 60, 60, 16)   32          input[0][0]                      
__________________________________________________________________________________________________
add_5 (Add)                     (None, 60, 60, 16)   0           conv2d_25[0][0]                  
                                                                 conv2d_26[0][0]                  
__________________________________________________________________________________________________
max_pooling2d_14 (MaxPooling2D) (None, 30, 30, 16)   0           add_5[0][0]                      
__________________________________________________________________________________________________
conv2d_27 (Conv2D)              (None, 30, 30, 32)   4640        max_pooling2d_14[0][0]           
__________________________________________________________________________________________________
conv2d_28 (Conv2D)              (None, 30, 30, 32)   9248        conv2d_27[0][0]                  
__________________________________________________________________________________________________
conv2d_29 (Conv2D)              (None, 30, 30, 32)   544         max_pooling2d_14[0][0]           
__________________________________________________________________________________________________
add_6 (Add)                     (None, 30, 30, 32)   0           conv2d_28[0][0]                  
                                                                 conv2d_29[0][0]                  
__________________________________________________________________________________________________
max_pooling2d_15 (MaxPooling2D) (None, 15, 15, 32)   0           add_6[0][0]                      
__________________________________________________________________________________________________
conv2d_30 (Conv2D)              (None, 15, 15, 48)   13872       max_pooling2d_15[0][0]           
__________________________________________________________________________________________________
conv2d_31 (Conv2D)              (None, 15, 15, 48)   20784       conv2d_30[0][0]                  
__________________________________________________________________________________________________
conv2d_32 (Conv2D)              (None, 15, 15, 48)   1584        max_pooling2d_15[0][0]           
__________________________________________________________________________________________________
add_7 (Add)                     (None, 15, 15, 48)   0           conv2d_31[0][0]                  
                                                                 conv2d_32[0][0]                  
__________________________________________________________________________________________________
max_pooling2d_16 (MaxPooling2D) (None, 8, 8, 48)     0           add_7[0][0]                      
__________________________________________________________________________________________________
conv2d_33 (Conv2D)              (None, 8, 8, 64)     27712       max_pooling2d_16[0][0]           
__________________________________________________________________________________________________
conv2d_34 (Conv2D)              (None, 8, 8, 64)     36928       conv2d_33[0][0]                  
__________________________________________________________________________________________________
conv2d_35 (Conv2D)              (None, 8, 8, 64)     3136        max_pooling2d_16[0][0]           
__________________________________________________________________________________________________
add_8 (Add)                     (None, 8, 8, 64)     0           conv2d_34[0][0]                  
                                                                 conv2d_35[0][0]                  
__________________________________________________________________________________________________
max_pooling2d_17 (MaxPooling2D) (None, 4, 4, 64)     0           add_8[0][0]                      
__________________________________________________________________________________________________
conv2d_36 (Conv2D)              (None, 4, 4, 80)     46160       max_pooling2d_17[0][0]           
__________________________________________________________________________________________________
conv2d_37 (Conv2D)              (None, 4, 4, 80)     57680       conv2d_36[0][0]                  
__________________________________________________________________________________________________
conv2d_38 (Conv2D)              (None, 4, 4, 80)     5200        max_pooling2d_17[0][0]           
__________________________________________________________________________________________________
add_9 (Add)                     (None, 4, 4, 80)     0           conv2d_37[0][0]                  
                                                                 conv2d_38[0][0]                  
__________________________________________________________________________________________________
max_pooling2d_18 (MaxPooling2D) (None, 2, 2, 80)     0           add_9[0][0]                      
__________________________________________________________________________________________________
global_max_pooling2d_1 (GlobalM (None, 80)           0           max_pooling2d_18[0][0]           
__________________________________________________________________________________________________
dense_8 (Dense)                 (None, 2)            162         global_max_pooling2d_1[0][0]     
__________________________________________________________________________________________________
dense_9 (Dense)                 (None, 1)            3           dense_8[0][0]                    
==================================================================================================
Total params: 230,165
Trainable params: 230,165
Non-trainable params: 0
__________________________________________________________________________________________________
```

```
# summarize filter shapes
for layer in model.layers:
	# check for convolutional layer
	if 'conv' not in layer.name:
		continue
  # get filter weights
	filters, biases = layer.get_weights()
	print(layer.name, filters.shape)
 ```
 ```
 filters, biases = model.layers[2].get_weights()
# Normalisasi nilai filter ke 0-1 untuk dapat divisualisasikan
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)
# plot beberapa filter pertama
n_filters, ix = 5, 1
for i in range(n_filters):
	# mendapatkan filter
	f = filters[:, :, :, i]
	# plot channel secara separately
	for j in range(3):
		# specify subplot
		ax = plt.subplot(n_filters, 3, ix)
		ax.set_xticks([])
		ax.set_yticks([])
		# plot filter channel dalam grayscale
		plt.imshow(f[:, :, j], cmap='gray')
		ix += 1
# Menampilkan gambar
plt.show()
```

## Melatih Model
dalam melatih model dilakukan pendefinisian beberapa callback seperti ReduceLROnPlateau dan EarlyStopping yang akan membantu kita mendapatkan pelatihan yang lebih cepat. Seperti yang didefinisikan di referensi Keras API, megurangi learning rate saat metrik berhenti meningkat, dan itu berguna karena model sering kali mendapatkan keuntungan dari pengurangan kecepatan pemelajaran sebesar faktor 2-10 setelah pembelajaran mandek. Callback ini memantau kuantitas dan jika tidak ada peningkatan yang terlihat untuk jumlah 'patience' epoch, kecepatan pembelajaran akan berkurang. Selain itu, penghentian awal adalah bentuk regularisasi dan dengan demikian akan membantu kami mencegah overfitting.

```
lr_reduce = ReduceLROnPlateau(monitor='val_acc', factor=0.7, min_delta=ALPHA, patience=7, verbose=1)
early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=7, restore_best_weights=True)
```
```
CALLBACKS = [lr_reduce, early_stopping_cb]
```

Untuk metrik, digunakan accuracy, precision, dan recall. Pilihan ini dibuat karena:

Accuracy, meskipun merupakan metrik yang paling banyak digunakan, memberikan informasi yang cukup tentang seberapa baik performa model, karena memiliki masalah data yang tidak seimbang.
Precision memberikan rasio tp / (tp + fp) atau secara intuitif menentukan kemampuan pengklasifikasi untuk tidak memberi label positif pada sampel yang negatif.
Recall memberikan rasio tp / (tp + fn) atau secara intuitif kemampuan pengklasifikasi untuk menemukan semua sampel positif.

```
METRICS = ['accuracy',
          tf.keras.metrics.Precision(name='precision'),
          tf.keras.metrics.Recall(name='recall')]
```
Optimizer yang digunakan adalah Adam, karena optimizer ini menggabungkan properti terbaik dari algoritma AdaGrad dan RMSProp untuk menyediakan algoritma pengoptimalan yang dapat menangani gradien renggang pada noisy. Untuk loss, hanya dua label yang memungkinkan, maka akan menggunakan binary crossentropy.
```
model.compile(
    optimizer=tf.keras.optimizers.Adam(lr=ALPHA),
    loss='binary_crossentropy', 
    metrics=METRICS)
```
Menampilkan visualisasi menggunakan Tensorboard pada setiap matrik
```
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
import os
import datetime
```
```
filepath="weight_pneunomia_best.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only= True, mode='max')
callbacks_list = [checkpoint]

logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
callbacks_list.append(TensorBoard(logdir,histogram_freq=1))
```
kemudian menyesuaikan model. Penting untuk diingat di sini untuk menggunakan weight untuk class yang yang sudah dihitung sebelumnya
```
history = model.fit(datagen.flow(x_train,y_train, batch_size = BATCH_SIZE),
                    steps_per_epoch=x_train.shape[0]/BATCH_SIZE, 
                    validation_data = (x_val, y_val),
                    validation_steps=x_val.shape[0]/BATCH_SIZE,
                    callbacks = callbacks_list,
                    class_weight = weights,
                    epochs = 30)
```

```
 Epoch 1/30
  1/130 [..............................] - ETA: 0s - loss: 0.3725 - accuracy: 0.8750 - precision: 0.9600 - recall: 0.8889WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow/python/ops/summary_ops_v2.py:1277: stop (from tensorflow.python.eager.profiler) is deprecated and will be removed after 2020-07-01.
Instructions for updating:
use `tf.profiler.experimental.stop` instead.
131/130 [==============================] - ETA: 0s - loss: 0.1651 - accuracy: 0.9379 - precision: 0.9827 - recall: 0.9326
Epoch 00001: val_accuracy improved from -inf to 0.94460, saving model to weight_pneunomia_best.h5
131/130 [==============================] - 16s 121ms/step - loss: 0.1651 - accuracy: 0.9379 - precision: 0.9827 - recall: 0.9326 - val_loss: 0.1334 - val_accuracy: 0.9446 - val_precision: 0.9878 - val_recall: 0.9372
Epoch 4/30
131/130 [==============================] - ETA: 0s - loss: 0.1645 - accuracy: 0.9331 - precision: 0.9799 - recall: 0.9288
Epoch 00004: val_accuracy improved from 0.94460 to 0.95129, saving model to weight_pneunomia_best.h5
Epoch 10/30
131/130 [==============================] - ETA: 0s - loss: 0.1431 - accuracy: 0.9431 - precision: 0.9835 - recall: 0.9391
Epoch 00010: val_accuracy improved from 0.95129 to 0.96753, saving model to weight_pneunomia_best.h5
Epoch 30/30
131/130 [==============================] - ETA: 0s - loss: 0.1386 - accuracy: 0.9446 - precision: 0.9828 - recall: 0.9417
Epoch 00030: val_accuracy did not improve from 0.96753
131/130 [==============================] - 15s 117ms/step - loss: 0.1386 - accuracy: 0.9446 - precision: 0.9828 - recall: 0.9417 - val_loss: 0.0894 - val_accuracy: 0.9628 - val_precision: 0.9869 - val_recall: 0.9628
```
visualisasi menggunakan tensorboard Salah satu hal terpenting yang harus dilakukan terkait pelatihan model adalah memvisualisasikan evolusi performanya. Dalam pengertian ini, akan memplot nilai akurasi, presisi, recall, dan AUC setiap epoch menggunakan tensorboard.
```
%load_ext tensorboard
%tensorboard --logdir logs
```
![Alt tensorboard](https://user-images.githubusercontent.com/72612848/96395498-fa0d5b00-11ee-11eb-9a95-2507e31010b2.png)

## Evaluasi Model
mencoba menghasilkan prediksi dari data yang belum pernah dilihat sebelumnya dan memeriksa performa model yang dimiliki.

```
print("Loss of the model is - " , model.evaluate(x_test,y_test)[0])
print("Accuracy of the model is - " , model.evaluate(x_test,y_test)[1]*100 , "%")
```
```
  20/20 [==============================] - 0s 7ms/step - loss: 0.2853 - accuracy: 0.9167 - precision: 0.8949 - recall: 0.9821
  Loss of the model is -  0.28529396653175354
  20/20 [==============================] - 0s 7ms/step - loss: 0.2853 - accuracy: 0.9167 - precision: 0.8949 - recall: 0.9821
  Accuracy of the model is -  91.66666865348816 %
```
Dengan nilai akurasi, presisi, dan f1-score ini, jelas bahwa model yang digunakan dalam kernel ini menampilkan dirinya sebagai arsitektur yang solid dan sangat efisien untuk masalah yang dimaksud.
```
predictions = model.predict(x_test)
predictions = predictions.reshape(1,-1)[0]
predictions[predictions >= 0.5] = 1
predictions[predictions < 0.5] = 0
```
```
print(classification_report(y_test, predictions, target_names = ['Normal (Class 0)','Pneumonia (Class 1)']))
```
```
                     precision    recall  f1-score   support

   Normal (Class 0)       0.96      0.81      0.88       234
Pneumonia (Class 1)       0.89      0.98      0.94       390

           accuracy                           0.92       624
          macro avg       0.93      0.89      0.91       624
       weighted avg       0.92      0.92      0.91       624
```
```
from mlxtend.plotting import plot_confusion_matrix

cm = confusion_matrix(y_test,predictions)
total = sum(sum(cm))
acc = (cm[0, 0] + cm[1, 1]) / total
sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])

fig, ax = plot_confusion_matrix(conf_mat=cm , show_normed=True, figsize=(5, 5))
plt.show()
```
![Alt confusion](https://user-images.githubusercontent.com/72612848/96395680-74d67600-11ef-11eb-801c-aae1cb870b7a.png)

Seperti yang dapat dilihat dalam report klassifikasi dan pada confusion matrix, model model yang dihasilkan bekerja dengan baik, baik dalam klasifikasi kasus positif pneumonia maupun dalam klasifikasi kasus negatif.

## Prediksi Gambar
```
testing_path="/content/NORMAL2-IM-1427-0001.jpeg"
img= image.load_img(testing_path)
plt.imshow(img)

x=image.img_to_array(img)
x=x.reshape((1,) + x.shape)

val=model.predict(x_val)
result = np.argmax(val[0]).tolist()
print('Class : '+str(result))
```
![Alt uji](https://user-images.githubusercontent.com/72612848/96395753-a6e7d800-11ef-11eb-9eab-4debfa3f1d59.png)
prediksi yang dilakukan pada gambar normal, dapat di klasifikasikan pada kelas 0 yaitu normal, artinya aplikasi berjalan dengan baik

## Menyimpan Model
```
model.save("project_pneunomia.h5") 
```
## Kesimpulan
Pengujian deep learning dengan menggunakan arsitektur Convolutional Neural Networks (CNN) untuk mengklasifikasikan gambar Chest X-Ray pada kelompok normal atau terdapat penyakit pneunomia didapat hasil yang cukup baik. Namun masih banyak hal yang dapat dikembangkan dengan menyempurnakan parameter atau yang serupa, lebih mendalami AI untuk data yang tidak seimbang maupun cara mengevaluasi model dimiliki.

penting untuk diingat bahwa kernel dan model ini juga tidak disarankan untuk diagnosis klinis karena hanya diperuntukkan sebagai projek akhir Bootcamp Indocyber dan studi kasus pribadi
