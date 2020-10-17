# Chest-X-Ray-Pneunomia
pengujian deep learning dengan menggunakan Convolutional Neural Networks (CNN) untuk mengklasifikasikan gambar Chest X-Ray pada kelompok normal atau terdapat penyakit pneunomia.
 ```
Saving kaggle.json to kaggle.json
{'kaggle.json': b'{"username":"amangiwan","key":"fb75919f49b5c8dd5087c496391a033c"}'}
 ```
 
 [image](https://user-images.githubusercontent.com/72612848/96331837-b4d01880-108a-11eb-959c-f612ddd84f54.jpeg)
 
 ![Alt NORMAL2-IM-1427-0001](https://user-images.githubusercontent.com/72612848/96331837-b4d01880-108a-11eb-959c-f612ddd84f54.jpeg)
 
 ```
 Epoch 1/30
  1/130 [..............................] - ETA: 0s - loss: 0.3725 - accuracy: 0.8750 - precision: 0.9600 - recall: 0.8889WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow/python/ops/summary_ops_v2.py:1277: stop (from tensorflow.python.eager.profiler) is deprecated and will be removed after 2020-07-01.
Instructions for updating:
use `tf.profiler.experimental.stop` instead.
131/130 [==============================] - ETA: 0s - loss: 0.1651 - accuracy: 0.9379 - precision: 0.9827 - recall: 0.9326
Epoch 00001: val_accuracy improved from -inf to 0.94460, saving model to weight_pneunomia_best.h5
131/130 [==============================] - 16s 121ms/step - loss: 0.1651 - accuracy: 0.9379 - precision: 0.9827 - recall: 0.9326 - val_loss: 0.1334 - val_accuracy: 0.9446 - val_precision: 0.9878 - val_recall: 0.9372
Epoch 2/30
131/130 [==============================] - ETA: 0s - loss: 0.1654 - accuracy: 0.9350 - precision: 0.9803 - recall: 0.9310
Epoch 00002: val_accuracy did not improve from 0.94460
131/130 [==============================] - 15s 118ms/step - loss: 0.1654 - accuracy: 0.9350 - precision: 0.9803 - recall: 0.9310 - val_loss: 0.1807 - val_accuracy: 0.9351 - val_precision: 0.9917 - val_recall: 0.9205
Epoch 3/30
131/130 [==============================] - ETA: 0s - loss: 0.1634 - accuracy: 0.9372 - precision: 0.9804 - recall: 0.9339
Epoch 00003: val_accuracy did not improve from 0.94460
131/130 [==============================] - 15s 117ms/step - loss: 0.1634 - accuracy: 0.9372 - precision: 0.9804 - recall: 0.9339 - val_loss: 0.1868 - val_accuracy: 0.9312 - val_precision: 0.9944 - val_recall: 0.9128
Epoch 4/30
131/130 [==============================] - ETA: 0s - loss: 0.1645 - accuracy: 0.9331 - precision: 0.9799 - recall: 0.9288
Epoch 00004: val_accuracy improved from 0.94460 to 0.95129, saving model to weight_pneunomia_best.h5
131/130 [==============================] - 15s 118ms/step - loss: 0.1645 - accuracy: 0.9331 - precision: 0.9799 - recall: 0.9288 - val_loss: 0.1327 - val_accuracy: 0.9513 - val_precision: 0.9880 - val_recall: 0.9462
Epoch 5/30
131/130 [==============================] - ETA: 0s - loss: 0.1677 - accuracy: 0.9350 - precision: 0.9790 - recall: 0.9323
Epoch 00005: val_accuracy did not improve from 0.95129
131/130 [==============================] - 15s 117ms/step - loss: 0.1677 - accuracy: 0.9350 - precision: 0.9790 - recall: 0.9323 - val_loss: 0.1587 - val_accuracy: 0.9417 - val_precision: 0.9945 - val_recall: 0.9269
Epoch 6/30
131/130 [==============================] - ETA: 0s - loss: 0.1532 - accuracy: 0.9384 - precision: 0.9801 - recall: 0.9359
Epoch 00006: val_accuracy did not improve from 0.95129
131/130 [==============================] - 15s 118ms/step - loss: 0.1532 - accuracy: 0.9384 - precision: 0.9801 - recall: 0.9359 - val_loss: 0.2012 - val_accuracy: 0.9255 - val_precision: 0.9958 - val_recall: 0.9038
Epoch 7/30
131/130 [==============================] - ETA: 0s - loss: 0.1527 - accuracy: 0.9405 - precision: 0.9808 - recall: 0.9381
Epoch 00007: val_accuracy did not improve from 0.95129
131/130 [==============================] - 15s 118ms/step - loss: 0.1527 - accuracy: 0.9405 - precision: 0.9808 - recall: 0.9381 - val_loss: 0.1314 - val_accuracy: 0.9484 - val_precision: 0.9919 - val_recall: 0.9385
Epoch 8/30
131/130 [==============================] - ETA: 0s - loss: 0.1489 - accuracy: 0.9446 - precision: 0.9845 - recall: 0.9401
Epoch 00008: val_accuracy did not improve from 0.95129
131/130 [==============================] - 15s 118ms/step - loss: 0.1489 - accuracy: 0.9446 - precision: 0.9845 - recall: 0.9401 - val_loss: 0.1561 - val_accuracy: 0.9398 - val_precision: 0.9931 - val_recall: 0.9256
Epoch 9/30
131/130 [==============================] - ETA: 0s - loss: 0.1440 - accuracy: 0.9438 - precision: 0.9838 - recall: 0.9397
Epoch 00009: val_accuracy did not improve from 0.95129
131/130 [==============================] - 15s 118ms/step - loss: 0.1440 - accuracy: 0.9438 - precision: 0.9838 - recall: 0.9397 - val_loss: 0.2075 - val_accuracy: 0.9217 - val_precision: 1.0000 - val_recall: 0.8949
Epoch 10/30
131/130 [==============================] - ETA: 0s - loss: 0.1431 - accuracy: 0.9431 - precision: 0.9835 - recall: 0.9391
Epoch 00010: val_accuracy improved from 0.95129 to 0.96753, saving model to weight_pneunomia_best.h5
131/130 [==============================] - 16s 119ms/step - loss: 0.1431 - accuracy: 0.9431 - precision: 0.9835 - recall: 0.9391 - val_loss: 0.0917 - val_accuracy: 0.9675 - val_precision: 0.9857 - val_recall: 0.9705
Epoch 11/30
131/130 [==============================] - ETA: 0s - loss: 0.1565 - accuracy: 0.9393 - precision: 0.9817 - recall: 0.9355
Epoch 00011: val_accuracy did not improve from 0.96753
131/130 [==============================] - 15s 117ms/step - loss: 0.1565 - accuracy: 0.9393 - precision: 0.9817 - recall: 0.9355 - val_loss: 0.1360 - val_accuracy: 0.9465 - val_precision: 0.9905 - val_recall: 0.9372
Epoch 12/30
131/130 [==============================] - ETA: 0s - loss: 0.1570 - accuracy: 0.9403 - precision: 0.9818 - recall: 0.9368
Epoch 00012: val_accuracy did not improve from 0.96753
131/130 [==============================] - 15s 117ms/step - loss: 0.1570 - accuracy: 0.9403 - precision: 0.9818 - recall: 0.9368 - val_loss: 0.1393 - val_accuracy: 0.9456 - val_precision: 0.9879 - val_recall: 0.9385
Epoch 13/30
131/130 [==============================] - ETA: 0s - loss: 0.1515 - accuracy: 0.9410 - precision: 0.9811 - recall: 0.9384
Epoch 00013: val_accuracy did not improve from 0.96753
131/130 [==============================] - 15s 117ms/step - loss: 0.1515 - accuracy: 0.9410 - precision: 0.9811 - recall: 0.9384 - val_loss: 0.1574 - val_accuracy: 0.9427 - val_precision: 0.9945 - val_recall: 0.9282
Epoch 14/30
131/130 [==============================] - ETA: 0s - loss: 0.1475 - accuracy: 0.9434 - precision: 0.9841 - recall: 0.9388
Epoch 00014: val_accuracy did not improve from 0.96753
131/130 [==============================] - 15s 117ms/step - loss: 0.1475 - accuracy: 0.9434 - precision: 0.9841 - recall: 0.9388 - val_loss: 0.2781 - val_accuracy: 0.9035 - val_precision: 1.0000 - val_recall: 0.8705
Epoch 15/30
131/130 [==============================] - ETA: 0s - loss: 0.1477 - accuracy: 0.9422 - precision: 0.9831 - recall: 0.9381
Epoch 00015: val_accuracy did not improve from 0.96753
131/130 [==============================] - 15s 117ms/step - loss: 0.1477 - accuracy: 0.9422 - precision: 0.9831 - recall: 0.9381 - val_loss: 0.1490 - val_accuracy: 0.9436 - val_precision: 0.9945 - val_recall: 0.9295
Epoch 16/30
131/130 [==============================] - ETA: 0s - loss: 0.1406 - accuracy: 0.9477 - precision: 0.9836 - recall: 0.9452
Epoch 00016: val_accuracy did not improve from 0.96753
131/130 [==============================] - 15s 117ms/step - loss: 0.1406 - accuracy: 0.9477 - precision: 0.9836 - recall: 0.9452 - val_loss: 0.2428 - val_accuracy: 0.9112 - val_precision: 1.0000 - val_recall: 0.8808
Epoch 17/30
131/130 [==============================] - ETA: 0s - loss: 0.1492 - accuracy: 0.9386 - precision: 0.9830 - recall: 0.9333
Epoch 00017: val_accuracy did not improve from 0.96753
131/130 [==============================] - 15s 117ms/step - loss: 0.1492 - accuracy: 0.9386 - precision: 0.9830 - recall: 0.9333 - val_loss: 0.1280 - val_accuracy: 0.9456 - val_precision: 0.9879 - val_recall: 0.9385
Epoch 18/30
131/130 [==============================] - ETA: 0s - loss: 0.1396 - accuracy: 0.9474 - precision: 0.9845 - recall: 0.9439
Epoch 00018: val_accuracy did not improve from 0.96753
131/130 [==============================] - 16s 118ms/step - loss: 0.1396 - accuracy: 0.9474 - precision: 0.9845 - recall: 0.9439 - val_loss: 0.1244 - val_accuracy: 0.9494 - val_precision: 0.9919 - val_recall: 0.9397
Epoch 19/30
131/130 [==============================] - ETA: 0s - loss: 0.1363 - accuracy: 0.9455 - precision: 0.9835 - recall: 0.9423
Epoch 00019: val_accuracy did not improve from 0.96753
131/130 [==============================] - 15s 117ms/step - loss: 0.1363 - accuracy: 0.9455 - precision: 0.9835 - recall: 0.9423 - val_loss: 0.0981 - val_accuracy: 0.9599 - val_precision: 0.9907 - val_recall: 0.9551
Epoch 20/30
131/130 [==============================] - ETA: 0s - loss: 0.1404 - accuracy: 0.9455 - precision: 0.9842 - recall: 0.9417
Epoch 00020: val_accuracy did not improve from 0.96753
131/130 [==============================] - 15s 117ms/step - loss: 0.1404 - accuracy: 0.9455 - precision: 0.9842 - recall: 0.9417 - val_loss: 0.1390 - val_accuracy: 0.9475 - val_precision: 0.9932 - val_recall: 0.9359
Epoch 21/30
131/130 [==============================] - ETA: 0s - loss: 0.1313 - accuracy: 0.9474 - precision: 0.9845 - recall: 0.9439
Epoch 00021: val_accuracy did not improve from 0.96753
131/130 [==============================] - 16s 119ms/step - loss: 0.1313 - accuracy: 0.9474 - precision: 0.9845 - recall: 0.9439 - val_loss: 0.1465 - val_accuracy: 0.9475 - val_precision: 0.9959 - val_recall: 0.9333
Epoch 22/30
131/130 [==============================] - ETA: 0s - loss: 0.1316 - accuracy: 0.9491 - precision: 0.9839 - recall: 0.9468
Epoch 00022: val_accuracy did not improve from 0.96753
131/130 [==============================] - 15s 116ms/step - loss: 0.1316 - accuracy: 0.9491 - precision: 0.9839 - recall: 0.9468 - val_loss: 0.1295 - val_accuracy: 0.9465 - val_precision: 0.9905 - val_recall: 0.9372
Epoch 23/30
131/130 [==============================] - ETA: 0s - loss: 0.1342 - accuracy: 0.9474 - precision: 0.9836 - recall: 0.9449
Epoch 00023: val_accuracy did not improve from 0.96753
131/130 [==============================] - 15s 118ms/step - loss: 0.1342 - accuracy: 0.9474 - precision: 0.9836 - recall: 0.9449 - val_loss: 0.1254 - val_accuracy: 0.9503 - val_precision: 0.9892 - val_recall: 0.9436
Epoch 24/30
131/130 [==============================] - ETA: 0s - loss: 0.1318 - accuracy: 0.9481 - precision: 0.9839 - recall: 0.9455
Epoch 00024: val_accuracy did not improve from 0.96753
131/130 [==============================] - 15s 118ms/step - loss: 0.1318 - accuracy: 0.9481 - precision: 0.9839 - recall: 0.9455 - val_loss: 0.0882 - val_accuracy: 0.9675 - val_precision: 0.9882 - val_recall: 0.9679
Epoch 25/30
131/130 [==============================] - ETA: 0s - loss: 0.1364 - accuracy: 0.9436 - precision: 0.9831 - recall: 0.9401
Epoch 00025: val_accuracy did not improve from 0.96753
131/130 [==============================] - 15s 117ms/step - loss: 0.1364 - accuracy: 0.9436 - precision: 0.9831 - recall: 0.9401 - val_loss: 0.1299 - val_accuracy: 0.9475 - val_precision: 0.9932 - val_recall: 0.9359
Epoch 26/30
131/130 [==============================] - ETA: 0s - loss: 0.1228 - accuracy: 0.9508 - precision: 0.9853 - recall: 0.9478
Epoch 00026: val_accuracy did not improve from 0.96753
131/130 [==============================] - 15s 117ms/step - loss: 0.1228 - accuracy: 0.9508 - precision: 0.9853 - recall: 0.9478 - val_loss: 0.2929 - val_accuracy: 0.8873 - val_precision: 1.0000 - val_recall: 0.8487
Epoch 27/30
131/130 [==============================] - ETA: 0s - loss: 0.1332 - accuracy: 0.9481 - precision: 0.9846 - recall: 0.9449
Epoch 00027: val_accuracy did not improve from 0.96753
131/130 [==============================] - 15s 117ms/step - loss: 0.1332 - accuracy: 0.9481 - precision: 0.9846 - recall: 0.9449 - val_loss: 0.1570 - val_accuracy: 0.9389 - val_precision: 0.9918 - val_recall: 0.9256
Epoch 28/30
131/130 [==============================] - ETA: 0s - loss: 0.1266 - accuracy: 0.9505 - precision: 0.9872 - recall: 0.9455
Epoch 00028: val_accuracy did not improve from 0.96753
131/130 [==============================] - 15s 118ms/step - loss: 0.1266 - accuracy: 0.9505 - precision: 0.9872 - recall: 0.9455 - val_loss: 0.1083 - val_accuracy: 0.9551 - val_precision: 0.9933 - val_recall: 0.9462
Epoch 29/30
131/130 [==============================] - ETA: 0s - loss: 0.1230 - accuracy: 0.9565 - precision: 0.9880 - recall: 0.9529
Epoch 00029: val_accuracy did not improve from 0.96753
131/130 [==============================] - 15s 118ms/step - loss: 0.1230 - accuracy: 0.9565 - precision: 0.9880 - recall: 0.9529 - val_loss: 0.1344 - val_accuracy: 0.9551 - val_precision: 0.9946 - val_recall: 0.9449
Epoch 30/30
131/130 [==============================] - ETA: 0s - loss: 0.1386 - accuracy: 0.9446 - precision: 0.9828 - recall: 0.9417
Epoch 00030: val_accuracy did not improve from 0.96753
131/130 [==============================] - 15s 117ms/step - loss: 0.1386 - accuracy: 0.9446 - precision: 0.9828 - recall: 0.9417 - val_loss: 0.0894 - val_accuracy: 0.9628 - val_precision: 0.9869 - val_recall: 0.9628
```
