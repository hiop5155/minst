# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from keras.utils import np_utils 
import tensorflow as tf
import keras
np.random.seed(10)
from keras.datasets import mnist

(X_train_image, y_train_label), \
(X_test_image, y_test_label) = mnist.load_data()

print("train data=",len(X_train_image))
print("train data=",len(X_test_image))

print("X_train_image:",X_train_image.shape)
print("y_train_label:",y_train_label.shape)

#look one image function
import matplotlib.pyplot as plt
def plot_image(image):
        fig = plt.gcf()
        fig.set_size_inches(2,2)
        plt.imshow(image, cmap= 'binary')
        plt.show

plot_image(X_train_image[0])
y_train_label[0]

#look lots image function
def plot_images_labels_prediction(images, labels, prediction, idx, num=10):
    fig = plt.gcf()
    fig.set_size_inches(12,14)
    if num>25: num=25
    for i in range(0, num):
        ax = plt.subplot(5,5, 1+i)
        ax.imshow(images[idx], cmap='binary')
        title="label="+ str(labels[idx])
        if len(prediction)>0:
            title+=",prediction="+str(prediction[idx])
            
        ax.set_title(title,fontsize=10)
        ax.set_xticks([]);ax.set_yticks([])
        idx+=1
    plt.show

plot_images_labels_prediction(X_train_image, y_train_label, [], 0,10)

print('X_test_image:', X_test_image.shape)
print('y_test_label:', y_test_label.shape)

plot_images_labels_prediction(X_test_image, y_test_label, [], 0,10)

#將28*28二維圖轉成784一維向量 才能當input
X_Train = X_train_image.reshape(60000,784).astype('float32')
X_Test = X_test_image.reshape(10000,784).astype('float32')
#image mormalize
X_Train_normalize = X_Train/255
X_Test_normalize = X_Test/255
#one-hot encoding
y_TrainOneHot = np_utils.to_categorical(y_train_label)
y_TestOneHot = np_utils.to_categorical(y_test_label)

#modeling
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
#輸入層與隱藏層建立
model.add(Dense(units=256,
                input_dim=784,
                kernel_initializer='normal',
                activation='relu'))
#輸出層建立
model.add(Dense(units=10,
                kernel_initializer='normal',
                activation='softmax'))
#print(model.summary())
#training
model.compile(loss='categorical_crossentropy',
              optimizer='adam',metrics=['accuracy'])
for i in range (10):
    train_history = model.fit(x=X_Train_normalize,
                              y=y_TrainOneHot,
                              validation_split=0.2,
                              epochs=10, batch_size=200, verbose=2)
#查看訓練結果
def show_train_history(train_history,train,validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train','validation'], loc='upper left')
    plt.show()
show_train_history(train_history,'acc','val_acc')
show_train_history(train_history,'loss','val_loss')

#以test data確認模型好壞
scores = model.evaluate(X_Test_normalize, y_TestOneHot)
print()
print('accuracy=',scores[1])

#預測
prediction = model.predict_classes(X_Test)
plot_images_labels_prediction(X_test_image,y_test_label,prediction,idx=340)

#confusion matrix
#import pandas as pd
pd.crosstab(y_test_label,prediction,rownames=['label'],colnames=['predict'])
df = pd.DataFrame({'label':y_test_label, 'predict':prediction})
df[:2]
#找label5 prediction3
df[(df.label==5)&(df.predict==3)]
#plot_images_labels_prediction(X_test_image,y_test_label,prediction,idx=340,num=1)

#把隱藏層改為1000個神經元看看
model = Sequential()
#輸入層與隱藏層建立
model = Sequential()
model.add(Dense(units=1000,
                input_dim=784,
                kernel_initializer='normal',
                activation='relu'))
#輸出層建立
model.add(Dense(units=10,
                kernel_initializer='normal',
                activation='softmax'))
#print(model.summary())

#train
model.compile(loss='categorical_crossentropy',
              optimizer='adam',metrics=['accuracy'])
train_history=model.fit(x=X_Train_normalize,
                        y=y_TrainOneHot,
                        validation_split=0.2,
                        epochs=10,batch_size=200,verbose=2)
#訓練結果
show_train_history(train_history,'acc','val_acc')
#預測準確率
scores = model.evaluate(X_Test_normalize, y_TestOneHot)
print('Test loss:', scores[0])
print('accuracy',scores[1])

#儲存模型
model.save('minst_prediction_use_MLP.h5')

#載入之前的模型
model = tf.contrib.keras.models.load_model('minst_prediction_use_MLP.h5')