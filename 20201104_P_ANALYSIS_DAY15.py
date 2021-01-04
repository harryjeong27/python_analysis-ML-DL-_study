run profile1

# ANN 이미지분석
from keras.datasets import mnist
from keras.utils import np_utils
import sys
import tensorflow as tf
import keras

# seed 고정
seed = 0
np.random.seed(seed)
tf.random.set_seed(seed)

# MNIST 데이터셋 불러오기
(X_train, Y_class_train), (X_test, Y_class_test) = mnist.load_data()

X_train.shape  # (60000, 28, 28) : 28 X 28의 해상도를 갖는 60000개의 data

print("학습셋 이미지 수 : %d 개" % (X_train.shape[0]))
print("테스트셋 이미지 수 : %d 개" % (X_test.shape[0]))

# 데이터 확인 - 그래프로 확인
import matplotlib.pyplot as plt
plt.imshow(X_train[0], cmap='Greys')

# 데이터 확인 - 코드로 확인
for x in X_train[0] :
    for i in x :
        sys.stdout.write('%d\t' % i)
    sys.stdout.write('\n')

# 차원 변환 과정
X_train = X_train.reshape(X_train.shape[0], 784)   # 2차원 형태로의 학습
X_train = X_train.astype('float64')
X_train = X_train / 255                            # minmax scaling

X_test = X_test.reshape(X_test.shape[0], 784).astype('float64') / 255

# 바이너리화 과정(더미변수로 변경)
Y_train = np_utils.to_categorical(Y_class_train, 10)
Y_test = np_utils.to_categorical(Y_class_test, 10)

# ANN 모델 설정
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping

model = Sequential()
model.add(Dense(512, input_dim=784, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# 모델 최적화 설정
import os
MODEL_DIR = './model/'

if not os.path.exists(MODEL_DIR) :    # [ -d $MODEL_DIR ]
    os.mkdir(MODEL_DIR)

modelpath="./model/{epoch:02d}-{val_loss:.4f}.hdf5"

checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss',
                               verbose=1, save_best_only=True)

early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)

# 모델의 실행
history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test),
                    epochs=30, batch_size=200, verbose=0,
                    callbacks=[early_stopping_callback,checkpointer])

# 테스트 정확도 출력
print("\n Test Accuracy: %.4f" % (model.evaluate(X_test, Y_test) [1]))

# 오차 확인 및 시각화
y_vloss = history.history['val_loss']
y_loss = history.history['loss']

x_len = np.arange(len(y_loss))
plt.plot(x_len, y_vloss, marker='.', c="red", label='Testset_loss')
plt.plot(x_len, y_loss, marker='.', c="blue", label='Trainset_loss')

plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')

# --------------------------------------------------------------------------- #

# CNN 학습 - mnist data set
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint,EarlyStopping
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import keras

# 1) seed값 고정
seed = 0
np.random.seed(seed)
tf.random.set_seed(seed)

# 2) 데이터 불러오기
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_train.shape    # (60000, 28, 28)

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32') / 255
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32') / 255
Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)

# 3) 컨볼루션 신경망 설정
model = Sequential()
model.add(Conv2D(32,                      # 32개의 filter 생성
                 kernel_size=(3, 3),      # 9개의 인근 pixel에 가중치 부여
                 input_shape=(28, 28, 1), 
                 activation='relu'))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=2))            # 선택적
# pooling : 차원 축소 기법, 의미 있는 신호만 전달
# - maxpooling : 가장 큰 신호만 전달
model.add(Dropout(0.25))                        # 선택적
model.add(Flatten())                            # 필수(NN의 output이 1차원)
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))                         # 선택적
# dropout : 차원 축소 기법, 신호를 아예 꺼버리는 방식
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# 4) 모델 최적화 설정
MODEL_DIR = './model/'

if not os.path.exists(MODEL_DIR) :
    os.mkdir(MODEL_DIR)
    
modelpath="./model/{epoch:02d}-{val_loss:.4f}.hdf5"

checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss',
                               verbose=1, save_best_only=True)

early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)

# 5) 모델의 실행
history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test),
                    epochs=30, 
                    batch_size=200, 
                    verbose=1,        # 1이 상세과정 출력, 0이 요약
                    callbacks=[early_stopping_callback,checkpointer])

# 6) 테스트 정확도 출력
print("\n Test Accuracy: %.4f" % (model.evaluate(X_test, Y_test)[1]))

# 7) 오차 확인 및 시각화
y_vloss = history.history['val_loss']
y_loss = history.history['loss']

x_len = np.arange(len(y_loss))
plt.plot(x_len, y_vloss, marker='.', c="red", label='Testset_loss')
plt.plot(x_len, y_loss, marker='.', c="blue", label='Trainset_loss')

plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')

# [ 연습 문제 - 얼굴인식 data의 deep learning model 적용 ]
# 1. ANN
# data loading
from sklearn.datasets import fetch_lfw_people
people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)

people.data.shape

# down sampling
v_nrow = []
for i in np.unique(people.target):
    nrow = np.where(people.target == i)[0][:50]
    v_nrow = v_nrow + list(nrow)

people_x = people.data[v_nrow]
people_y = people.target[v_nrow]

# train, test data split
train_x, test_x, train_y, test_y = train_test_split(people_x,
                                                    people_y,
                                                    random_state=0)

# data scaling
train_x = train_x.astype('float64') / 255
test_x = test_x.astype('float64') / 255

# 종속변수의 이진화(더미변수 생성)
train_y = np_utils.to_categorical(train_y, len(np.unique(people.target)))
test_y = np_utils.to_categorical(test_y, len(np.unique(people.target)))

# ANN 모델 설정
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping

model = Sequential()
model.add(Dense(2800, input_dim=5655, activation='relu'))
model.add(Dense(1400, activation='relu'))
model.add(Dense(700, activation='relu'))
model.add(Dense(350, activation='relu'))
model.add(Dense(62, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# 모델 최적화 설정
import os
MODEL_DIR = './model/'

if not os.path.exists(MODEL_DIR) :    # [ -d $MODEL_DIR ]
    os.mkdir(MODEL_DIR)

modelpath="./model/{epoch:02d}-{val_loss:.4f}.hdf5"

checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss',
                               verbose=1, save_best_only=True)

early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)

# 모델의 실행
history = model.fit(train_x, train_y, validation_data=(test_x, test_y),
                    epochs=200, batch_size=200, 
                    verbose=0,    # 0이 자세히, 1이 간략
                    callbacks=[early_stopping_callback,checkpointer])

# 테스트 정확도 출력
print("\n Test Accuracy: %.4f" % (model.evaluate(test_x, test_y)[1]))

# 2. CNN
# data loading
from sklearn.datasets import fetch_lfw_people
people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)

# down sampling
v_nrow = []
for i in np.unique(people.target):
    nrow = np.where(people.target == i)[0][:50]
    v_nrow = v_nrow + list(nrow)

people_x = people.data[v_nrow]
people_y = people.target[v_nrow]

# train, test split
train_x, test_x, train_y, test_y = train_test_split(people_x,
                                                    people_y,
                                                    random_state=0)

# CNN 학습용 reshape
train_x = train_x.reshape(train_x.shape[0],87,65,1).astype('float64') / 255
test_x = test_x.reshape(test_x.shape[0],87,65,1).astype('float64') / 255

# Y값 이진화(더미변수 생성)
train_y = np_utils.to_categorical(train_y,  len(np.unique(people.target)))
test_y = np_utils.to_categorical(test_y,  len(np.unique(people.target)))

# 모델 설정
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), input_shape=(87, 65, 1), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=2))            
model.add(Dropout(0.25))                        
model.add(Flatten())                                           
model.add(Dense(700, activation='relu'))
model.add(Dense(350, activation='relu'))
model.add(Dense(62, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# 모델 최적화 설정
import os
MODEL_DIR = './model/'

if not os.path.exists(MODEL_DIR) :    # [ -d $MODEL_DIR ]
    os.mkdir(MODEL_DIR)

modelpath="./model/{epoch:02d}-{val_loss:.4f}.hdf5"

checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss',
                               verbose=1, save_best_only=True)

early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)


# 모델의 실행
history = model.fit(train_x, train_y, validation_data=(test_x, test_y),
                    epochs=200, batch_size=200, verbose=0,
                    callbacks=[early_stopping_callback,checkpointer])

# 테스트 정확도 출력
print("\n Test Accuracy: %.4f" % (model.evaluate(test_x, test_y) [1]))














