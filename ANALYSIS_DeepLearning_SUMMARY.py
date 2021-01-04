# Deep Learning

# machine learning > deep learning
# deep learning 구현
# - tensorflow
# - 이전 모델, 문법 복잡
# - keras
# - 최신, 문법 쉬움, 성능 비슷

pip install tensorflow
!pip install keras    # 현재 프롬프트 이전 프롬프트에 명령어 전달하는 방법 : ! 사용
import tensorflow as tf

pip install keras
import keras as kr

# 1. ANN classification
# deep learning : 신경망 구조를 본따 만든 모델
# ANN, CNN, RNN, ....

# - node : 뉴런
# - layer : 여러 뉴런이 모여 있는 단위
#   1) input layer : 외부자극을 받아들이는 뉴런의 집합 => 설명변수
#   2) hidden layer : 중간 사고를 담당하는 뉴런의 집합 *** => X, Y관계 파악하는 층
#   3) output layer : 최종 판단을 출력하는 뉴런의 집합 => 종속변수

# 예제) cancer.csv 평가할 때 평가 점수를 그대로 사용한 예제이니, split해서 진행해보기
import tensorflow as tf                            # tensorflow 문법 구현
import keras                                       # keras 함수 사용

from keras.models import Sequential                # layer를 구성하는 함수
from keras.layers.core import Dense                # node를 구성하는 함수
from keras.utils import np_utils                   # dummy 변수 생성
from sklearn.preprocessing import LabelEncoder     # 숫자형 변수로 변경

# 1) data loading
cancer = pd.read_csv('cancer.csv', index_col = 0)

# array data set으로 변경 => NN은 array, 숫자만 학습 가능
cancer_datasets = cancer.values
cancer_y = cancer_datasets[:, 0]
cancer_x = cancer_datasets[:, 1:].astype('float')

# 2) data 변환
# 2-1) Y(종속변수) 데이터 숫자로 변경
m_label = LabelEncoder()
m_label.fit(cancer_y)
cancer_y_tr = m_label.transform(cancer_y)    # 이미 0, 1로 바뀜 => 출력 노드 1개

# Y(종속변수) 더미변수로 변경 => NN은 0, 1로만 구성/분할 되어 있는 것을 선호함
cancer_y_tr = np_utils.to_categorical(cancer_y_tr)    # 출력 노드 2개

# 2-2) scaling
m_sc = standard()
m_sc.fit(cancer_x)
cancer_x_sc = m_sc.transform(cancer_x)

# 3) data split
train_x, test_x, train_y, test_y = train_test_split(cancer_x_sc,
                                                    cancer_y_tr,
                                                    random_state = 0)

# 4) ANN 모델 생성
nx = train_x.shape[1]
model = Sequential()    # 빈 껍데기
model.add(Dense(15, input_dim = nx, activation = 'relu'))    # 첫번째 층 => 설명변수 30개 (len(train_x[0]))
# 두번째 층(첫번째 hidden layer)의 노드의 개수 15 => NN에서 input layer - 첫번째 hidden layer은 동시에 존재할 수 밖에 없으므로 2개 동시 생성
# relu => 중간층에 사용, 마지막 층은 다양하게 사양할 수 있음

model.add(Dense(2, activation = 'softmax'))    # 마지막 층 => 종속변수 2개 (더미변수로 2개로 나눠줌)

# activation => 활성화 함수 : 뉴런에 입력된 신호를 다음 뉴런에 전달 할지 결정하는 함수
#            => 실제로 NN은 외부자극에 대해 모두 반응하는 것은 아님, 일정 크기 이상일 경우 전달

# softmax => 마지막 층에서 Y의 값(가장 큰 신호에 1을 나머지 신호에 0)을 변환하도록 도와주는 함수
#            ex) 마지막층에 온 신호가 10, 15, 9 -> 0, 1, 0으로 변환하여 Y값 전달
# 마지막 노드의 개수가 1개이면 회귀일 가능성이 높음, 2개 이상이면 무조건 분류
# => 선호하지는 않으나 분류분석에서 마지막 노드의 개수(종속변수)가 1개일 수도 있음
# => 회귀분석일 경우 마지막 노드를 0, 1 등으로 바꾸지 않아도 됨 (알아서 조절)
# ex) model.add(Dense(1)) => 회귀 모델

# 모델 컴파일
model.compile(optimizer = 'adam',                  # 성능개선을 위한 성능화 함수
              loss = 'categorical_crossentropy',   # 오차를 측정할 함수 => 분류분석에 자주 사용
              metrics = ['accuracy'])              # 평가매트릭 => 몇개 중 몇개를 맞췄는냐? (분류분석용)

# =============================================================================
# ANN 모델에서의 Y의 형태에 따른 활성화 함수
# 1) Y가 연속형
#   - 활성화 함수 필요 없음 (있어도 오류는 발생 X)
#   
# 2) Y가 범주형(2개 level)
#   - 1개의 Y로 학습되는 경우 : 주로 sigmoid function (0 또는 1의 신호로 변환해주므로)  
#   - 2개의 Y로 분리학습 되는 경우 : softmax function
#   
# 3) Y가 범주형(3개 level 이상)
#   - 0, 1, 2로의 신호 변환을 해주는 활성화 함수 없으므로
#     반드시 레벨의 수 만큼 Y를 분할하여 학습시켜야 함
#     
# loss 함수
# 1) 회귀 모델 (Y가 연속형) : MSE 기반 오차 함수 사용
#     - mean_squared_error : (y-yhat)**2
#     - mean_absolute_error : |y-yhat|
#     - mean_absolute_percentage_error 
#     ....
# 
# 2) 분류 모델 (Y가 범주형) : crossentropy 계열 함수 (log함수 기반)
#     - Y의 범주가 2개 : binary_crossentropy
#     - Y의 범주가 3개 이상 : categorical_crossentropy
# =============================================================================

# 모델 학습
model.fit(train_x, train_y, epochs = 25, batch_size = 1)    # 25번 반복
# epochs를 줄이면 과대적합을 피할 수도 있다, 너무 많은 epochs는 좋지 않음

# 모델 평가
model.evaluate(train_x, train_y)[0]    # loss => 0.0012
model.evaluate(train_x, train_y)[1]    # score => 1.0 (중간층 1개)

model.evaluate(test_x, test_y)[0]    # loss => 0.124
model.evaluate(test_x, test_y)[1]    # score => 0.937 (중간층 1개)
# 반복할 때마다 달라짐

Y = w1X1 + w2X2 + .... + w10X10    # 회귀분석에서 가중치의 합인 Y가 0 또는 1이 되기는 어려움
# 회귀분석에서 loss는 실제값과 Y값의 차이 ex) 실제 부동산값 5억, Y가 5억1천 -> loss 줄이기 위해 모델이 계속 수정됨
# => 위 과정을 epochs 수치만큼 반복 -> 얼마이상 오차가 없다면 그만하라고 하는 stop point 지정 가능
# => batch_size는 한번에 epochs 몇개를 묶어서 작업하겠다는 의미 -> 컴퓨터 용량 커야 함

# 6) 모델 저장 및 loading
model.save('model_ann_cancer2.h5')

from keras.models import load_model
model = load_model('model_ann_cancer2.h5')

# 7) 모델 평가
model.evaluate(test_x, test_y)

# 8) 다른 모델과 비교
m_rf = rf_c()
m_rf.fit(train_x, train_y)
m_rf.score(test_x, test_y)    # 0.972

# => 아주 고차원의 데이터는 NN이 효과적 (설명변수 100개 기준?), 그 아래는 RF도 충분함

# 9) 모델 시각화
print(model.summary())

# plot_model 함수를 사용한 layer 이미지 출력
# 1. window용 graphviz 설치 및 설치 경로 path 등록
# download link : https://graphviz.gitlab.io/_pages/Download/Download_windows.html

# 1) GRAPHVIZ_DOT 환경 변수 생성 : graphviz 설치위치/bin
# 2) PATH에 경로 추가

# 2. python용 graphviz 설치
pip install graphviz
conda install graphviz

# 3. python용 pydot 설치
pip install pydot

plt.rc('font', family = 'Malgun Gothic')
model.summary()
from keras.utils import plot_model
plot_model(model, to_file = 'model_ann_cancer.png',
                  show_shapes = True,
                  show_layer_names = True)
# => 실패 -> 따로 해보기
# --------------------------------------------------------------------------- #

# 2. ANN regressor
# - NN은 릿지, 라쏘처럼 불필요한 변수들을 가공힘
# ANN regressor - boston data

import tensorflow as tf
import keras

from keras.models import Sequential 
from keras.layers.core import Dense 
from keras.utils import np_utils 
from sklearn.preprocessing import LabelEncoder

# mean_squared_error를 사용자 정의함수로 구해봄
def mse(y_true, y_pred):
    from keras import backend
    return backend.mean(backend.square(y_pred - y_true), axis=-1)    # backend.mean => keras의 mean

# 평가 메트릭에서 구할 수 없는 값들을 위해 직접 사용자 정의함수로 구해주고 평가메트릭에 넣어줌
# R^2 구하기 위한 사용자 정의 함수
def r_square(y_true, y_pred):
    from keras import backend as K
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return (1 - SS_res/(SS_tot + K.epsilon()))

# R^2로 설명할 수 없는 부분
def r_square_loss(y_true, y_pred):
    from keras import backend as K
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return 1 - ( 1 - SS_res/(SS_tot + K.epsilon()))

# 1) data loading
from mglearn.datasets import load_extended_boston
df_boston_et_x, df_boston_et_y = load_extended_boston()

# 2) scaling
m_sc = standard()
m_sc.fit(df_boston_et_x)
df_boston_et_x = m_sc.transform(df_boston_et_x)

# 3) data split
train_x, test_x, train_y, test_y = train_test_split(df_boston_et_x,
                                                    df_boston_et_y,
                                                    random_state=0)

# 4) 모델 생성
nx = df_boston_et_x.shape[1]

model = Sequential()
model.add(Dense(52, input_dim = nx, activation = 'relu'))
model.add(Dense(26, activation = 'relu'))
model.add(Dense(13, activation = 'relu'))
model.add(Dense(1))

model.compile(optimizer = 'adam', 
              loss = 'mean_squared_error',    # 회귀에서는 mean_squared_error(오차) 사용
              metrics = ['mean_squared_error', mse,    # accuracy x, 회귀분석의 평가메트릭은 R^2 값이 필요함
                         r_square, r_square_loss])

# 5) 자동 학습 중단(EarlyStopping) 적용 
from keras.callbacks import EarlyStopping

earlystopping = EarlyStopping(monitor="mean_squared_error", 
                              patience=10, 
                              verbose=1, 
                              mode='auto')

result = model.fit(train_x, train_y, epochs = 3000, batch_size = 10,
          validation_data=(test_x, test_y), callbacks=[earlystopping])
# split 해놓은 경우
# collbacks 쓰려면 validation data set 꼭 필요

model.fit(df_boston_et_x, df_boston_et_y, epochs = 3000, batch_size = 10,
           validation_split=0.25, callbacks=[earlystopping])    # split 하지 않은 경우

# 6) 모델 시각화
# 6-1) 모델 요약
model.summary()

# 6-2) layer 시각화
from keras.utils import plot_model
plot_model(model, to_file='model_ann_boston.png', 
                  show_shapes=True,
                  show_layer_names=True)

# 7) 모델 평가
result.history.keys() # 모델 정보 확인 => 그냥 정보는 train data set, val_ 정보는 test data set
# dict_keys(['loss', 'mean_squared_error', 'mse', 'r_square', 
#            'r_square_loss', 'val_loss', 'val_mean_squared_error', 
#            'val_mse', 'val_r_square', 'val_r_square_loss'])

plt.plot(result.history['loss'], label = 'train')
plt.plot(result.history['val_loss'], label = 'test', c='red')
plt.legend()

plt.plot(result.history['r_square'], label = 'train')
plt.plot(result.history['val_r_square'], label = 'test', c='red')
plt.legend()

model.evaluate(test_x, test_y)       # r_square: 0.9203
model.evaluate(test_x, test_y)[3]    # r_square: 0.9203
# test data set으로 위에 fit했기 때문에 높게 나옴
# => 아래와 같이 진행하면 더 낮을 수도 있음

model.fit(train_x, train_y, epochs = 3000, batch_size = 10,
           validation_split=0.25, callbacks=[earlystopping])   

# loss, metrics에 전달되는 모든 값 순서대로 출력
# 점수 (tr > te) => train이 보통 더 큼
# 오차 (tr < te) => 위와 반대 개념

# 실제값과 예측값의 비교
v_y_pre = model.predict(test_x).flatten()

for i in range(0, len(test_y)) :
    print('실제값 : %s, 예측값 : %s' % (test_y[i], v_y_pre[i]))
# --------------------------------------------------------------------------- #
    
# 2.1 ANN regressor 모델의 교차 검증 과정
# 1) data load
from mglearn.datasets import load_extended_boston
df_boston_et_x, df_boston_et_y = load_extended_boston()

# 2 data scaling
m_sc = standard()
m_sc.fit(df_boston_et_x)
df_boston_et_x = m_sc.transform(df_boston_et_x)

# 3 자동 학습 중단 모델 생성
earlystopping = EarlyStopping(monitor="mean_squared_error", 
                              patience=10, 
                              verbose=1, 
                              mode='auto')

# 4) 교차 검증을 통한 모델링 수행
from sklearn.model_selection import StratifiedKFold   # 분류
from sklearn.model_selection import KFold             # 회귀

kfold = KFold(n_splits=5, shuffle=True)
vscore = []
for train, test in kfold.split(df_boston_et_x, df_boston_et_y) :
    model = Sequential()
    model.add(Dense(52, input_dim = nx, activation = 'relu'))
    model.add(Dense(26, activation = 'relu'))
    model.add(Dense(13, activation = 'relu'))
    model.add(Dense(1))
    
    model.compile(optimizer = 'adam', 
              loss = 'mean_squared_error',
              metrics = ['mean_squared_error', r_square])
    
    model.fit(df_boston_et_x[train], df_boston_et_y[train], 
              epochs = 500, batch_size = 10,
              validation_split=0.2, callbacks=[earlystopping])
    
    vrsquare = model.evaluate(df_boston_et_x[test], df_boston_et_y[test])[2]
    vscore.append(vrsquare)

np.mean(vscore)

# sigmoid => 각 노드에서 다음 노드로 신호를 보낼 때 특정 값 이상일 경우만 보냄
# relu => sigmoid의 대체    
# --------------------------------------------------------------------------- #

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import keras

# 2.2 단순 선형 회귀의 계수 추정 과정 - 경사하강법
# 1) x, y의 데이터 값
data = [[2, 81], [4, 93], [6, 91], [8, 97]]

x_data = [x_row[0] for x_row in data]   # 공부시간
y_data = [y_row[1] for y_row in data]   # 시험성적

# 2) 초기 계수 생성
a = tf.Variable(tf.random.uniform([1], 0, 10, dtype=tf.float64, seed=0))
b = tf.Variable(tf.random.uniform([1], 0, 100, dtype=tf.float64, seed=0))

# 3) 선형 회귀 식 생성
y = a * x_data + b     # y = predict value

# 4) RMSE 함수 생성
rmse = tf.sqrt(tf.reduce_mean(tf.square(y - y_data)))

# 학습률 값
learning_rate = 0.1

# RMSE 값을 최소로 하는 값 찾기
gradient_decent = tf.compat.v1.train.GradientDescentOptimizer(learning_rate).minimize(rmse)

# 텐서플로를 이용한 학습
with tf.Session() as sess:
 # 변수 초기화
 sess.run(tf.global_variables_initializer())
 
 # 2001번 실행(0번째를 포함하므로)
 for step in range(2001) :
     sess.run(gradient_decent)
     
 # 100번마다 결과 출력
     if step % 100 == 0 :
         print("Epoch: %.f, RMSE = %.04f, 기울기 a = %.4f, y 절편 b= %.4f" % (step,sess.run(rmse), sess.run(a), sess.run(b))) 

# 선형 회귀를 통한 계수의 추정
from sklearn.linear_model import LinearRegression     
m_reg = LinearRegression()
m_reg.fit(np.array(x_data).reshape(-1,1), y_data)

m_reg.coef_        # 2.3
m_reg.intercept_   # 79
# 따라서, y = 2.3 * X + 79 라는 선형 회귀식이 추정됨
# --------------------------------------------------------------------------- #

# 2.3 로지스틱 회귀
# - 회귀선 추정으로 y의 값을 분류할 수 있음
# - 참, 거짓으로 구성된 factor의 level이 두 개인 종속변수의 형태
# - 로지스틱 회귀식을 sigmoid function이라 함

# 2.4 오차 역전파(backpropagation)
# - 기울기를 추정하는 과정에서 마지막 노드에서나 결정되는 오차를
#   이전 layer들에게 전달하여 해당 layer에서의 오차와 기울기와의 관계를 파악하는 방식
# - 각 가중치에 따라 전달되는 오차도 비례한다는 가정하에
#   오차를 각 가중치에 따라 분해하여 역으로 전달하는 방식
# --------------------------------------------------------------------------- #

# 3. 이미지분석 by Deep Learning
# 3.1 ANN 이미지분석 (학습 - mnist data set)
from keras.datasets import mnist
from keras.utils import np_utils
import sys
import tensorflow as tf
import keras

# 1) seed 고정
seed = 0
np.random.seed(seed)
tf.random.set_seed(seed)

# 2) MNIST 데이터셋 불러오기
(X_train, Y_class_train), (X_test, Y_class_test) = mnist.load_data()

X_train.shape  # (60000, 28, 28) : 28 X 28의 해상도를 갖는 60000개의 data

print("학습셋 이미지 수 : %d 개" % (X_train.shape[0]))
print("테스트셋 이미지 수 : %d 개" % (X_test.shape[0]))

# 3) 데이터 확인 - 그래프로 확인
import matplotlib.pyplot as plt
plt.imshow(X_train[0], cmap='Greys')

# 4) 데이터 확인 - 코드로 확인
for x in X_train[0] :
    for i in x :
        sys.stdout.write('%d\t' % i)
    sys.stdout.write('\n')

# 5) 차원 변환 과정
X_train = X_train.reshape(X_train.shape[0], 784)   # 2차원 형태로의 학습
X_train = X_train.astype('float64')
X_train = X_train / 255                            # minmax scaling

X_test = X_test.reshape(X_test.shape[0], 784).astype('float64') / 255

# 6) 바이너리화 과정(더미변수로 변경)
Y_train = np_utils.to_categorical(Y_class_train, 10)
Y_test = np_utils.to_categorical(Y_class_test, 10)

# 7) ANN 모델 설정
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

# 8) 모델 최적화 설정
import os
MODEL_DIR = './model/'

if not os.path.exists(MODEL_DIR) :    # [ -d $MODEL_DIR ]
    os.mkdir(MODEL_DIR)

modelpath="./model/{epoch:02d}-{val_loss:.4f}.hdf5"

checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss',
                               verbose=1, save_best_only=True)

early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)

# 9) 모델의 실행
history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test),
                    epochs=30, batch_size=200, verbose=0,
                    callbacks=[early_stopping_callback,checkpointer])

# 10) 테스트 정확도 출력
print("\n Test Accuracy: %.4f" % (model.evaluate(X_test, Y_test) [1]))

# 11) 오차 확인 및 시각화
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
# 1) data loading
from sklearn.datasets import fetch_lfw_people
people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)

people.data.shape

# 2) down sampling
v_nrow = []
for i in np.unique(people.target):
    nrow = np.where(people.target == i)[0][:50]
    v_nrow = v_nrow + list(nrow)

people_x = people.data[v_nrow]
people_y = people.target[v_nrow]

# 3) train, test data split
train_x, test_x, train_y, test_y = train_test_split(people_x,
                                                    people_y,
                                                    random_state=0)

# 4) data scaling
train_x = train_x.astype('float64') / 255
test_x = test_x.astype('float64') / 255

# 5) 종속변수의 이진화(더미변수 생성)
train_y = np_utils.to_categorical(train_y, len(np.unique(people.target)))
test_y = np_utils.to_categorical(test_y, len(np.unique(people.target)))

# 6) ANN 모델 설정
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

# 7) 모델 최적화 설정
import os
MODEL_DIR = './model/'

if not os.path.exists(MODEL_DIR) :    # [ -d $MODEL_DIR ]
    os.mkdir(MODEL_DIR)

modelpath="./model/{epoch:02d}-{val_loss:.4f}.hdf5"

checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss',
                               verbose=1, save_best_only=True)

early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)

# 8) 모델의 실행
history = model.fit(train_x, train_y, validation_data=(test_x, test_y),
                    epochs=200, batch_size=200, 
                    verbose=0,    # 0이 자세히, 1이 간략
                    callbacks=[early_stopping_callback,checkpointer])

# 9) 테스트 정확도 출력
print("\n Test Accuracy: %.4f" % (model.evaluate(test_x, test_y)[1]))
# --------------------------------------------------------------------------- #

# 3.2 CNN 이미지분석 (학습 - mnist data set)
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
# 1) data loading
from sklearn.datasets import fetch_lfw_people
people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)

# 2) down sampling
v_nrow = []
for i in np.unique(people.target):
    nrow = np.where(people.target == i)[0][:50]
    v_nrow = v_nrow + list(nrow)

people_x = people.data[v_nrow]
people_y = people.target[v_nrow]

# 3) train, test split
train_x, test_x, train_y, test_y = train_test_split(people_x,
                                                    people_y,
                                                    random_state=0)

# 4) CNN 학습용 reshape
train_x = train_x.reshape(train_x.shape[0],87,65,1).astype('float64') / 255
test_x = test_x.reshape(test_x.shape[0],87,65,1).astype('float64') / 255

# 5) Y값 이진화(더미변수 생성)
train_y = np_utils.to_categorical(train_y,  len(np.unique(people.target)))
test_y = np_utils.to_categorical(test_y,  len(np.unique(people.target)))

# 6) 모델 설정
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

# 7) 모델 최적화 설정
import os
MODEL_DIR = './model/'

if not os.path.exists(MODEL_DIR) :    # [ -d $MODEL_DIR ]
    os.mkdir(MODEL_DIR)

modelpath="./model/{epoch:02d}-{val_loss:.4f}.hdf5"

checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss',
                               verbose=1, save_best_only=True)

early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)


# 8) 모델의 실행
history = model.fit(train_x, train_y, validation_data=(test_x, test_y),
                    epochs=200, batch_size=200, verbose=0,
                    callbacks=[early_stopping_callback,checkpointer])

# 9) 테스트 정확도 출력
print("\n Test Accuracy: %.4f" % (model.evaluate(test_x, test_y) [1]))
# --------------------------------------------------------------------------- #