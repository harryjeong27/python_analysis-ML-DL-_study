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

# deep learning : 신경망 구조를 본따 만든 모델
# ANN, CNN, RNN, ....

# - node : 뉴런
# - layer : 여러 뉴런이 모여 있는 단위
#   1) input layer : 외부자극을 받아들이는 뉴런의 집합 => 설명변수
#   2) hidden layer : 중간 사고를 담당하는 뉴런의 집합 *** => X, Y관계 파악하는 층
#   3) output layer : 최종 판단을 출력하는 뉴런의 집합 => 종속변수

# 1. ANN classification
# ANN 모델을 통한 iris data의 예측 (꽃의 품종)
import tensorflow as tf                            # tensorflow 문법 구현
import keras                                       # keras 함수 사용

from keras.models import Sequential                # layer를 구성하는 함수
from keras.layers.core import Dense                # node를 구성하는 함수
from keras.utils import np_utils                   # dummy 변수 생성
from sklearn.preprocessing import LabelEncoder     # 숫자형 변수로 변경

# data loading
df_iris = pd.read_csv('iris.csv', names = ['sepal_length', 'sepal_width',
                                           'petal_length', 'petal_width', 'species'])

# array data set으로 변경 => NN은 array, 숫자만 학습 가능
datasets = df_iris.values    # array로 불러줌, 문자형태임
iris_x = datasets[:, :4].astype('float')
iris_y = datasets[:, 4]

# Y(종속변수) 데이터 숫자로 변경
m_label = LabelEncoder()
m_label.fit(iris_y)
iris_y = m_label.transform(iris_y)

# Y(종속변수) 더미변수로 변경 => NN은 0, 1로만 구성/분할 되어 있는 것을 선호함
iris_y_tr = np_utils.to_categorical(iris_y)

# ANN 모델 생성
model = Sequential()    # 빈 껍데기
model.add(Dense(8, input_dim = 4, activation = 'relu'))    # 첫번째 층 => 설명변수 4개
# 두번째 층(첫번째 hidden layer)의 노드의 개수 8 => NN에서 input layer - 첫번째 hidden layer은 동시에 존재할 수 밖에 없으므로 2개 동시 생성
model.add(Dense(16, activation = 'relu'))    # 세번째 층 => 노드 16개
# relu => 중간층에 사용

model.add(Dense(3, activation = 'softmax'))    # 마지막 층 => 종속변수 3개 (더미변수로 3개로 나눠줌)

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
              loss = 'categorical_crossentropy',   # 오차를 측정할 함수
              metrics = ['accuracy'])              # 평가매트릭

# 모델 학습
model.fit(iris_x, iris_y_tr, epochs = 50, batch_size = 1)    # 50번 반복

# 모델 평가
model.evaluate(iris_x, iris_y_tr)[0]    # loss => 0.0873
model.evaluate(iris_x, iris_y_tr)[1]    # score => 0.980 (중간층 1개), 0.946 (중간층 2개)
# 반복할 때마다 달라짐

Y = w1X1 + w2X2 + .... + w10X10    # 회귀분석에서 가중치의 합인 Y가 0 또는 1이 되기는 어려움
# 회귀분석에서 loss는 실제값과 Y값의 차이 ex) 실제 부동산값 5억, Y가 5억1천 -> loss 줄이기 위해 모델이 계속 수정됨
# => 위 과정을 epochs 수치만큼 반복 -> 얼마이상 오차가 없다면 그만하라고 하는 stop point 지정 가능
# => batch_size는 한번에 epochs 몇개를 묶어서 작업하겠다는 의미 -> 컴퓨터 용량 커야 함

# =============================================================================
# 예제) cancer.csv 평가할 때 평가 점수를 그대로 사용한 예제이니, split해서 진행해보기
# 1) data loading
cancer = pd.read_csv('cancer.csv', index_col = 0)

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

# 모델 컴파일
model.compile(optimizer = 'adam',                  # 성능개선을 위한 성능화 함수
              loss = 'categorical_crossentropy',   # 오차를 측정할 함수 => 분류분석에 자주 사용
              metrics = ['accuracy'])              # 평가매트릭 => 몇개 중 몇개를 맞췄는냐? (분류분석용)

# 모델 학습
model.fit(train_x, train_y, epochs = 25, batch_size = 1)    # 25번 반복
# epochs를 줄이면 과대적합을 피할 수도 있다, 너무 많은 epochs는 좋지 않음

# 모델 평가
model.evaluate(train_x, train_y)[0]    # loss => 0.0012
model.evaluate(train_x, train_y)[1]    # score => 1.0 (중간층 1개)

model.evaluate(test_x, test_y)[0]    # loss => 0.124
model.evaluate(test_x, test_y)[1]    # score => 0.937 (중간층 1개)

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

# =============================================================================
# # 선형 회귀와 비교
# =============================================================================
from sklearn.linear_model import LinearRegression

m_reg = LinearRegression()
m_reg.fit(train_x, train_y)
m_reg.score(test_x, test_y)    # 0.607 => 훨씬 낮음 why? 교호작용 등 진행하며 다중공선성 발생
# NN은 릿지, 라쏘처럼 불필요한 변수들을 가공했다는 의미
# --------------------------------------------------------------------------- #