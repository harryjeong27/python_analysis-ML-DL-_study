# [ 데이터 분석 ]
# - 데이터로부터 의미 있는 정보를 추출, 정보를 통해 미래를 예측하는 모든 행위
# ex)
# - 주가, 날씨, 수요 예측
# - 양/악성 유무 예측
# - 고객 이탈 예측
# - 연관성 분석 (장바구니 분석)
# - 고객의 세분화

# [ 데이터 분석의 분류 ]
# - 데이터마이닝 : 예측이 아닌 그래프나 수치분석 등을 통한 분석

# - 머신러닝(기계학습)
# 1. 지도 학습 : Y값(target)을 알고 있는 경우의 분석
# 1) 회귀 분석 : Y가 연속형
# 2) 분류 분석 : Y가 이산형(범주형)

# 2. 비지도 학습 : Y값(target)을 알고 있지 않은 경우의 분석
# 1) 군집 분석 : 데이터 축소 테크닉 (세분류)
# 2) 연관성 분석 : 장바구니 분석

# ex) 목적 : 종양의 양/악성 유무 판별 예측
# f(x) = y
# y : 양성, 악성
# x : 종양의 크기, 모양, 색깔, ...
# --------------------------------------------------------------------------- #
import mglearn


# [ 파이썬에서의 데이터 분석 환경 ]
# 1) numpy : 숫자 데이터의 빠른 연산 처리
# 2) pandas : 정형 데이터의 입력
# 3) scikit-learn : 머신러닝 데이터 분석을 위한 모듈 (샘플 데이터 제공, 알고리즘)
#                   anaconda에 포함된 모듈
# 4) scipy : 조금 더 복잡한 과학적 연산 처리 가능 (선형 대수 ...)
#            anaconda에 포함된 모듈
# 5) matplotlib : 시각화 (anaconda에 포함된 모듈)
# 6) mglearn : 외부 모듈, 분석에서의 북잡한 시각화 함수 제공
# c:\Users\harryjeong> pip install mglearn
# c:\Users\harryjeong> ipython

# [ Data Analysis Process - 분류분석, 비통계적]
# 0) Setting Purpose of Analysis
# 1) data loading (데이터 수집 - Y의 분류에 영향을 미칠 것 같은 X들을 수집)
# 2) preprocessing (데이터 전처리 - 이상치/결측치 제거 및 수정)
# 3) model selection based on data (모델 선택)
# 4) data split into train/test (데이터 분리)
# 5) model fitting by train data set (모델 학습)
# 6) model scoring by test data set (모델 평가)
# - 비통계적이기 때문에 모델평가가 어려움 => 몇개 중 몇개를 맞췄는지로 판단
# - 모델의 학습과 평가의 데이터셋은 분리시킴 => 같은 데이터 사용 시 확률 높아지는 문제발생
# 7) model tuning (모델 튜닝)
# 8) review & apply
# --------------------------------------------------------------------------- #

a1 = 1

# scikit-learn에서의 sample date 형식
from sklearn.datasets import load_iris

iris_dataset = load_iris()    # 딕셔너리 구조로 저장

iris_dataset.keys()

iris_dataset.data             # 설명변수 데이터 (array 형식)

iris_dataset.feature_names    # 각 설명변수 이름
feature selection             # feature(설명변수) 선택
iris_dataset.target           # 종속변수 데이터 (array 형식)
iris_dataset.target_names     # 종속변수 Y의 이름 (factor level name)
print(iris_dataset.DESCR)     # 데이터 설명

# [ 참고 : Y값의 학습 형태 ]
# ex) 이탈예측 : 이탈, 비이탈의 범주를 갖는 종속변수           
# Y : 이탈, 비이탈
# Y : 0, 1
# Y_0, Y_1 : 두 개 종속변수로 분리

      way1 | way2 : 더미변수로 Y값을 쪼갠 형태 (보통 뉴럴네트워크에서 사용)
Y      Y      Y_0(이탈)    Y_1(비이탈)
이탈    0         1            0
이탈    0         1            0
비이탈  1         0            1

# [ 분류분석 ]
# 1. knn (연속형에 적합)
# - 분류 모델 (지도학습)
# - 거리기반 모델
# - input data와 가장 거리가 가까운 k개의 관측치를 통해 input data의 Y값 결정
# - 이상치에 민감한 모델
# - 스케일링에 매우 민감 (변수의 스케일 표준화)
# - 학습되는 설명변수의 조합에 매우 민감
# - 내부 feature selection 기능 없음
# - 훈련 데이터 셋이 많아질수록 예측이 느린 경향 (학습이 느린 건 모든 분석이 그러함)
# - 설명변수가 많으면 가중치가 있지 않은 이상 좋지 않음

# knn (iris data) in python
# 1) 데이터 분리 (train/test)
X = iris_dataset.data
Y = iris_dataset.target

from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(X,                   # X data
                                                    Y,                   # Y data
                                                    train_size = 0.7,    # train set 추출 비율 (default 0.75)
                                                    random_state = 99)   # seed값 고정
# 위와 같이 처리하면 데이터를 순서대로 변수에 저장함

# =============================================================================
# # [ 참고 - random sampling 기법 for row number 추출 in R ]
# # import random
# # iris_nrow = iris_dataset.data.shape[0]
# # rn = random.sample(range(iris_nrow), round(iris_nrow * 0.7))
# # iris_train = iris_dataset.data[rn, ]
# # iris_test = DataFrame(iris_dataset.data).drop(rn, axis = 0).values
# =============================================================================

# =============================================================================
# # [ 참고 : sample 함수 ]
# random.sample([1, 20, 31, 49 10],    # 추출 대상
#                1)                    # 추출 개수 (정수)
# =============================================================================

# 2) 데이터 학습
from sklearn.neighbors import KNeighborsClassifier as knn_c
from sklearn.neighbors import KNeighborsRegressor as knn_r

m_knn = knn_c(5)
m_knn.fit(train_x, train_y)    # fitting 순간 해당 모델에 학습된 내용이 저장됨
m_knn.predict(test_x)          # m_knn.predict는 모델평가 및 예측 2가지에 사용

# 3) 평가
sum(m_knn.predict(test_x) == test_y) / test_x.shape[0] * 100    # 직접계산 => 97.78
m_knn.score(test_x, test_y)    # 함수활용 => 97.78

# 4) 예측
new_data = np.array([6.1, 6.9, 5.3, 1.9])     # 한줄이라 1개의 컬럼에 4개의 값이 있는 것처럼 인식함
m_knn.predict(new_data.reshape(1, -1))        # 1 x 4 shape임을 확실히 해줌 => 2

new_data1 = np.array([[6.1, 6.9, 5.3, 1.9]])  # 2차원으로 만들어 버림
m_knn.predict(new_data1)                      # 정상 출력 => 2

iris_dataset.target_names[2]
iris_dataset.target_names[m_knn.predict(new_data1)[0]]    # virginica

# 5) 튜닝
# k수 변화에 따른 train, test scroe 시각화 
score_tr = []
score_te = []
for i in range(1, 11) :
    m_knn = knn_c(i)
    m_knn.fit(train_x, train_y)
    score_tr.append(m_knn.score(train_x, train_y))
    score_te.append(m_knn.score(test_x, test_y))

import matplotlib.pyplot as plt
plt.plot(range(1, 11), score_tr, label = 'train_score')
plt.plot(range(1, 11), score_te, label = 'test_score', color = 'red')
plt.legend()

# 6) 데이터 시각화
import mglearn
df_iris = DataFrame(X, columns = iris_dataset.feature_names)

# 산점도
pd.plotting.scatter_matrix(df_iris,                     # X data set
                           c = Y,                       # Y data set (color 표현)
                           cmap = mglearn.cm3,          # cm = color mapping, cm1은 1개, cm2는 2개, cm3는 3개로 색 표현
                           marker = 'o',                # 산점도 점 모양
                           s = 60,                      # 점 크기
                           hist_kwds = {'bins':30})     # 히스토그램 인자 전달
# --------------------------------------------------------------------------- #