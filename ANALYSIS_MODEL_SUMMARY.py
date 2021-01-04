run profile1
# [ 분류분석 ]
# 1. knn (연속형에 적합)
# - 분류 모델 (지도학습)
# - 거리기반 모델
# - input data와 가장 거리가 가까운 k개의 관측치를 통해 input data의 Y값 결정
# - k개의 이웃이 갖는 정답(Y)의 평균 혹은 다수결로 최종 결론 내림
# - Y class의 개수의 배수가 되는 k수는 적절치 않음
# - 이상치에 민감한 모델 => 반드시 제거/수정
# - 스케일링에 매우 민감 (변수의 스케일 표준화)
# - 범주형 설명 변수가 많이 포함될수록 예측력 떨어짐 (거리화 시키기 어려움)
# - 학습되는 설명변수의 조합에 매우 민감
# - 고차원 데이터에 비적합
# - 내부 feature selection 기능 없음
# - 훈련 데이터 셋이 많아질수록 예측이 느린 경향 (대신 학습과정 생략, 데이터 들어올 때마다 계산)
# - 설명변수가 많으면 가중치가 있지 않은 이상 좋지 않음

# 0) Setting Purpose of Analysis

# 1) data loading (데이터 수집 - Y의 분류에 영향을 미칠 것 같은 X들을 수집)
# scikit-learn에서의 sample data 형식
from sklearn.datasets import load_iris
iris_dataset = load_iris()   # 딕셔너리 구조로 저장

iris_dataset.keys()

iris_dataset.data            # 설명변수 데이터(array 형식)
iris_dataset.feature_names   # 각 설명변수 이름
iris_dataset.target          # 종속변수 데이터(array 형식)
iris_dataset.target_names    # Y의 각 값의 이름(factor level name)
print(iris_dataset.DESCR)    # 데이터 설명

# 2) preprocessing (데이터 전처리 - 이상치/결측치 제거 및 수정)
# 3) model selection based on data (모델 선택)
# 4) data split into train/test (데이터 분리)
X = iris_dataset.data
Y = iris_dataset.target

from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(X,               # X data
                                                    Y,               # Y data
                                                    train_size=0.7,  # train set 추출 비율
                                                    random_state=99) # seed값 고정

# 5) model fitting by train data set (모델 학습)
from sklearn.neighbors import KNeighborsClassifier as knn_c
from sklearn.neighbors import KNeighborsRegressor as knn_r

m_knn = knn_c(5)             # 모델생성 => k는 거리계산할 주변 데이터 개수
m_knn.fit(train_x, train_y)  # 모델학습 using train data set => train data set으로 모델을 학습

# 6) model scoring by test data set (모델 평가)
# - 비통계적이기 때문에 모델평가가 어려움 => 몇개 중 몇개를 맞췄는지로 판단
# - 모델의 학습과 평가의 데이터셋은 분리시킴 => 같은 데이터 사용 시 확률 높아지는 문제발생
m_knn.predict(test_x)        # 모델예측 => 모델평가를 위해 test_x data를 바탕으로 test_y를 유추
m_knn.score(test_x, test_y)  # 모델평가 => 0.95, test data set을 모델에 넣어 예측력 확인

# 7) model tuning (모델 튜닝)
# 7-1) 매개변수 튜닝 => k수 변화에 따른 train, test score 시각화
score_tr = [] ; score_te = []

for i in range(1,11) :
    m_knn = knn_c(i)
    m_knn.fit(train_x, train_y)
    score_tr.append(m_knn.score(train_x, train_y))
    score_te.append(m_knn.score(test_x, test_y))

# *) 시각화 - 일반 선 그래프
import matplotlib.pyplot as plt
plt.plot(range(1,11), score_tr, label = 'train_score')
plt.plot(range(1,11), score_te, label = 'test_score', color = 'red')
plt.legend()
# => train이 test보다 높고, 간격이 가장 좁은 구간 찾기
# => k 값이 너무 커지면 성향이 다른 데이터와의 거리까지 계산하므로 예측력 떨어질 수 있음
# => 보통 3-5를 많이 사용하고 10을 초과하지 않음

# *) 시각화 - 교차산점도
import pandas as pd
from pandas import DataFrame
df_iris = DataFrame(X, columns = iris_dataset.feature_names)

import mglearn
pd.plotting.scatter_matrix(df_iris,               # X data set
                           c=Y,                   # Y data set(color표현) 
                           cmap = mglearn.cm3,    # color mapping 
                           marker='o',            # 산점도 점 모양
                           s=60,                  # 점 크기
                           hist_kwds={'bins':30}) # 히스토그램 인자 전달

# 8) review & apply

# --------------------------------------------------------------------------- #
# 2. 트리기반 모델
# [ 트리기반 모델 ]
DT -> RF -> GB -> XGB -> ... 
# GB => 이전 트리 보완해서 다음 모델 만드는 구조, 오분류된 데이터를 정분류하도록 함 (TBC)
# 이런 발전 과정에도 대기업들은 RF, GB 많이 사용 - scikit learn에 포함

# 2.1 Decision Tree (트리기반 모델)
# - 분류 분석을 수행하는 트리기반 모델의 가장 시초 모델
# - 패턴 학습이 단순하여 패턴을 시각화 할 수 있음
# - 패턴이 Tree 구조를 띔
# - 비통계적 모델이므로 모델 해석이 매우 용이
# - 단일 의사결정이므로 예측력이 불안하거나 과대적합 가능성 있음

# [ 트리 기반 모델의 변수 선택 기능 ]
# - 트리 모델은 각 변수의 중요도를 자식 노드의 불순도를 통해 계산
# - 중요도가 높은 변수를 상위 split 조건으로 사용
# - 변수간 중요도 차이가 큰 경우 특정 변수만 재사용 가능성 있음
# - 트리 모델의 변수 중요도는 분석 시 중요 변수 파악에 용이하게 사용

# [ 불순도 ]
# - 특정 조건으로 인해 분리된 자식노드의 클래스의 혼합 정도
# - 주로 지니계수로 측정
# - 2개의 class를 갖는경우 f(p) = p(1-p)로 계산

# 0) Setting Purpose of Analysis
# 1) data loading (데이터 수집 - Y의 분류에 영향을 미칠 것 같은 X들을 수집)
from sklearn.datasets import load_iris
df_iris = load_iris()

# 2) preprocessing (데이터 전처리 - 이상치/결측치 제거 및 수정)
# 3) model selection based on data (모델 선택)
# 4) data split into train/test (데이터 분리)
train_x, test_x, train_y, test_y = train_test_split(df_iris.data,
                                                    df_iris.target,
                                                    random_state = 0)

# 5) model fitting by train data set (모델 학습)
from sklearn.tree import DecisionTreeClassifier as dt_c
m_dt = dt_c()                 # 모델생성 => 매개변수 기본값
m_dt.fit(train_x, train_y)    # 모델학습
                              # min_samples_split = 2
                              # max_depth = None
                              # max_features = None

# 6) model scoring by test data set (모델 평가)
m_dt.score(test_x, test_y)    # 97.37   

# 7) model tuning (모델 튜닝)
# - min_samples_split : 각 노드별 오분류 개수가 몇개 이하일 때까지 쪼개라
#                       min_samples_split 값보다 오분류 개수가 크면 split
#                       min_samples_split 값이 작을수록 모델은 복잡해지는 경향
#                     - minbucket = round(minsplit / 3)
#                     - 추가 가지치기 매개변수 (추가로 가지치기를 할지 말지)
#                     - how? 오분류 개수 > minsbucket인 경우 추가 가지치기 진행
#                     - minbucket값 보다 오분류개수가 작아질 때까지 가지치기 계속 시도
#                     - 하지만 추가로 가지치기할 분류 기준이 더이상 없는 경우에는 stop
#                     - minbucket값이 작을수록 더 복잡한 모델의 생성 확률 높아짐 (예측력)
#                     - minbucket값이 작을수록 모델이 복잡해져 overfit 현상 발생 확률 증가 (과대적합)
#                     - 과대적합 : train 데이터에만 너무 적합해져 실제 데이터에서는 확률이 떨어지는 현상
# - max_features : fitting된 데이터는 같으나 그 데이터를 다르게 나눔 (복원추출 통해)
#                  각 노드의 고정시킬 설명변수의 후보의 개수
#                  max_features 클수록 서로 비슷한 트리로 구성될 확률 높아짐
#                  (설명력 가장 높은 변수가 선택될 가능성 높음)
#                  max_features 작을수록 서로 다른 트리로 구성될 확률 높아짐
#                  (복잡한 트리를 구성할 확률 높아짐)
# - max_depth : 설명변수의 중복 사용의 최대 개수
#               각 트리 내 노드의 분기 시 설명변수의 재사용 횟수 제한
#               max_depth 작을수록 단순한 트리를 구성할 확률이 높아짐
# - cp? 찾아보기 *

# 7-1) parameter tuning => 매개변수 튜닝
# *) 교차검증
from sklearn.model_selection import cross_val_score     
# cross validation(교차검증)의 목적은 평가점수의 일반화 => 매 검증마다 결과가 다르므로    

m_dt = dt_c(random_state = 0)
v_score = cross_val_score(m_dt,              # 적용모델
                          df_iris.data,      # 전체 설명변수 데이터 셋
                          df_iris.target,    # 전체 종속변수 데이터 셋
                          cv = 5)            # 교차검증 횟수 (tr 80%, te 20%)
# cv값이 너무 커지면 test data 비중이 너무 작아져서 좋지 않음 (3~5가 적당)
# 결과는 test data set에 대한 score
# array([0.96666667, 0.96666667, 0.9       , 0.96666667, 1.        ])
# 전체 데이터 안에서 알아서 tr, te로 나눠서 5번 진행
# 장점 : 교차검증으로 평가 일반화됨
# 단점 : train set score 확보하기 어려우므로 오버핏까지 고려한 매개변수 튜닝은 어려움

v_score.mean()    # 0.9600000000000002

score_te = []
for i in range(2, 11) :
    m_dt = dt_c(min_samples_split = i, random_state = 0)
    v_score = cross_val_score(m_dt, df_iris.data, df_iris.target, cv = 5)
    score_te.append(v_score.mean())

# [0.9600000000000002,
#  0.9666666666666668,
#  0.9666666666666668,
#  0.9666666666666668,
#  0.9666666666666668,
#  0.9666666666666668,
#  0.9666666666666668,
#  0.9666666666666668,
#  0.9666666666666668] => 이 데이터에서는 min_samples_split이 큰 의미가 없음

# 7-2) feature importance check => 특성 중요도 튜닝 (설명변수 중요도)
m_dt?    # 모델 자체가 가지고 있는 정보확인
m_dt.feature_importances_
# array([0.        , 0.02014872, 0.89994526, 0.07990602]) => 설명변수 4개의 중요도
df_iris.feature_names
# ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

s1 = Series(m_dt.feature_importances_, index = df_iris.feature_names)
s1.sort_values(ascending = False)

# *) 시각화 => DT가 시각화 가능한 거의 유일한 모델
#   - graphviz 설치 (window)
download 후 압축해제 (C:/Program Files (x86))
# download link : https://graphviz.gitlab.io/_pages/Download/Download_windows.html

#   - graphviz 설치 (파이썬)
C:\Users\harryjeong> pip install graphviz

pip install graphviz
conda install graphviz

#   - 파이썬 graphviz path 설정
import os
os.environ['PATH'] += os.pathsep + '/Users/harryjeong/Desktop/release/share/graphviz'

#   - 파이썬 시각화
import graphviz

from sklearn.tree import export_graphviz
export_graphviz(m_dt,                           # 모델명 
                out_file="tree.dot", 
                class_names = df_iris.target_names,
                feature_names = df_iris.feature_names, 
                impurity = False, 
                filled = True)

with open("tree.dot", encoding='UTF8') as f:
    dot_graph = f.read()

g1 = graphviz.Source(dot_graph)
g1.render('a1', cleanup=True) 

# 8) review & apply

# --------------------------------------------------------------------------- #
# 2.3 Ramdom Forest
# - Decision Tree의 과대적합 현상을 해결하기 위해 여러 개의 서로 다른
#   모양의 tree를 구성, 종합하여 최종 결론을 내는 방식
# - Random Forest Classifier와 Random Forest Regressor 존재
# - 분류모델인 경우는 다수결로, 회귀모델인 경우는 평균으로 최종결론

# 0) Setting Purpose of Analysis
# 1) data loading (데이터 수집 - Y의 분류에 영향을 미칠 것 같은 X들을 수집)
from sklearn.datasets import load_iris
df_iris = load_iris()

# 2) preprocessing (데이터 전처리 - 이상치/결측치 제거 및 수정)
# 3) model selection based on data (모델 선택)
# 4) data split into train/test (데이터 분리)
train_x, test_x, train_y, test_y = train_test_split(df_iris.data,
                                                    df_iris.target,
                                                    random_state = 0)

# 5) model fitting by train data set (모델 학습)
m_rf = rf_c(random_state = 0)
# => random_state를 고정해줘야 같은 데이터를 샘플링하기 때문에 매개변수 튜닝이 쉬움
m_rf.fit(train_x, train_y)    # n_estimators = 100, n_jobs = None
# n_jobs => 패러럴, 병렬처리 가능; 2개하면 2개 프로세스가 동시에 돌아감; cpu 많이 먹으므로 너무 많이 쓰지 말기

# 6) model scoring by test data set (모델 평가)
m_rf.score(test_x, test_y)    # 97.37

# 7) model tuning (모델 튜닝)
# 7-1) parameter tuning => 매개변수 튜닝
v_score_te = []

for i in range(1, 101) :
    m_rf = rf_c(random_state = 0, n_estimators = i)
    m_rf.fit(train_x, train_y)
    v_score_te.append(m_rf.score(test_x, test_y))

# *) 시각화 - 일반 선 그래프
import matplotlib.pyplot as plt    
plt.plot(np.arange(1, 101), v_score_te, color = 'red')    # 5이상이면 충분히 높게 나옴

# 7-2) feature importance check => 특성 중요도 튜닝 (설명변수 중요도)
m_rf.base_estimator_
m_rf.feature_importances_
# array([0.10749462, 0.02616594, 0.42160356, 0.44473587]) => DT와 결과가 다름 -> 여러 가지 방식으로 중요도 체크해야 함

# *) 시각화 - 변수중요도 체크 가로막대 그래프
def plot_feature_importances_cancer(model, data) : 
    n_features = data.data.shape[1]    # 컬럼 사이즈
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), data.feature_names)    # y 눈금
    plt.xlabel("특성 중요도")    # x축 이름
    plt.ylabel("특성")         # y축 이름
    plt.ylim(-1, n_features)

plt.rc('font', family = 'Apple Gothic')
plot_feature_importances_cancer(m_rf, df_cancer)

# 8) review & apply

# --------------------------------------------------------------------------- #
# 4. Gradiant Boosting Tree(GB)
# - 이전 트리의 오차를 보완하는 트리를 생성하는 구조
# - 비교적 단순한 초기 트리를 형성, 오분류 data point에 더 높은 가중치를 부여, 오분류 data point
#   를 정분류 하도록 더 보완된, 복잡한 트리를 생성
# - learning rate 만큼의 오차 보완률 결정 (0 ~ 1, 높을수록 과적합 발생 가능성 높음)
# - random forest 모델보다 더 적은 수의 tree로도 높은 예측력을 기대할 수 있음
# - 각 트리는 서로 독립적이지 않으므로 (이전 트리가 끝나야 다음 트리 시작)
#   병렬처리에 대한 효과를 크게 기대하기 어려움

# 0) Setting Purpose of Analysis
# 1) data loading (데이터 수집 - Y의 분류에 영향을 미칠 것 같은 X들을 수집)
from sklearn.datasets import load_iris
df_iris = load_iris()

# 2) preprocessing (데이터 전처리 - 이상치/결측치 제거 및 수정)
# 3) model selection based on data (모델 선택)
# 4) data split into train/test (데이터 분리)
train_x, test_x, train_y, test_y = train_test_split(df_iris.data,
                                                    df_iris.target,
                                                    random_state = 0)

# 5) model fitting by train data set (모델 학습)
from sklearn.ensemble import GradientBoostingClassifier as gb_c
from sklearn.ensemble import GradientBoostingRegressor as gb_r

m_gb = gb_c()
m_gb.fit(train_x, train_y)    # learning_rate = 0.1, max_depth = 3,
                              # max_features = None, min_samples_split = 2,
                              # n_estimators = 100

# 6) model scoring by test data set (모델 평가)
m_gb.score(test_x, test_y)    # 97.37 

# 7) model tuning (모델 튜닝)
# 7-1) parameter tuning => 매개변수 튜닝
vscore_tr = [] ; vscore_te = []

for i in [0.001, 0.01, 0.1, 0.5, 1] :
    m_gb = gb_c(learning_rate = i)
    m_gb.fit(train_x, train_y)
    vscore_tr.append(m_gb.score(train_x, train_y))
    vscore_te.append(m_gb.score(test_x, test_y))

# *) 시각화
plt.plot([0.001, 0.01, 0.1, 0.5, 1], vscore_tr, label = 'train_score')
plt.plot([0.001, 0.01, 0.1, 0.5, 1], vscore_te, label = 'test_score', color = 'red')
plt.legend()    # 낮은 learning rate에서도 충분한 예측력 나옴

# 7-2) feature importance check => 특성 중요도 튜닝 (설명변수 중요도)
def plot_feature_importances(model, data) : 
    n_features = data.data.shape[1]    # 컬럼 사이즈
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), data.feature_names)    # y 눈금
    plt.xlabel("특성 중요도")    # x축 이름
    plt.ylabel("특성")         # y축 이름
    plt.ylim(-1, n_features)

plot_feature_importances(m_gb, df_iris) 

# 8) review & apply
# --------------------------------------------------------------------------- #

# [ 분석 시 고려 사항 ]
# 1. 변수 선택
# 2. 변수 변형 (결합 포함)
# 3. 교호작용 (interaction)
# 4. 교차검증 (cross validation)
# 5. 최적의 매개변수 조합 (grid search) : train/val/test
# 6. 변수 표준화 (scaling)

# [ 분석 시 고려사항 6. 스케일링 ]
# - 설명변수의 서로 다른 범위를 동일한 범주내 비교하기 위한 작업
# - 거리기반 모델, 회귀계수의 크기 비교, NN의 모델 등에서 필요로 하는 작업
# - 각 설명변수의 중요도를 정확히 비교하기 위해서도 요구되어짐
# - 특히 interaction 고려 시 결합되는 설명변수끼리의 범위를 동일하게 만들 필요 있음

# 6.1 scaling 종류
# 1) standard scaling : 변수를 표준화 하는 작업 (평균 0, 표준편차 1)
#    표준화 : (X - 평균) / 표준편차
from sklearn.preprocessing import StandardScaler as standard
    
# 2) minmax scaling : 최소값을 0으로, 최대값을 1로 만드는 작업
from sklearn.preprocessing import MinMaxScaler as minmax

# 6.2 scaling 생성
# 순서 : scaling 모델생성 - 모델학습 - scaling 생성
# 1) standard scaling
m_sc1 = standard()        # scaling 모델생성
m_sc1.fit(train_x)        # 모델학습 => 각 설명변수의 평균, 표준편차 계산
m_sc1.transform(train_x)  # scaling 생성 => 위 값 바탕으로 표준화 시킴
m_sc1.transform(test_x)   # scaling 생성 => train값을 scaling하면 test값도 scaling 해야 함

m_sc1.transform(train_x).mean(axis = 0)    # 사실상 0이라 보는게 맞음
m_sc1.transform(train_x).std(axis = 0)     # 1

# 2) minmax scaling
m_sc2 = minmax()
m_sc2.fit(train_x)        # 각 설명변수의 최대, 최소 구하기
m_sc2.transform(train_x)  # 최소를 0, 최대를 1에 맞춰 계산
m_sc2.transform(test_x)

m_sc2.transform(train_x).min(axis = 0)    # 0
m_sc2.transform(train_x).max(axis = 0)    # 1

# 6.3 test => scaling + knn
train_x_sc = m_sc1.transform(train_x)
test_x_sc = m_sc1.transform(test_x)

# 1) scaling 전
m_knn = knn_c(5)
m_knn.fit(train_x, train_y)
m_knn.score(test_x, test_y)    # 97

# 2) scaling 후
m_knn = knn_c(5)
m_knn.fit(train_x_sc, train_y)
m_knn.score(test_x_sc, test_y)  # 97

# 3) 중요도 높은 변수 선택 및 scaling 후
m_knn = knn_c(5)
m_knn.fit(train_x_sc[:, 2:4], train_y)
m_knn.score(test_x_sc[:, 2:4], test_y)

# =============================================================================
# # [ 참고 : minmax scaling 방식의 비교 ]
# m_sc2 = minmax()
# m_sc2.fit(train_x)
# 
# m_sc2.fit_transform(train_x)   # fit과 transform 동시에 처리
# 
# m_sc3 = minmax()
# m_sc3.fit(test_x)
# 
# # - 올바른 scaling => 기준 같음 (train set fit)
# train_x_sc1 = m_sc2.transform(train_x)
# test_x_sc1 = m_sc2.transform(test_x)
# 
# # - 잘못된 scaling => 기준 다름 (train -> train fit, test -> test fit)
# train_x_sc2 = m_sc2.transform(train_x)
# test_x_sc2 = m_sc3.transform(test_x)
# =============================================================================

# 4) 시각화
# 4-1) figure와 subplot 생성 (1 x 3)
fig, ax = plt.subplots(1, 3)

# 4-2) 원본의 분포(x1, x2)의 산점도
import mglearn

plt.rc('font', family = 'Malgun Gothic')
ax[0].scatter(train_x[:, 0], train_x[:, 1], c = mglearn.cm2(0), label = 'train')
ax[0].scatter(test_x[:, 0], test_x[:, 1], c = mglearn.cm2(1), label = 'test')
ax[0].legend()
ax[0].set_title('원본 산점도')
ax[0].set_xlabel('sepal ')
ax[0].set_ylabel('sepal ')

# 4-3) 올바른 scaling 후 데이터(x1, x2)의 산점도
ax[1].scatter(train_x_sc1[:, 0], train_x_sc1[:, 1], c = mglearn.cm2(0))
ax[1].scatter(test_x_sc1[:, 0], test_x_sc1[:, 1], c = mglearn.cm2(1))

# 4-4) 잘못된 scaling 후 데이터(x1, x2)의 산점도
ax[2].scatter(train_x_sc2[:, 0], train_x_sc2[:, 1], c = mglearn.cm2(0))
ax[2].scatter(test_x_sc2[:, 0], test_x_sc2[:, 1], c = mglearn.cm2(1))

# => train data set과 test data set이 분리되어진 상태일 경우 각각 서로 다른 기준으로 scaling
#    을 진행하면 (3번째 subplot) 원본의 데이터와 산점도가 달라지는 즉, 원본의 데이터의 왜곡이 발생
# => 따라서 같은 기준으로 train/test를 scaling하는 것이 올바른 scaling 방식!

# --------------------------------------------------------------------------- #

# [ 분석 시 고려사항 3. 교호작용 ]
# - 변수 상호 간 서로 결합된 형태로 의미 있는 경우
# - 2차, 3차항 ... 추가 가능
# - 발생 가능한 모든 다차항의 interaction으로 부터 의미 있는 변수 추출
# - scaling된 변수로 하기

# 3.1 interaction 적용 변수생성   
from sklearn.preprocessing import PolynomialFeatures as poly

# 원본         => 2차항 적용 (transform 작업)
# x1 x2 x3       x1^2 x2^2 x3^2 x1x2 x1x2 x2x3
# 1  2  3        1    4    9    2    3    6
# 2  4  5        4    16   25   8    10   20

m_poly = poly(degree = 2)    # 2차항을 만들겠다
m_poly.fit(train_x_sc)          # 각 설명변수에 2차항 모델 생성
# ** test data set은 fitting 필요 없음 why?
train_x_sc_poly = m_poly.transform(train_x_sc)    # 스케일링 된 데이터셋으로 하는게 더 좋음
test_x_sc_poly = m_poly.transform(test_x_sc)

m_poly.get_feature_names()   # 변경된 설명변수들의 형태 = 2차항 모습

DataFrame(m_poly.transform(train_x_sc),
          columns = m_poly.get_feature_names())  # 보기 좋음 -> 변수가 엄청 많으면 이 또한 쉽진 않음

col_poly = m_poly.get_feature_names(df_iris.feature_names)  # 실제 컬럼이름이 반영된 교호작용 출력

DataFrame(m_poly.transform(train_x_sc),
          columns = m_poly.get_feature_names(df_iris.feature_names)) # 훨씬 보기 좋음

# 3.3 선택된 interaction 학습
# => 확장된 데이터셋을 RF에 학습, feature importance 확인 
m_rf = rf_c(random_state = 0)
m_rf.fit(train_x_sc_poly, train_y)

# => 대부분은 설명변수 증가하면 예측력 높아지지만, Tree기반 모델은 어차피 정하는 변수개수가 있으므로 큰 의미 없어짐
# => knn모델은 효과 있음

# 위 내용 바탕 knn에 적용
s1 = Series(m_rf.feature_importances_, index = col_poly)
s1.sort_values(ascending = False)

train_x_sc_poly_sel = DataFrame(train_x_sc_poly, columns = col_poly).iloc[:, :6]
test_x_sc_poly_sel = DataFrame(test_x_sc_poly, columns = col_poly).iloc[:, :6]

m_knn = knn_c(5)
m_knn.fit(train_x_sc_poly_sel, train_y)
m_knn.score(test_x_sc_poly_sel, test_y)    # 0.973

## 전진 선택법 (변수 하나씩 추가)
l1 = s1.sort_values(ascending = False).index

collist = []
df_result = DataFrame()

for i in l1 :
    collist.append(i)
    train_x_sc_poly_sel = DataFrame(train_x_sc_poly, columns = col_poly).loc[:, collist]
    test_x_sc_poly_sel = DataFrame(test_x_sc_poly, columns = col_poly).loc[:, collist]
    
    m_knn1 = knn_c(5)
    m_knn1.fit(train_x_sc_poly_sel, train_y)
    vscore = m_knn1.score(test_x_sc_poly_sel, test_y)
    
    df1 = DataFrame([Series(collist).str.cat(sep = '+'), vscore], index = ['column_list', 'score']).T
    df_result = pd.concat([df_result, df1], ignore_index = True)
    
df_result.sort_values(by = 'score', ascending = False)
# --------------------------------------------------------------------------- #

# [ 분석 시 고려사항 2. 변수 변형 (결합 포함) ]
# 2.1 종속 변수의 변형 (NN 기반 모델일 경우 필수)
# - 종속변수가 범주형일때 하나의 종속변수를 여러 개의 종속변수로 분리시키는 작업
# - 모델에 따라 문자형태의 변수의 학습이 불가할 경우 종속변수를 숫자로 변경
# - NN에서는 주로 종속변수의 class의 수에 맞게 종속변수를 분리
# - 0과 1의 숫자로만 종속변수를 표현

# ex)
# Y   Y_남  Y_여
# 남    1     0
# 여    0     1
# 여    0     1 

df1 = DataFrame({'col1':['M','M','F','F'],
                 'col2':[98,90,96,95]})

pd.get_dummies(df1)                           # 숫자 변수는 분할 대상 X
pd.get_dummies(df1, columns=['col1','col2'])  # 숫자 변수 강제 분할
pd.get_dummies(df1, drop_first=True)          # Y의 개수가 class -1개 분리

# Y  Y_A  Y_B  Y_C
# A   1    0    0
# B   0    1    0
# C   0    0    1
    
# Y  Y_A  Y_B  
# A   1    0   
# B   0    1    
# C   0    0     

# 2.2 변수 선택(feature selection)
# - 모델 학습 전 변수를 선택하는 과정
# - 트리기반, 회귀 모델 자체가 변수 선택하는 기준을 제시하기도 함
# - 거리기반, 회귀기반 모델들은 학습되는 변수에 따라 결과가 달라지므로
#   사전에 최적의 변수의 조합을 찾는 과정이 중요
# - 트리기반, NN기반 모델들은 내부 변수를 선택하는 과정이 포함,
#   다른 모델들에 비해 사전 변수 선택의 중요도가 낮음
  
# 2.2.1 모델 기반 변수 선택
# - 트리, 회귀 기반 모델에서의 변수 중요도를 사용하여 변수를 선택하는 방식
# - 트리에서는 변수 중요도를 참고, 회귀에서는 각 변수의 계수 참고
# - 모델에 학습된 변수끼리의 상관 관계도 함께 고려 (종합적 판단)

from sklearn.feature_selection import SelectFromModel

# 예제) SelectFromModel iris data
from sklearn.datasets import load_iris
df_iris = load_iris()

# 1) noise 변수 추가(10개)
vrandom = np.random.RandomState(0)
vcol = vrandom.normal(size = (len(df_iris.data), 10))  # 150 X 10
vrandom.normal?
df_iris_new = np.hstack([df_iris.data, vcol])    # df_iris.data에 vcol 추가
df_iris_new.shape   # 150 X 14

# =============================================================================
# # [ 참고 - array의 컬럼, 행 추가 ]
# np.hstack : 두 array의 가로방향 결합(컬럼 추가)
# np.vstack : 두 array의 세로방향 결합(행 추가)
# 
# arr1 = np.arange(1,10).reshape(3,3)
# arr2 = np.arange(1,91,10).reshape(3,3)
# 
# np.hstack([arr1,arr2])    # 3 X 6
# np.vstack([arr1,arr2])    # 6 X 3
# =============================================================================

# 2) 확장된 dataset을 SelectFromModel에 적용
m_rf = rf_c()
m_select1 = SelectFromModel(m_rf,               # 변수 중요도를 파악할 모델 명 전달 = RF
                            threshold='median') # 선택 범위
 
m_select1.fit(df_iris.data, df_iris.target)
m_select1.get_support()    # 중요변수 보여줌

m_select1.fit(df_iris_new, df_iris.target)
m_select1.get_support()    # 중요변수 보여줌

# 3) 선택된 변수의 dataset 추출
df_iris_new[:, m_select1.get_support()]     # 중요변수 선택 후 dataset
m_select1.transform(df_iris_new)            # 위와 같음

# 4) 변수 중요도 확인
m_select1.estimator_.feature_importances_

# 2.2.2 변수선택 방법 2 : 일변량 통계 기법
# - 변수 하나와 종속변수와의 상관 관계 중심으로 변수 선택
# - 다른 변수가 함께 학습될 때의 판단과는 다른 결과가 나올 수 있음
# - 학습 시킬 모델이 필요 없어 연산속도가 매우 빠름
from sklearn.feature_selection import SelectPercentile
  
# 1) 변수 선택 모델 생성 및 적용
m_select2 = SelectPercentile(percentile=30)    # percentile => 상위 몇 % 뽑을지?
m_select2.fit(df_iris_new, df_iris.target)

# 2) 변수 선택 결과 dataset 확인
m_select2.get_support()
df_iris_new[:, m_select2.get_support()]
m_select2.transform(df_iris_new)    # 위와 같음

# 3) 변수 선택 시각화
plt.matshow(m_select2.get_support().reshape(1,-1), cmap='gray_r')

# 2.2.3 변수선택 방법 3 : 반복적 선택(RFE)
# - step wise 기법과 유사
# - 전체 변수를 학습 시킨 후 가장 의미 없는 변수 제거,
#   반복하다 다시 변수 추가가 필요한 경우 추가하는 과정
# - 특성의 중요도를 파악하기 위한 모델 필요
from sklearn.feature_selection import RFE

# [ RFE iris data ]
m_rf = rf_c()
m_select3 = RFE(m_rf, n_features_to_select = 2)  # 개수 기반 선택 가능
m_select3.fit(df_iris.data, df_iris.target)

m_select3.get_support()
m_select3.ranking_                          # 전체 특성 중요도 순서 확인
m_select3.estimator_.feature_importances_   # 선택된 특성 중요도 값 확인
# --------------------------------------------------------------------------- #

# [ 분석 시 고려사항 5. 최적의 매개변수 조합 (grid search) : train/val/test ]
# - 변수의 최적의 조합을 찾는 과정
# - 중첩 for문으로 구현 가능, grid search 기법으로 간단히 구현 가능
# - train/validation/test set으로 분리
# - 매개변수의 선택은 validation set으로 평가
# => Grid는 for문 돌리지 않고도 알아서 진행해주는 것은 장점이지만, fit시킨 순간 매개변수가 고정되서
#    test data set scoring 할 때 매개변수 변화 불가 (이미 세팅된 매개변수로만 가능)
#    -> train data set과 비교 불가하여 overfit 확인 불가

# 예제) grid search by 중첩 for문 - knn iris data
# 1) data loading
# 2) data split
trainval_x, test_x, trainval_y, test_y = train_test_split(df_iris.data,
                                                          df_iris.target,
                                                          random_state=0)

train_x, val_x, train_y, val_y = train_test_split(trainval_x,
                                                  trainval_y,
                                                  random_state=0)

# 3) 모델 학습 및 매개변수 튜닝
best_score = 0
for i in range(1,11) :
    m_knn = knn_c(i)
    m_knn.fit(train_x, train_y)
    vscore = m_knn.score(val_x, val_y)
    
    if vscore > best_score :
        best_score = vscore    # best_score의 갱신
        best_params = i        # best parameter의 저장

# 4) 매개변수 고정 후 재학습 및 평가
m_knn = knn_c(best_params)  
m_knn.fit(trainval_x, trainval_y)
m_knn.score(test_x, test_y)    # 0.973, best parameter = 5

# 예제) grid search - random forest cancer data set
# 1) data loading
from sklearn.datasets import load_breast_cancer
df_cancer = load_breast_cancer()
df_cancer.data.shape    # (569, 30)

# 2) data split
trainval_x, test_x, trainval_y, test_y = train_test_split(df_cancer.data,
                                                          df_cancer.target,
                                                          random_state = 0)

train_x, val_x, train_y, val_y = train_test_split(trainval_x,
                                                  trainval_y,
                                                  random_state = 0)

# 3) 모델 학습 및 매개변수 튜닝 (min_samples_split, max_features)
# min_samples_split : 2 ~ 10
# max_features : 1 ~ 30
# 3-1) 교차 검증 수행 x => 9 * 30
best_score = 0
for i in range(2, 11) : 
    for j in range(1, 31) : 
        m_rf = rf_c(min_samples_split = i, max_features = j, random_state = 0)
        m_rf.fit(train_x, train_y)
        vscore = m_rf.score(val_x, val_y)
        
        if vscore > best_score :
            best_score = vscore
            best_params = {'min_samples_split' : i,
                           'max_features' : j}
        
m_rf = rf_c(**best_params, random_state = 0)    # ** 사용하면 키워드 인자 넣을 수 있음
m_rf.fit(trainval_x, trainval_y)
m_rf.score(test_x, test_y)    # 0.972

# 3-2) 교차 검증 수행 (매개변수 튜닝 시) => 9 * 30 * 5
best_score = 0
for i in range(2, 11) : 
    for j in range(1, 31) : 
        m_rf = rf_c(min_samples_split = i, max_features = j, random_state = 0)
        m_rf.fit(train_x, train_y)
        cr_vscore = cross_val_score(m_rf, trainval_x, trainval_y, cv = 5)
        
        vscore = cr_vscore.mean()
        if vscore > best_score :
            best_score = vscore
            best_params = {'min_samples_split' : i,
                           'max_features' : j}
        
m_rf = rf_c(**best_params, random_state = 0)    # ** 사용하면 키워드 인자 넣을 수 있음
m_rf.fit(trainval_x, trainval_y)
m_rf.score(test_x, test_y)    # 0.944

# grid search 기법
# - 위의 중첩 for문을 사용한 매개변수의 조합을 찾는 과정을 함수화
# - CV 기법을 포함시켜 validation data set에 대한 반복 test를 수행

from sklearn.model_selection import GridSearchCV

# 1) 모델 생성
m_rf = rf_c()

# 2) 그리드 서치 기법을 통한 매개변수 조합 찾기
# 2-1) 매개변수 조합 생성
v_params = {'min_samples_split' : np.arange(2, 11),
            'max_features' : np.arange(1, 31)}

# 2-2) 그리드 서치 모델 생성
m_grid = GridSearchCV(m_rf,           # 적용 모델
                      v_params,       # 매개변수 조합 (딕셔너리)
                      cv = 5)

# 2-3) 그리드 서치에 의한 모델 학습
m_grid.fit(trainval_x, trainval_y)

# 2-4) 결과 확인
dir(m_grid)
m_grid.best_score_    # 베스트 매개변수 값을 갖는 평가 점수 => 0.962
m_grid.best_params_   # {'max_features': 1, 'min_samples_split': 5}

df_result = DataFrame(m_grid.cv_results_)
df_result.T    # 여기서 test_score들은 우리가 아는 test score가 아니고, validation score라고 생각하면 됨

df_result.T.iloc[:, 0]    # 첫 번째 매개변수 셋 결과

# cv = 5이기 때문에 아래처럼 5회 진행
# split0_test_score                                              0.94186
# split1_test_score                                             0.952941
# split2_test_score                                             0.929412
# split3_test_score                                             0.988235
# split4_test_score                                             0.988235

# 2-5) 최종 평가
m_grid.score(test_x, test_y)    # 0.951 => 모델 자체에 best params가 저장되어 있음

# 2-6) 그리드 서치 결과 시각화
df_result.mean_test_score    # 교차 검증의 결과 (5개의 점수에 대한 평균)
arr_score = np.array(df_result.mean_test_score).reshape(30, 9)    
# reshape(max_features, min_samples) => reshape은 행우선 순위로 배치 되기 때문에
df_result.loc[:, ['params', 'mean_test_score']]

# =============================================================================
# # [ 참고 : df_result의 결과 reshape 시 배치 순서 ]
# #                             min_samples_split
# #                         2          3          4          5      ...     10
# # 'max_features': 1   0.960137   0.953133   0.960109   0.955486   
# # 'max_features': 2
# #                     ....
# # 'max_features': 30
# =============================================================================

import mglearn
plt.rc('figure', figsize = (10, 10))
plt.rc('font', size = 7)

mglearn.tools.heatmap(value,        # 숫자 배열
                      xlabel,       # x축 이름
                      ylabel,       # y축 이름
                      xticklabels,  # x축 눈금
                      yticklabels)  # y축 눈금

mglearn.tools.heatmap(arr_score,                
                      'min_samples_split',  
                      'max_features',     
                      v_params['min_samples_split'],
                      v_params['max_features'],
                      cmap = 'viridis')
# --------------------------------------------------------------------------- #

# 5. SVM (Support Vector Machine)
# - 분류 기준을 회귀분석처럼 선형선, 혹은 초평면(다차원)을 통해 찾는 과정
# - 다차원(고차원) 데이터셋에 주로 적용
# - 초평면을 만드는 과정이 매우 복잡, 해석 불가 (black box 모델)
# - c(비선형성 강화), gamma(고차원선 강화)의 매개변수의 조합이 매우 중요 (상호 연관적)
# - 계수를 추정하는 방식이 회귀와 유사, 학습 전 변수의 scaling 필요
# - 이상치에 민감
# - 초기 분류기준으로부터 support vector에 가중치를 부여, 분류기준 강화 하는 과정
# - 지나치게 train 데이터에 맞추면 안됨 -> overfit
# - 과거에는 자주 쓰였으나 요즘에는 NN가 대부분을 대체함
# - 지지벡터 : 초평면을 그렸을 때 오분류된 데이터들 -> 얘네 기준으로 선을 확실히 확인 가능

# SVM cancer data in python
from sklearn.svm import SVC

# 1) data loading
from sklearn.datasets import load_breast_cancer
df_cancer = load_breast_cancer()

# 2) 데이터 분리
train_x, test_x, train_y, test_y = train_test_split(df_cancer.data,
                                                    df_cancer.target,
                                                    random_state = 0)

# 3) 모델 생성 및 학습
m_svm = SVC()
m_svm.fit(train_x, train_y)    # C = 1.0, gamma = 'scale'

# 4) 모델 평가
m_svm.score(test_x, test_y)    # 0.937

# 5) 변수 스케일링 후 적용
from sklearn.preprocessing import StandardScaler as standard
m_sc1 = standard()

m_sc1.fit(train_x)
train_x_sc = m_sc1.transform(train_x)
test_x_sc = m_sc1.transform(test_x)

m_svm2 = SVC()
m_svm2.fit(train_x_sc, train_y)    # C = 1.0, gamma = 'scale'
m_svm2.score(test_x_sc, test_y)    # 0.965

# 6) 매개변수 튜닝
v_parameter = {'C':[0.001, 0.01, 0.1, 1, 10, 100, 1000],
               'gamma':[0.001, 0.01, 0.1, 1, 10, 100, 1000]}

m_svm = SVC()
m_grid = GridSearchCV(m_svm, v_parameter, cv = 5)
m_grid.fit(train_x_sc, train_y)
m_grid.score(test_x_sc, test_y)    # 0.979
m_grid.best_params_
df_result = DataFrame(m_grid.cv_results_)

df_result.loc[:, ['params', 'mean_test_score']]

arr_score = np.array(df_result.mean_test_score).reshape(7, 7)
arr_score.max()    # 0.985 => training data set score
import mglearn
mglearn.tools.heatmap(arr_score,
                      'gamma',
                      'C',
                      v_parameter['gamma'],
                      v_parameter['C'],
                      cmap = 'viridis')

# => Grid는 for문 돌리지 않고도 알아서 진행해주는 것은 장점이지만, fit시킨 순간 매개변수가 고정되서
#    test data set scoring 할 때 매개변수 변화 불가 (이미 세팅된 매개변수로만 가능)
#    -> train data set과 비교 불가하여 overfit 확인 불가

# overfit 확인
v_para_c = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
v_para_gamma = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

v_score_tr = [] ; v_score_te = [] ; v_c = [] ; v_gamma = []
for i in v_para_c : 
    for j in v_para_gamma :
        m_svc = SVC(C = i, gamma = j)
        m_svc.fit(train_x_sc, train_y)
        v_score_tr.append(m_svc.score(train_x_sc, train_y))
        v_score_te.append(m_svc.score(test_x_sc, test_y))
        v_c.append(i)
        v_gamma.append(j)

df_result2 = DataFrame({'C':v_c,
                        'gamma':v_gamma,
                        'train_score':v_score_tr,
                        'test_score':v_score_te})

df_result2.loc[:, ['train_score', 'test_score']].plot()
# 위 그래프는 x축에 c와 gamma의 수치가 결합되어 들어가 있음

# => x축 눈금 표현 ('C':0.001 'gamma':0.001)
# way 1)
f1 = lambda x, y : 'C:' + str(x) + ', ' + 'gamma:' + str(y)

df_result2.C.map(f1, df_result2.gamma)    # Error
l1 = list(map(f1, df_result2.C, df_result2.gamma))

# way 2)
f2 = lambda x, y : 'C:' + str(x) + ', ' + 'gamma:' + str(y)
df_result2.apply(f2, axis = 1)    # Error => y를 알 수 없음

f3 = lambda x : 'C:' + str(x[0]) + ', ' + 'gamma:' + str(x[1])
df_result2.apply(f3, axis = 1)

plt.xticks(df_result2.index,    # x축 눈금 (숫자)
           l1,                  # 각 눈금의 이름 (문자)
           rotation = 270,      # 축 이름 출력 방향
           fontsize = 6)

# --------------------------------------------------------------------------- #

# [ 분류분석 활용 1 : PCA + knn ]
# 1. PCA(Principal Component Analysis : 주성분 분석)
# - 비지도 학습
# - 기존의 변수로 새로운 인공변수를 유도하는 방식 (변수 결합)
# - 유도된 인공변수끼리 서로 독립적
# - 첫번째 유도된 인공변수가 기존 데이터의 분산을 가장 많이 설명하는 형식
# - 회귀의 다중공선성 문제 해결
# - 기존 데이터를 모두 사용, 저차원 모델 생성 가능 => 과대적합 해소
# - 의미 있는 인공변수 유도 => 설명변수를 추가할수록 좋음
# - 변수 scaling 필요

# Y = X1 + X2 + X3 + X4 + X5               # 다중공선성이 예상됨 => 인공변수를 넣자
# C1 = a1X1 + a2X2 + a3X3 + a4X4 + a5X5    # 가장 설명력이 높음
# C2 = b1X1 + b2X2 + b3X3 + b4X4 + b5X5    # 그다음 설명력이 높음

# Y = c1C1 + c2C2    # PCA + regressor 

# 1.1 PCA for iris data
# 1) data loading
from sklearn.datasets import load_iris
df_iris = load_iris()

train_x, test_x, train_y, test_y = train_test_split(df_iris.data,
                                                    df_iris.target,
                                                    random_state = 0)

# 2) 인공변수 유도
from sklearn.decomposition import PCA
m_pca = PCA(n_components = 2)              # 인공변수는 2개
m_pca.fit(train_x)                         # y를 알고 있는 분류분석임에도 y없이 진행
train_x_pca = m_pca.transform(train_x)     # (112, 4)
test_x_pca = m_pca.transform(test_x) 

# 3) 인공변수 확인
m_pca.components_                          # 유도된 인공변수의 계수

# array([[ 0.37649644, -0.06637905,  0.85134571,  0.35924188],
#        [ 0.6240207 ,  0.75538031, -0.18479376, -0.07648543]])

C1 = 0.37649644 * X1 + -0.06637905 * X2 + 0.85134571 * X3 + 0.35924188 * X4
C2 = 0.6240207 * X1 + 0.75538031 * X2 + -0.18479376 * X3 + -0.07648543 * X4

# 1.2 PCA + knn (앙상블) for iris data

# 4) 유도된 인공변수 knn 모델에 적용
m_knn = knn_c(5)
m_knn.fit(train_x_pca, train_y)
m_knn.score(test_x_pca, test_y)    # 0.973

# 5) data point들의 분포 확인 (산점도)
import mglearn
mglearn.discrete_scatter(train_x_pca[:, 0], train_x_pca[:, 1], train_y)

# 1.3 PCA + SVM 적용 for cancer data
# 1. data loading
from sklearn.datasets import load_breast_cancer
df_cancer = load_breast_cancer()

train_x, test_x, train_y, test_y = train_test_split(df_cancer.data,
                                                    df_cancer.target,
                                                    random_state=0)

# 2. 변수 스케일링
m_stand = standard()
m_stand.fit(train_x)
train_x_sc = m_stand.transform(train_x)
test_x_sc  = m_stand.transform(test_x)

# 3. 인공변수 유도
m_pca1 = PCA(2)
m_pca2 = PCA(3)

m_pca1.fit(train_x_sc)
m_pca2.fit(train_x_sc)

train_x_sc_pca1 = m_pca1.transform(train_x_sc)
test_x_sc_pca1 = m_pca1.transform(test_x_sc)

train_x_sc_pca2 = m_pca2.transform(train_x_sc)
test_x_sc_pca2 = m_pca2.transform(test_x_sc)

# 4. SVM 모델 적용
v_parameter = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
               'gamma': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

m_svm = SVC()
m_grid = GridSearchCV(m_svm, v_parameter, cv=5)

m_grid.fit(train_x_sc_pca1, train_y)
m_grid.score(test_x_sc_pca1, test_y)   # 92.30

m_grid.fit(train_x_sc_pca2, train_y)
m_grid.score(test_x_sc_pca2, test_y)   # 92.30

# 5. 시각화
# 1) 2개 인공변수(2차원)
import mglearn
mglearn.discrete_scatter(train_x_sc_pca1[:,0],  # X축
                         train_x_sc_pca1[:,1],  # Y축
                         train_y)               # target 값에 따른 색 표현

# 2) 3개 인공변수(3차원)
from mpl_toolkits.mplot3d import Axes3D, axes3d
figure = plt.figure()
ax = Axes3D(figure)
    
ax.scatter(train_x_sc_pca2[train_y == 0 , 0],   # X축 좌표
           train_x_sc_pca2[train_y == 0, 1],   # Y축 좌표
           train_x_sc_pca2[train_y == 0, 2],   # Z축 좌표
           c='b',                      # color(blue)
           cmap=mglearn.cm2,           # color mapping
           s=60,                       # 점 사이즈
           edgecolor='k')              # 점 테두리 색(black)

ax.scatter(train_x_sc_pca2[train_y == 1, 0], 
           train_x_sc_pca2[train_y == 1, 1], 
           train_x_sc_pca2[train_y == 1, 2], 
           c='r', 
           cmap=mglearn.cm2, 
           s=60, 
           edgecolor='k')
# --------------------------------------------------------------------------- #

# [ 분류분석 활용 1 : PCA + knn ]
# 2. 이미지 인식 / 분석 : PCA + knn
# sklearn에서의 이미지 데이터 셋
# - 2000년 초반 이후 유명인사 얼굴 데이터
# - 전처리 속도를 위해 흑백으로 제공
# - 총 62명의 사람의 얼굴을 여러 장 촬영한 데이터 제공
# - 총 3023개의 이미지 데이터 (행의 수), 87x65 (5655) 픽셀로 규격화 제공 (컬럼 수)

# 1) 데이터 로딩 및 설명
from sklearn.datasets import fetch_lfw_people
people = fetch_lfw_people(min_faces_per_person = 20, resize = 0.7)
# 20 => 한사람당 최소 이미지 개수

people.keys()        # ['data', 'images', 'target', 'target_names', 'DESCR']
people.data.shape    # (2936, 5655) => 2차원
people.images.shape  # (2936, 87, 65) (층, 행, 열) => 3차원 2936의 층 각각에 1개의 이미지가 있는 모습

people.data[0, :]                       # 첫번쨰 이미지의 변환된 값
people.target[0]                        # 17 => 학습용 Y값 (숫자로 변환된)
people.target_names[people.target[0]]   # 'Gray Davis' => 원래 Y값

people.data[0, :].min()    # 9.666667
people.data[0, :].max()    # 250.0

# RGB 변환된 sample data를 다시 이미지로 변환
fig, ax = plt.subplots(1, 3, figsize = (15, 8))

# imshow는 2차원 데이터가 필요 -> people.images 사용 (not people.data)
# 머신러닝도 2차원 데이터만 학습 가능
ax[0].imshow(people.images[0, :, :])   
ax[1].imshow(people.images[100, :, :])
ax[2].imshow(people.images[1000, :, :])

name1 = people.target_names[people.target[0]]
name2 = people.target_names[people.target[100]]
name3 = people.target_names[people.target[1000]]

ax[0].set_title(name1)
ax[1].set_title(name2)
ax[2].set_title(name3)

people.images[0]
people.data[0]

# =============================================================================
# # [ 참고 : 이미지에서 RGB 추출 ]
# import imageio
# im = imageio.imread('cat2.gif')
# 
# im.shape    # (200, 200, 4) (픽셀 세로, 픽셀 가로, 컬러)
# =============================================================================

# 2) 각 클래스별 균등 추출 * why?
people.target

np.bincount(people.target)
np.bincount([1, 1, 1, 2, 2])     # array([0, 3, 2]) => 0의 개수, 1의 개수, 2의 개수

# Y값이 10인 대상을 50개만 추출
np.where(people.target == 10)    # 조건에 맞는 행 번호 리턴 => 튜플 형태
# (array([   0,    1,   88,  764,  793,  862,  876,  914,  951,  962, 1077,
# 1154, 1442, 1465, 1500, 1758, 1807, 1878, 1883, 2099, 2273, 2462,
# 2661, 2662, 2687, 2723]),)    => 각각의 행의 target이 17임

np.where(people.target == 10)[0]         # 튜플을 array형태로 변경 => 236개 
np.where(people.target == 10)[0][:50]    # 처음부터 50개 추출

# =============================================================================
# # [ 참고 : array 확장 ]
# a1 = np.array([1, 2, 3])
# a2 = np.array([4, 5, 6])
# a1.append?    # 불가
# 
# pd.concat([Series(a1), Series(a2)])    # Series로 확장
# np.hstack([a1, a2])                    # array로 확장
# list(a1) + list(a2)                    # list로 확장
# =============================================================================

# 전체 Y에 대한 최대 50개 추출
v_nrow = []
for i in np.unique(people.target) :
    nrow = np.where(people.target == i)[0][:50]
    v_nrow = v_nrow + list(nrow)
    
len(v_nrow)        # 각 클래스별 최대 50개 추출 후 data set 크기 : 1976 (이전 2936)

people_x = people.data[v_nrow]
people_y = people.target[v_nrow]

# 3) train, test 분리
train_x, test_x, train_y, test_y = train_test_split(people_x,
                                                    people_y,
                                                    random_state = 0)

# 4) knn 모델 적용
m_knn = knn_c(5)
m_knn.fit(train_x, train_y)
m_knn.score(test_x, test_y)    # 0.224

v_pre = m_knn.predict(test_x[0].reshape(1, -1))    # 2차원 형식으로 전달
v_pre = v_pre[0]    # 차원 축소를 위한 색인 (값만 추출)
y_pre = people.target_names[v_pre][0]    # 예측값 => 'Roh Moo-hyun'
y_val = people.target_names[test_y[0]]   # 실제값 => 'Alejandro Toledo'

# 5) 예측값과 실제값의 시각화
fig, ax = plt.subplots(1, 2, figsize = (15, 8))
ax[0].imshow(test_x[0].reshape(87, 65))                # 실제값
ax[1].imshow(people.images[people.target == v_pre][0])    # 예측값

plt.rc('font', family = 'Malgun Gothic')

ax[0].set_title('예측값 : ' + y_pre)
ax[0].set_title('실제값 : ' + y_val)

# 6) 기타 튜닝
# 스케일링
m_sc = standard()
m_sc.fit(train_x)
train_x_sc = m_sc.transform(train_x)
test_x_sc = m_sc.transform(test_x)

v_score_tr = [] ; v_score_te = []
for i in range(1, 11) :
    m_knn = knn_c(i)
    m_knn.fit(train_x_sc, train_y)
    v_score_tr.append(m_knn.score(train_x_sc, train_y))
    v_score_te.append(m_knn.score(test_x_sc, test_y))

plt.plot(v_score_tr, label = 'train_score')
plt.plot(v_score_te, c = 'red', label = 'test_score')

plt.legend()

# 7) PCA로 변수가공
m_pca = PCA(n_components = 100, whiten = True)
m_pca.fit(train_x_sc)
train_x_sc_pca = m_pca.transform(train_x_sc)
test_x_sc_pca = m_pca.transform(test_x_sc)

train_x_sc_pca.shape    # (1482, 100)
test_x_sc_pca.shape     # (494, 100)

m_knn = knn_c(3)
m_knn.fit(train_x_sc_pca, train_y)
m_knn.score(test_x_sc_pca, test_y)    # 0.32 => 더 높아짐
# --------------------------------------------------------------------------- #

# [ 문자 변수 -> 숫자 변수 변경 ]
df1 = DataFrame({'col1' : [1, 2, 3, 4],
                 'col2' : ['M', 'M', 'F', 'M'],
                 'y' : ['N', 'Y', 'N', 'Y']})

# 1) if문으로 직접 치환
np.where(df1.col2 == 'M', 0, 1)

# 2) dummy 변수 치환 함수 사용
pd.get_dummies(df1, drop_first = True)

# 3) 기타 함수 사용
mr = pd.read_csv('mushroom.csv', header = None)

# 문자값가지고 있는 컬럼 숫자로 변경
# 3-1)
def f_char(df) :
    target = []
    data = []
    attr_list = []
    for row_index, row in df.iterrows() :
        target.append(row.iloc[0])     # Y값열 분리
        row_data = []
        for v in row.iloc[1:] :        # X값열 하나씩 v로 전달
            row_data.append(ord(v))    # ord를 사용한 숫자 변환 방식
        data.append(row_data)

f_char(mr)
DataFrame(data)
        
# ord
ord('a')    # 97 => 유니크하게 문자별로 고유의 번호를 불러줌
ord('abc')  # Error => 1개의 length인 문자만 가능

# iterrows
# => row에 대한 index와 각 row별 하나씩 꺼내서 전달함 (반복문에 쓰기 좋음)
for row_index, row in mr.iterrows() :
    print(str(row_index) + ':' + str(list(row)))
# 8119:['e', 'k', 's', 'n', 'f', 'n', 'a', 'c', 'b', 'y', 'e', '?', 's', 's', 'o', 'o', 'p', 'o', 'o', 'p', 'b', 'c', 'l']
# => 8119 row에서 e, k, s 순으로 전달

# target까지 숫자와 하는 함수 만들어 보기*

# 3-2) target 포함해서 변환해주고, ord 한계를 넘어 2자 이상 문자도 변환하는 방법
from sklearn.preprocessing import LabelEncoder

df2 = DataFrame({'col1' : [1, 2, 3, 4],
                 'col2' : ['ABC1', 'BCD1', 'BCD2', 'CDF1'],
                 'y' : ['N', 'N', 'N', 'Y']})

f_char(df2)    # Error => ord length

m_label = LabelEncoder()
m_label.fit(df2.col2)          # 1차원만 학습 가능
m_label.transform(df2.col2)    # 값의 unique value마다 서로 다른숫자 부여
# => 열의 unique value 찾고, 순서대로 정렬한 후 서로 다른 숫자 부여하는 방식
# => 위 mushroom data는 모든 값이 같은 속성이라 열 상관 없이 각 문자에 하나의 숫자를 대입하면 됨
# => 실제로는 열별로 봐야함

# 행별로 진행하면 아래 같은 오류 발생
# a b c  => 0 1 2 
# b c d  => 0 1 2 
# c d e  => 0 1 2 

# 4) LabelEncoder에 의한 변환 기법
def f_char2(df) :
    def f1(x) :
        m_label = LabelEncoder()
        m_label.fit(x)
        return m_label.transform(x)
    return df.apply(f1, axis = 0)

f_char2(mr)
f_char2(df2)    # 숫자 컬럼은 변경하지 않도록 수정, dtype으로 해보기*

# --------------------------------------------------------------------------- #

# [ 회귀 분석 ]
# - 통계적 모델이므로 여러가지 통계적 가정 기반
# - 평가 메트릭이 존재
# - 인과관계 파악 => 다른 모델들은 모델을 들여다 보기 힘들기 때문
# - 이상치에 민감
# - 다중공선성(설명 변수끼리의 정보의 중복) 문제를 해결하지 않으면 잘못된 회귀 계수가 유도될 수 있음
#   (유의성 결과, 반대의 부호로 추정)
# - 학습된 설명변수가 많을수록 모델 자체의 설명력이 대체적으로 높아짐 (overfit)
# - 기존 회귀를 최근에는 분류모델기반 회귀나 NN로 대체해서 많이 사용

# - 연속형 변수의 범주화(binding)에 따라 성능이 좋아질 수 있음
# Y : 성적
# x1 : 성별
# x2 : 나이
# x3 : IQ
# a4x4 : 공부시간
# ex) 공부시간이 1-4시간, 5-10시간 사이에 설명력 차이가 크다 -> 둘을 각각 범주화 -> 설명력 향상
# => RF 같은 트리 기반 모델에서는 이미 위 작업을 진행하므로 크게 의미가 없음

# a4  1.7(회귀 계수)  0.0001(p-value)   H0 : a4 = 0, H1 : a4 != 0
# => 다중공선성의 문제로 모델을 통제하지 못한다면 회귀 계수는 100% 신뢰하기는 어려움

# 1) 회귀 모델 적용 (sklearn 모델)
from sklearn.linear_model import LinearRegression

m_reg = LinearRegression()
m_reg.fit(df_boston.data, df_boston.target)
m_reg.score(df_boston.data, df_boston.target)    # R^2 : 0.74

# R^2
# - 회귀분석에서의 모델을 평가하는 기준
# - SSR/SST : 총 분산 중 회귀식으로 설명할 수 있는 분산의 비율
# - 0 ~ 1의 값을 갖고, 1에 가까울수록 좋은 회귀 모델
# - 대체적으로 설명변수가 증가할수록 높아지는 경향

# SST, SSR, SSE
# y - yhat = (y - ybar) + (ybar - yhat)
# sum((y - ybar)^2) = sum((y - yhat)^2) + sum((yhat - ybar)^2)
# SST(총편차제곱합)    = SSR(오차제곱항) + SSE(회귀제곱항)
# MST(총분산)         = MSE(평균제곱오차) + MSR(회귀제곱합)
# 고정                 작아짐             커짐           => 좋은 회귀식일수록
# SSR/SST : 좋은 모델일수록 작아져야 함
# SSE/SST : 좋은 모델일수록 커져야 함

# 2) 2차항 설명변수 추가 후 회귀 모델 적용
m_reg = LinearRegression()
m_reg.fit(df_boston_et_x, df_boston_et_y)
m_reg.score(df_boston_et_x, df_boston_et_y)    # R^2 : 0.929
# => 설명력 매우 높아짐 -> 자세히 조사해보면 overfit이 매우 크다는 것을 알 수 있음 -> 그냥 믿으면 안됨

v1 = [1, 2, 3, 4, 5]
np.var(v1) 

# 보스턴 주택 데이터 가격 셋
from sklearn.datasets import load_boston
df_boston = load_boston()

df_boston.data
df_boston.feature_names
df_boston.target        # 종속변수가 연속형

print(df_boston.DESCR)

train_x, test_x, train_y, test_y = train_test_split(df_boston.data,
                                                    df_boston.target,
                                                    random_state = 0)

# 1) 2차 interaction이 추가된 확장된 boston data set
from mglearn.datasets import load_extended_boston
df_boston_et_x, df_boston_et_y = load_extended_boston()

train_x_et, test_x_et, train_y_et, test_y_et = train_test_split(df_boston_et_x,
                                                                df_boston_et_y,
                                                                random_state = 0)

from sklearn.linear_model import LinearRegression

# 2) 회귀분석 적용
m_reg1 = LinearRegression()
m_reg1.fit(train_x, train_y)
m_reg1.score(train_x, train_y)   # 0.769
m_reg1.score(test_x, test_y)     # 0.635 => R^2, 회귀선이 해당 데이터를 설명하는 정도

m_reg2 = LinearRegression()
m_reg2.fit(train_x_et, train_y_et)
m_reg2.score(train_x_et, train_y_et)   # 0.952
m_reg2.score(test_x_et, test_y_et)     # 0.607

dir(m_reg2)
m_reg1.coef_    # 회귀 계수
m_reg2.coef_    # 회귀 계수에 대한 유의성 검정의 결과 출력 x

# y = -1.17 * X1 + ... 

# 5) 회귀분석의 유의성 검정 결과 확인 : statmodels 패키지(모듈)

import statsmodels.api as sm

m_reg3 = sm.OLS(train_y, train_x).fit()

dir(m_reg3)
print(m_reg3.summary())

 OLS Regression Results                                
=======================================================================================
Dep. Variable:                      y   R-squared (uncentered):                   0.963
Model:                            OLS   Adj. R-squared (uncentered):              0.962
Method:                 Least Squares   F-statistic:                              738.2
Date:                Mon, 26 Oct 2020   Prob (F-statistic):                   2.27e-253    # 작을수록 모델 신뢰성이 높아짐
Time:                        09:35:52   Log-Likelihood:                         -1122.8
No. Observations:                 379   AIC:                                      2272.
Df Residuals:                     366   BIC:                                      2323.
Df Model:                          13                                                  
Covariance Type:            nonrobust                                                  
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]    # P>|t| => 양측검정
------------------------------------------------------------------------------
x1            -0.1163      0.040     -2.944      0.003      -0.194      -0.039
x2             0.0458      0.016      2.895      0.004       0.015       0.077
x3            -0.0341      0.070     -0.485      0.628      -0.172       0.104
x4             2.5435      1.014      2.508      0.013       0.549       4.538
x5            -0.0774      3.812     -0.020      0.984      -7.573       7.419    # 회귀계수는 0에 가까움
x6             5.9751      0.346     17.252      0.000       5.294       6.656
x7            -0.0153      0.016     -0.975      0.330      -0.046       0.016
x8            -0.9255      0.222     -4.178      0.000      -1.361      -0.490
x9             0.1046      0.074      1.423      0.156      -0.040       0.249
x10           -0.0086      0.004     -2.027      0.043      -0.017      -0.000
x11           -0.4486      0.126     -3.566      0.000      -0.696      -0.201
x12            0.0141      0.003      4.608      0.000       0.008       0.020
x13           -0.3814      0.058     -6.615      0.000      -0.495      -0.268
==============================================================================
Omnibus:                      179.629   Durbin-Watson:                   2.029
Prob(Omnibus):                  0.000   Jarque-Bera (JB):             1538.969
Skew:                           1.798   Prob(JB):                         0.00
Kurtosis:                      12.194   Cond. No.                     8.70e+03
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 8.7e+03. This might indicate that there are
strong multicollinearity or other numerical problems.

# [ 회귀분석에서의 다중공선성 문제 ]
# - 모형은 유의하나 회귀 계수의 유의성 문제
# - 유의하다 판단되는 회귀 계수의 유의성 문제 (p-value가 큼)
# - 예상한 회귀 계수의 부호와 다른 부호로 추정되는 경우
dir(m_reg3)

# 1) 다중공선성 문제 해결
# - 변수 제거 (간단한 모델로 변경)
# - 변수 결합 (정보의 중복이 있는 변수끼리 결합)
# - PCA에 의한 전체 변수 결합
# - 기타 모델 적용 (릿지, 리쏘)

# 2) 릿지 회귀
# - 의미가 약한 변수의 회귀 계수를 0에 가깝게 만듦 -> 제거하지는 않음
# - 변수를 축소함으로 모델이 갖는 정보의 중복을 줄임
# - 다중공선성의 문제를 어느 정도 개선할 수 있음
# - alpha라는 매개변수 튜닝을 통해 모델의 복잡도를 제어할 수 있음

from sklearn.linear_model import Ridge

m_ridge = Ridge()
m_ridge.fit(train_x_et, train_y_et)
m_ridge.score(train_x_et, train_y_et)   # 0.885
m_ridge.score(test_x_et, test_y_et)     # 0.752

train_x_et.shape[1]              # 설명변수의 개수 => 104
sum(abs(m_reg2.coef_) > 0.1)     # 104개 설명변수의 의미 있는 학습
sum(abs(m_ridge.coef_) > 0.1)    # 회귀계수의 절대값이 0.1보다 큰 설명변수 => 98 -> 104개 사용하는 것보다 설명력 높아짐

# 매개변수 튜닝 => 릿지는 매개변수 튜닝이 꼭 필요
v_alpha = [0.001, 0.01, 0.1, 1, 10, 100]
v_score_tr = [] ; v_score_te = []

for i in v_alpha :
    m_ridge = Ridge(alpha = i)
    m_ridge.fit(train_x_et, train_y_et)
    v_score_tr.append(m_ridge.score(train_x_et, train_y_et))
    v_score_te.append(m_ridge.score(test_x_et, test_y_et))
    
plt.plot(v_score_tr, label = 'train_score')
plt.plot(v_score_te, label = 'test_score', c = 'red')
plt.legend()
plt.xticks(np.arange(len(v_alpha)), v_alpha)

# 3) 라쏘 회귀
# - 의미가 약한 변수의 회귀 계수를 0으로 만듦 -> 변수제거
# - 변수를 축소함으로 모델이 갖는 정보의 중복을 줄임
# - 다중공선성의 문제를 어느 정도 개선할 수 있음
# - alpha라는 매개변수 튜닝을 통해 모델의 복잡도를 제어할 수 있음

from sklearn.linear_model import Lasso

m_lasso = Lasso()
m_lasso.fit(train_x_et, train_y_et)
m_lasso.score(train_x_et, train_y_et)     # 29.32
m_lasso.score(test_x_et, test_y_et)       # 20.94

sum(m_lasso.coef_ == 0)    # 100 => 제거된 변수

m_lasso2 = Lasso(alpha = 10)
m_lasso2.fit(train_x_et, train_y_et)
m_lasso2.score(train_x_et, train_y_et)     # 0
m_lasso2.score(test_x_et, test_y_et)       # -0

sum(m_lasso2.coef_ == 0)    # 104 => 알파값이 커지니 제거된 변수 증가 -> 모델 단순화

# 매개변수 튜닝 => 릿지는 매개변수 튜닝이 꼭 필요
v_alpha = [0.001, 0.01, 0.1, 1, 10, 100]
v_score_tr = [] ; v_score_te = []

for i in v_alpha :
    m_lasso = Lasso(alpha = i)
    m_lasso.fit(train_x_et, train_y_et)
    v_score_tr.append(m_lasso.score(train_x_et, train_y_et))
    v_score_te.append(m_lasso.score(test_x_et, test_y_et))
    
plt.plot(v_score_tr, label = 'train_score')
plt.plot(v_score_te, label = 'test_score', c = 'red')
plt.legend()
plt.xticks(np.arange(len(v_alpha)), v_alpha)

# RF regressor
m_rfr = rf_r()
m_rfr.fit(train_x_et, train_y_et)
m_rfr.score(train_x_et, train_y_et)    # 0.985
m_rfr.score(test_x_et, test_y_et)      # 0.764

# 당뇨병 수치 예측 data set로 정리해보기
from sklearn.datasets import load_diabetes
df_diabetes = load_diabetes()

print(df_diabetes.DESCR)
# --------------------------------------------------------------------------- #