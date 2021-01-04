# 3. RF(iris data) in Python
# 1) 데이터 로딩
from sklearn.datasets import load_iris
df_iris = load_iris()

# 2) 데이터 분리
train_x, test_x, train_y, test_y = train_test_split(df_iris.data,
                                                    df_iris.target,
                                                    random_state = 0)

# 3) 모델 학습
m_rf = rf_c(random_state = 0)
# => random_state를 고정해줘야 같은 데이터를 샘플링하기 때문에 매개변수 튜닝이 쉬움
m_rf.fit(train_x, train_y)    # n_estimators = 100, n_jobs = None
# n_jobs => 패러럴, 병렬처리 가능; 2개하면 2개 프로세스가 동시에 돌아감; cpu 많이 먹으므로 너무 많이 쓰지 말기

# 4) 모델 평가
m_rf.score(test_x, test_y)    # 97.37

# 5) 매개변수 튜닝
v_score_te = []

for i in range(1, 101) :
    m_rf = rf_c(random_state = 0, n_estimators = i)
    m_rf.fit(train_x, train_y)
    v_score_te.append(m_rf.score(test_x, test_y))

import matplotlib.pyplot as plt    
plt.plot(np.arange(1, 101), v_score_te, color = 'red')    # 5이상이면 충분히 높게 나옴

# 6) 특성 중요도 파악
m_rf.base_estimator_
m_rf.feature_importances_
# array([0.10749462, 0.02616594, 0.42160356, 0.44473587]) => DT와 결과가 다름 -> 여러 가지 방식으로 중요도 체크해야 함

# -------------------------------- 연 습 문 제 -------------------------------- #
# 연습문제 2. cancer data의 분류 모델 생성 및 비교
from sklearn.datasets import load_breast_cancer
df_cancer = load_breast_cancer()
df_cancer.data.shape    # (569, 30)

df_cancer2 = pd.read_csv('cancer.csv')
df_cancer2.shape    # (569, 32)

# knn, DT, RF 예측력 비교
# 1) knn
# step 1) 데이터 분리
df_cancer.data
df_cancer.feature_names
df_cancer.target
df_cancer.target_names

X = df_cancer.data
Y = df_cancer.target

from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(X, Y, train_size = 0.7, random_state = 0)

# 7:3 = train:test 비율로 분리
train_x.shape    # (398, 30)
test_x.shape     # (171, 30)

# step 2) 데이터 학습
from sklearn.neighbors import KNeighborsClassifier as knn_c

m_knn = knn_c(n_neighbors = 1)
m_knn.fit(train_x, train_y)    # train data로 모델 학습
m_knn.predict(test_x)          # 모델 평가 및 예측용

# step 3) 평가 => test 데이터 활용
m_knn.score(test_x, test_y)    # 91.91

# step 4) 튜닝 - k수 변화에 따른 시각화
score_tr = []
score_te = []
for i in range(1, 11) :
    m_knn = knn_c(i)
    m_knn.fit(train_x, train_y)
    
    score_tr.append(m_knn.score(train_x, train_y))
    score_te.append(m_knn.score(test_x, test_y))

import matplotlib.pyplot as plt
plt.plot(range(1, 11), score_tr, label = 'train_score')
plt.plot(range(1, 11), score_te, label = 'test_score')
plt.legend()
# train이 test보다 높고, 간격이 가장 좁은 구간 => 4..?

# 2) DT
# step 2) 모델 생성 및  훈련
m_dt = dt_c()
m_dt.fit(train_x, train_y)

# step 3) 모델 평가
m_dt.score(test_x, test_y)    # 92.40

# 교차 검증
from sklearn.model_selection import cross_val_score  
v_score = cross_val_score(m_dt, df_cancer.data, df_cancer.target, cv = 5)
# array([0.9122807 , 0.92105263, 0.92105263, 0.94736842, 0.88495575])
v_score.mean()    # 91.73

# step 4) 매개 변수 튜닝
score_te = []
for i in range(2, 11) :
    m_dt = dt_c(min_samples_split = i, random_state = 0)
    v_score = cross_val_score(m_dt, df_cancer.data, df_cancer.target, cv = 5)
    score_te.append(v_score.mean())

# [0.9173730787144851,
#  0.9173730787144851,
#  0.915618692749573,
#  0.915618692749573,
#  0.915618692749573,
#  0.915618692749573,
#  0.915618692749573,
#  0.9103555348548362,
#  0.9103555348548362] => 이 데이터에서는 min_samples_split이 큰 의미가 없음

# 3) RF
m_rf = rf_c(random_state = 0)
m_rf.fit(train_x, train_y)
m_rf.score(test_x, test_y)    # 95.91

# RF가 가장 높은 예측력 보임 (일반적인)

# 매개변수의 특성 중요도 시각화
s1 = Series(m_rf.feature_importances_, index = df_cancer.feature_names)
s1.sort_values(ascending = False)

# 특성 중요도 시각화
s1 = Series(m_rf.feature_importances_, index = df_cancer.feature_names)
s1.sort_values(ascending=False)

def plot_feature_importances_cancer(model, data) : 
    n_features = data.data.shape[1]    # 컬럼 사이즈
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), data.feature_names)    # y 눈금
    plt.xlabel("특성 중요도")    # x축 이름
    plt.ylabel("특성")         # y축 이름
    plt.ylim(-1, n_features)

plt.rc('font', family = 'Apple Gothic')
plot_feature_importances_cancer(m_rf, df_cancer)

# => 4개 변수 정도가 중요도 매우 높아 보임
# => RF는 불필요한 변수 제거하고 모델수행하나 knn이나 전통 통계분석 등은 변수 선택 직접해야 함
# --------------------------------------------------------------------------- #