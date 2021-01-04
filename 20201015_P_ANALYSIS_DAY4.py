# [ 분석 시 고려 사항 ]
# 1. 변수 선택
# 2. 변수 변형 (결합 포함)
# 3. 교호작용 (interaction)
# 4. 교차검증 (cross validation)
# 5. 최적의 매개변수 조합 (grid search) : train/val/test
# 6. 변수 표준화 (scaling)

# --------------------------------------------------------------------------- #

# [ 트리기반 모델 ]
DT -> RF -> GB -> XGB -> ... 
# GB => 이전 트리 보완해서 다음 모델 만드는 구조, 오분류된 데이터를 정분류하도록 함 (TBC)
# 이런 발전 과정에도 대기업들은 RF, GB 많이 사용 - scikit learn에 포함

C:\Users\harryjeong> pip install xgboost

from xgboost.sklearn import XGBClassifier as xgb_c
from xgboost.sklearn import XGBRegressor as xgb_r

# 4. Gradiant Boosting Tree(GB)
# - 이전 트리의 오차를 보완하는 트리를 생성하는 구조
# - 비교적 단순한 초기 트리를 형성, 오분류 data point에 더 높은 가중치를 부여, 오분류 data point
#   를 정분류 하도록 더 보완된, 복잡한 트리를 생성
# - learning rate 만큼의 오차 보완률 결정 (0 ~ 1, 높을수록 과적합 발생 가능성 높음)
# - random forest 모델보다 더 적은 수의 tree로도 높은 예측력을 기대할 수 있음
# - 각 트리는 서로 독립적이지 않으므로(이전 트리가 끝나야 다음 트리 시작)
#   병렬처리에 대한 효과를 크게 기대하기 어려움

# GB(iris data) in python
# 1) 데이터 로딩
run profile1
from sklearn.datasets import load_iris
df_iris = load_iris()

train_x, test_x, train_y, test_y = train_test_split(df_iris.data,
                                                    df_iris.target,
                                                    random_state = 0)

# 2) 모델 생성 및 학습
from sklearn.ensemble import GradientBoostingClassifier as gb_c
from sklearn.ensemble import GradientBoostingRegressor as gb_r

m_gb = gb_c()
m_gb.fit(train_x, train_y)    # learning_rate = 0.1, max_depth = 3,
                              # max_features = None, min_samples_split = 2,
                              # n_estimators = 100
                              
# 3) 모델 평가
m_gb.score(test_x, test_y)    # 97.37                              

# 4) 매개 변수 튜닝
vscore_tr = [] ; vscore_te = []

for i in [0.001, 0.01, 0.1, 0.5, 1] :
    m_gb = gb_c(learning_rate = i)
    m_gb.fit(train_x, train_y)
    vscore_tr.append(m_gb.score(train_x, train_y))
    vscore_te.append(m_gb.score(test_x, test_y))

plt.plot([0.001, 0.01, 0.1, 0.5, 1], vscore_tr, label = 'train_score')
plt.plot([0.001, 0.01, 0.1, 0.5, 1], vscore_te, label = 'test_score', color = 'red')
plt.legend()    # 낮은 learning rate에서도 충분한 예측력 나옴

# 5) 특성 중요도 시각화
def plot_feature_importances(model, data) : 
    n_features = data.data.shape[1]    # 컬럼 사이즈
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), data.feature_names)    # y 눈금
    plt.xlabel("특성 중요도")    # x축 이름
    plt.ylabel("특성")         # y축 이름
    plt.ylim(-1, n_features)

plot_feature_importances(m_gb, df_iris)    
# --------------------------------------------------------------------------- #

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

# 6.2 scaling 실행
# 1) standard scaling
m_sc1 = standard()
m_sc1.fit(train_x)        # 각 설명변수의 평균, 표준편차 계산
m_sc1.transform(train_x)  # 위 값 바탕으로 표준화 시킴

m_sc1.transform(train_x).mean(axis = 0)    # 사실상 0이라 보는게 맞음
m_sc1.transform(train_x).std(axis = 0)     # 1

# 2) minmax scaling
m_sc2 = minmax()
m_sc2.fit(train_x)        # 각 설명변수의 최대, 최소 구하기
m_sc2.transform(train_x)  # 최소를 0, 최대를 1에 맞춰 계산

m_sc2.transform(train_x).min(axis = 0)    # 0
m_sc2.transform(train_x).max(axis = 0)    # 1

# train값을 scaling하면 test값도 scaling 해야 함
m_sc1.transform(test_x)
m_sc2.transform(test_x)

m_sc2.transform(test_x).min(axis = 0)    # 0이 아님 => train으로 fit한 값이기 때문
m_sc2.transform(test_x).max(axis = 0)    # 1이 아님 => train으로 fit한 값이기 때문
# test를 fit하고 다시 실험하면 다시 0, 1나옴
m_sc3 = minmax()
m_sc3.fit(test_x)
m_sc3.transform(test_x)
m_sc3.transform(test_x).min(axis = 0)    # 0
m_sc3.transform(test_x).max(axis = 0)    # 1

# way 1) train fit -> train, test
# way 2) train fit -> train  / test fit -> test
# way 1)이 정답 => 하늘 아래 두개의 태양이 있을 수 없음

# -------------------------------- 연 습 문 제 -------------------------------- #
# 연습문제 3. iris data의 knn모델 적용 시 scaling 전/후 비교
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

# 3) 변수 선택 및 scaling 후
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

# 3.1 interaction 적용 data 추출    
from sklearn.preprocessing import PolynomialFeatures as poly

원본         => 2차항 적용 (transform 작업)
x1 x2 x3       x1^2 x2^2 x3^2 x1x2 x1x2 x2x3
1  2  3        1    4    9    2    3    6
2  4  5        4    16   25   8    10   20

m_poly = poly(degree = 2)    # 2차항을 만들겠다
m_poly.fit(train_x)          # 각 설명변수에 2차항 모델 생성
# ** test data set은 fitting 필요 없음 why?
train_x_poly = m_poly.transform(train_x)    # 스케일링 된 데이터셋으로 하는게 더 좋음
test_x_poly = m_poly.transform(test_x)

m_poly.get_feature_names()   # 변경된 설명변수들의 형태 = 2차항 모습

DataFrame(m_poly.transform(train_x),
          columns = m_poly.get_feature_names())  # 보기 좋음 -> 변수가 엄청 많으면 이 또한 쉽진 않음

col_poly = m_poly.get_feature_names(df_iris.feature_names)  # 실제 컬럼이름이 반영된 교호작용 출력

DataFrame(m_poly.transform(train_x),
          columns = m_poly.get_feature_names(df_iris.feature_names)) # 훨씬 보기 좋음

# 순서 : 원본 -> Scaling -> Poly -> RF 등으로 설명변수 체크 -> Knn

# 3.2 확장된 데이터셋을 RF에 학습, feature importance 확인 
m_rf = rf_c(random_state = 0)
m_rf.fit(train_x_poly, train_y)

m_rf.score(test_x)    # Error => fit에서 교호작용 적용했기 때문에 변수 개수 다름

m_rf.score(test_x_poly, test_y)    # 0.973

# 대부분은 설명변수 증가하면 예측력 높아지지만, Tree기반 모델은 어차피 정하는 변수개수가 있으므로 큰 의미 없어짐

s1 = Series(m_rf.feature_importances_, index = col_poly)
s1.sort_values(ascending = False)
# --------------------------------------------------------------------------- #