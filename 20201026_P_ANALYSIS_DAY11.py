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
# MST(총분산)        = MSE(평균제곱오차) + MSR(회귀제곱합)
# 고정                 작아짐           커짐           => 좋은 회귀식일수록
# SSR/SST : 좋은 모델일수록 작아져야 함
# SSE/SST : 좋은 모델일수록 커져야 함

# 2) 2차항 설명변수 추가 후 회귀 모델 적용
m_reg = LinearRegression()
m_reg.fit(df_boston_et_x, df_boston_et_y)
m_reg.score(df_boston_et_x, df_boston_et_y)    # R^2 : 0.929
# => 설명력 매우 높아짐 -> 자세히 조사해보면 overfit이 매우 크다는 것을 알 수 있음 -> 그냥 믿으면 안됨

y1 = [1, 2, 3, 4, 5]
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

# 3) 2차 interaction이 추가된 확장된 boston data set
from mglearn.datasets import load_extended_boston
df_boston_et_x, df_boston_et_y = load_extended_boston()

train_x_et, test_x_et, train_y_et, test_y_et = train_test_split(df_boston_et_x,
                                                                df_boston_et_y,
                                                                random_state = 0)

from sklearn.linear_model import LinearRegression

# 4) 회귀분석 적용
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
# - 유의하다 판단되는 회귀 계쑤의 유의성 문제 (p-value가 큼)
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

m_rfr = rf_r()
m_rfr.fit(train_x_et, train_y_et)
m_rfr.score(train_x_et, train_y_et)    # 0.985
m_rfr.score(test_x_et, test_y_et)      # 0.764

# 당뇨병 수치 예측 data set로 정리해보기
from sklearn.datasets import load_diabetes
df_diabetes = load_diabetes()

print(df_diabetes.DESCR)
# --------------------------------------------------------------------------- #