# [ 분석 시 고려 사항 ]
# 1. 변수 선택
# 2. 변수 변형 (결합 포함)
# 3. 교호작용 (interaction)
# 4. 교차검증 (cross validation)
# 5. 최적의 매개변수 조합 (grid search) : train/val/test
# 6. 변수 표준화 (scaling)

# [ 분석 시 고려사항 2. 변수 변형 (결합 포함) ]
# 2.1 종속 변수의 변형(NN 기반 모델일 경우 필수)
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
  
# 2.2.1 변수선택 방법 1 : 모델 기반 변수 선택
# - 트리, 회귀 기반 모델에서의 변수 중요도를 사용하여 변수를 선택하는 방식
# - 트리에서는 변수 중요도를 참고, 회귀에서는 각 변수의 계수 참고
# - 모델에 학습된 변수끼리의 상관 관계도 함께 고려(종합적 판단)
from sklearn.feature_selection import SelectFromModel

# 예제) SelectFromModel iris data
from sklearn.datasets import load_iris
df_iris = load_iris()

# 1) noise 변수 추가(10개)
vrandom = np.random.RandomState(0)
vcol = vrandom.normal(size = (len(df_iris.data), 10))  # 150 X 10
vrandom.normal?
df_iris_new = np.hstack([df_iris.data, vcol])
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
m_select1 = SelectFromModel(m_rf,               # 변수 중요도를 파악할 모델 명 전달 
                            threshold='median') # 선택 범위
 
m_select1.fit(df_iris.data, df_iris.target)
m_select1.get_support()

m_select1.fit(df_iris_new, df_iris.target)
m_select1.get_support()

# 3) 선택된 변수의 dataset 추출
df_iris_new[:, m_select1.get_support()]     # 중요변수 선택 후 dataset
m_select1.transform(df_iris_new)

# 4) 변수 중요도 확인
m_select1.estimator_.feature_importances_

# 2.2.2 변수선택 방법 2 : 일변량 통계 기법
# - 변수 하나와 종속변수와의 상관 관계 중심으로 변수 선택
# - 다른 변수가 함께 학습될때의 판단과는 다른 결과가 나올 수 있음
# - 학습 시킬 모델이 필요 없어 연산속도가 매우 빠름
from sklearn.feature_selection import SelectPercentile
  
# 1) 변수 선택 모델 생성 및 적용
m_select2 = SelectPercentile(percentile=30)
m_select2.fit(df_iris_new, df_iris.target)

# 2) 변수 선택 결과 dataset 확인
m_select2.get_support()
df_iris_new[:, m_select2.get_support()]
m_select2.transform(df_iris_new)

# 3) 변수 선택 시각화
plt.matshow(m_select2.get_support().reshape(1,-1), cmap='gray_r')

# 2.2.3 변수선택 방법 3 : 반복적 선택(RFE)
# - step wise 기법과 유사
# - 전체 변수를 학습 시킨 후 가장 의미 없는 변수 제거,
#   반복하다 다시 변수 추가가 필요한 경우 추가하는 과정
# - 특성의 중요도를 파악하기 위한 모델 필요
  
from sklearn.feature_selection import RFE

# [ RFE iris data ]
m_rf = rf()
m_select3 = RFE(m_rf, n_features_to_select=2)  # 개수 기반 선택 가능
m_select3.fit(df_iris.data, df_iris.target)

m_select3.get_support()
m_select3.ranking_                          # 전체 특성 중요도 순서 확인
m_select3.estimator_.feature_importances_   # 선택된 특성 중요도 값 확인

# -------------------------------- 연 습 문 제 -------------------------------- #
# 연습문제 5. cancer data set에 대한 feature selection(상위 50%)
# 1) data loading
from sklearn.datasets import load_breast_cancer
df_cancer = load_breast_cancer()

# 2) 변수 선택 모델 생성
m_rf=rf_c()
m_select1 = SelectFromModel(m_rf, threshold = 'median')
m_select2 = SelectPercentile(percentile = 50)
m_select3 = RFE(m_rf, n_features_to_select = df_cancer.data.shape[1]/2)

m_select1.fit(df_cancer.data, df_cancer.target)
m_select2.fit(df_cancer.data, df_cancer.target)
m_select3.fit(df_cancer.data, df_cancer.target)

# 3) 선택된 변수 확인
m_select1.get_support()
m_select2.get_support()
m_select3.get_support()

v_sel_col = m_select1.get_support() & m_select2.get_support() & m_select3.get_support()
df_cancer.feature_names[v_sel_col]

# 4) 변수 선택 전/후 모델 비교
# --------------------------------------------------------------------------- #