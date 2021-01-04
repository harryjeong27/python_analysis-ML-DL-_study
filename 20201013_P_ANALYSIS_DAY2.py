# 예제) 보스턴 주택 가격 데이터 셋 (회귀 분석 데이터)
from sklearn.datasets import load_boston
df_boston = load_boston()
df_boston.keys()        # 'data', 'target', 'feature_names', 'DESCR'

df_boston.data.shape    # (506, 13)

import mglearn
boston_x, boston_y = mglearn.datasets.load_extended_boston()
boston_x.shape          # (506, 104) => 기존 데이터셋에서 2차 교호작용을 추가한 형태
# --------------------------------------------------------------------------- #

# [ 분석 시 고려사항 3. 교호작용 ]
# 1차원 데이터셋으로는 예측력이 너무 적음(13%) -> 교호작용으로 예측력 높여보자 -> 각 차원의 확률 13%, 13%, 78%
# 좋은 방법은 아님 -> 가장 좋은 방법은 중요 변수가 뭔지 파악해서 집어 넣는 것
# 마땅히 떠오르는 변수가 없을 경우만 사용해보기
13(1차원) + 13(2차원) + 78(상호작용)

# =============================================================================
# # [ 참고 : 파이썬 combination 출력 (발생 가능한 조합) ]
# import itertools
# list(itertools.combinations(['x1', 'x2', 'x3'], 2))    # choose(10, 2) in R
# =============================================================================

# 교호작용(interaction) 데이터 셋 => 혹시 변수 간에 의미 있는 관계가 있는지 체크
x1, x2, x2
- 2차원 교호작용 추가 : x1, x2, x3, x1x2, x1x3, x2x3, x1^2, x2^2, x3^2
- 3차원 교호작용 추가 : x1, x2, x3, x1x2, x1x3, x2x3, x1^2, x2^2, x3^2,
                    x1x2x3, x1^3, x2^3, x3^3
                    
# 주택가격 <- x1(지하 주차장 공급면적) * x2(강수량) => 강수량이라는 다른 변수를 넣으면 예측력이 확 높아질 수 있음     
# --------------------------------------------------------------------------- #

# 2. DT(iris data) in python
# 1) data loading
from sklearn.datasets import load_iris
df_iris = load_iris()

# 2) data split
train_x, test_x, train_y, test_y = train_test_split(df_iris.data,
                                                    df_iris.target,
                                                    random_state = 0)

# 3) 모델 생성
m_dt = dt_c()        # 매개변수 기본값

# 4) 훈련셋 적용 (모델 훈련)
m_dt.fit(train_x, train_y)    # min_samples_split = 2
                              # max_depth = None
                              # max_features = None
                              
# 5) 모델 평가
m_dt.score(test_x, test_y)    # 97.37                              

# 6) 매개 변수 튜닝
# - min_samples_split : 각 노드별 오분류 개수가 몇개 이하일 때까지 쪼개라
#                       min_samples_split 값보다 오분류 개수가 크면 split
#                       min_samples_split 값이 작을수록 모델은 복잡해지는 경향
# - max_features : fitting된 데이터는 같으나 그 데이터를 다르게 나눔 (복원추출 통해)
#                  각 노드의 고정시킬 설명변수의 후보의 개수
#                  max_features 클수록 서로 비슷한 트리로 구성될 확률 높아짐
#                  (설명력 가장 높은 변수가 선택될 가능성 높음)
#                  max_features 작을수록 서로 다른 트리로 구성될 확률 높아짐
#                  (복잡한 트리를 구성할 확률 높아짐)
# - max_depth : 설명변수의 중복 사용의 최대 개수
#               max_depth 작을수록 단순한 트리를 구성할 확률이 높아짐

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

# -------------------------------- 연 습 문 제 -------------------------------- #
# 연습문제 1.  test score가 가장 높은 매개변수 찾기
from sklearn.model_selection import cross_val_score

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

# 6) 매개변수의 특성 중요도 확인
m_dt?    # 모델 자체가 가지고 있는 정보확인
m_dt.feature_importances_
# array([0.        , 0.02014872, 0.89994526, 0.07990602]) => 설명변수 4개의 중요도
df_iris.feature_names
# ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

s1 = Series(m_dt.feature_importances_, index = df_iris.feature_names)
s1.sort_values(ascending = False)

# 7) 시각화 => DT가 시각화 가능한 거의 유일한 모델
# 7-1) graphviz 설치 (window)
download 후 압축해제 (C:/Program Files (x86))
# download link : https://graphviz.gitlab.io/_pages/Download/Download_windows.html

# 7-2) graphviz 설치 (파이썬)
C:\Users\harryjeong> pip install graphviz

pip install graphviz
conda install graphviz

# 7-3) 파이썬 graphviz path 설정
import os
os.environ['PATH'] += os.pathsep + '/Users/harryjeong/Desktop/release/share/graphviz'

# 7-4) 파이썬 시각화
import graphviz

from sklearn.tree import export_graphviz
export_graphviz(m_dt,                           # 모델명 
                out_file="tree.dot", 
                class_names = df_cancer.target_names,
                feature_names = df_cancer.feature_names, 
                impurity = False, 
                filled = True)

with open("tree.dot", encoding='UTF8') as f:
    dot_graph = f.read()

g1 = graphviz.Source(dot_graph)
g1.render('a1', cleanup=True) 
# --------------------------------------------------------------------------- #