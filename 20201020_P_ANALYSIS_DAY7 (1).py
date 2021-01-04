# [ 분석 시 고려사항 5. 최적의 매개변수 조합 (grid search) : train/val/test ]
# - 변수의 최적의 조합을 찾는 과정
# - 중첩 for문으로 구현 가능, grid search 기법으로 간단히 구현 가능
# - train/validation/test set으로 분리
# - 매개변수의 선택은 validation set으로 평가

# 예제) grid search - knn iris data
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
# --------------------------------------------------------------------------- #

# -------------------------------- 연 습 문 제 -------------------------------- #
# 연습문제 6. grid search - random forest cancer data set
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
m_grid.best_score_    # 베스트 매개변수 값을 갖는 평가 점수
m_grid.best_params_   # {'max_features': 1, 'min_samples_split': 2}

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