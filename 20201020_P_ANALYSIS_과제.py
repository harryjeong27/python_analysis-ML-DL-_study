# 실습 - ThoraricSurgery_rawdata
# 그리드 서치 및 히트맵
df1 = pd.read_csv('ThoraricSurgery.csv', header = None)

# 1. 데이터 보고 분석계획
# 종속변수가 있으므로 지도학습이며 종속변수가 범주형이므로 분류학습으로 수행하기
# Y = df1[17]이고, 나머지 17개가 X
df1.shape    # (470, 18)

# RF 수행 -> 1차 튜닝(변수 선택 및 가공) -> 모델 학습 및 예측 -> 추가 튜닝 예정

# 2. RF 수행
# 2-1) 데이터 분리
Y = df1[17]
X = df1.drop(17, axis = 1)

train_x, test_x, train_y, test_y = train_test_split(X, Y, random_state = 0)

# 2-2) 모델 학습
m_rf = rf_c(random_state = 0)
m_rf.fit(train_x, train_y)

# 2-3) 모델 평가
m_rf.score(test_x, test_y)    # 0.906*

# 3. 튜닝 (변수 선택 및 가공)
# 데이터 분석 시 고려사항
# 1) 변수 선택
# 2) 변수 표준화
# 3) interaction
x1 x2 x1^2 x2^2 x1x2
# 4) 매개변수 튜닝(그리드 서치) : train/val/test*
# 5) 교차검증(cross validation)
# 6) 변수 변형

# 4. 모델 학습 및 예측 by GridSearch
# 4-1) data split
trainval_x, test_x, trainval_y, test_y = train_test_split(X, Y, random_state = 0)
train_x, val_x, train_y, val_y = train_test_split(trainval_x, trainval_y, random_state = 0)

# 4-2) 모델 학습 및 매개변수 튜닝 (min_samples_split, max_features)
# min_samples_split : 2 ~ 10
# max_features : 1 ~ 17
from sklearn.model_selection import GridSearchCV

# 4-2-1) 모델 생성
m_rf = rf_c()

# 4-2-2) 그리드 서치 기법을 통한 매개변수 조합 찾기
v_params = {'min_samples_split' : np.arange(2, 11),
            'max_features' : np.arange(1, 18)}

# 4-2-3) 그리드 서치 모델 생성
m_grid = GridSearchCV(m_rf, v_params, cv = 5)

# 4-2-4) 그리드 서치에 의한 모델 학습
m_grid.fit(trainval_x, trainval_y)

# 4-2-5) 결과 확인
m_grid.best_score_    # 0.838 => 낮음
m_grid.best_params_

df_result = DataFrame(m_grid.cv_results_)

# 4-2-6) 최종 평가
m_grid.score(test_x, test_y)    # 0.890 => 낮음

# 4-2-7) 그리드 서치 결과 시각화
df_result.mean_test_score
arr_score = np.array(df_result.mean_test_score).reshape(17, 9)

import mglearn
plt.rc('figure', figsize = (10, 10))
plt.rc('font', size = 7)

mglearn.tools.heatmap(arr_score,                
                      'min_samples_split',  
                      'max_features',     
                      v_params['min_samples_split'],
                      v_params['max_features'],
                      cmap = 'viridis')

# 결론 => RF의 기본모델과 그리드서치만을 실행한 결과, 기본모델에서 0.906으로
#     => 그리드서치한 결과 0.890으로 높은 예측력 보여줌
#     => 더 많은 테스트를 해봐야 하지만 현재 상태로는 기본모델이 더 예측력 좋은 모델로 보여짐
