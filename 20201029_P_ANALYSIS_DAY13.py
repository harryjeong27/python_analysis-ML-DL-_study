# ANN 모델의 교차 검증 과정
# 1) data load
from mglearn.datasets import load_extended_boston
df_boston_et_x, df_boston_et_y = load_extended_boston()

# 2 data scaling
m_sc = standard()
m_sc.fit(df_boston_et_x)
df_boston_et_x = m_sc.transform(df_boston_et_x)

# 3 자동 학습 중단 모델 생성
earlystopping = EarlyStopping(monitor="mean_squared_error", 
                              patience=10, 
                              verbose=1, 
                              mode='auto')

# 4) 교차 검증을 통한 모델링 수행
from sklearn.model_selection import StratifiedKFold   # 분류
from sklearn.model_selection import KFold             # 회귀

kfold = KFold(n_splits=5, shuffle=True)
vscore = []
for train, test in kfold.split(df_boston_et_x, df_boston_et_y) :
    model = Sequential()
    model.add(Dense(52, input_dim = nx, activation = 'relu'))
    model.add(Dense(26, activation = 'relu'))
    model.add(Dense(13, activation = 'relu'))
    model.add(Dense(1))
    
    model.compile(optimizer = 'adam', 
              loss = 'mean_squared_error',
              metrics = ['mean_squared_error', r_square])
    
    model.fit(df_boston_et_x[train], df_boston_et_y[train], 
              epochs = 500, batch_size = 10,
              validation_split=0.2, callbacks=[earlystopping])
    
    vrsquare = model.evaluate(df_boston_et_x[test], df_boston_et_y[test])[2]
    vscore.append(vrsquare)

np.mean(vscore)

# sigmoid => 각 노드에서 다음 노드로 신호를 보낼 때 특정 값 이상일 경우만 보냄
# relu => sigmoid의 대체