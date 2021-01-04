# -------------------------------- 연 습 문 제 -------------------------------- #
# 연습문제 4.
# cancer data를 knn 모델로 예측, 의미 있는 interaction이 있다면 추가 이후 예측률 변화 확인

# 1) 전체 data 학습
# step 1) Data loading
from sklearn.datasets import load_breast_cancer
df_cancer = load_breast_cancer()
df_cancer.data.shape    # (569, 30)

df_cancer.data
df_cancer.feature_names
df_cancer.target
df_cancer.target_names

X = df_cancer.data
Y = df_cancer.target

# step 2) train, test split
train_x, test_x, train_y, test_y = train_test_split(X, Y,
                                                    train_size = 0.7,
                                                    random_state = 0)

# step 3) 모델 생성 & 데이터 학습 & 평가
m_knn = knn_c(5)
m_knn.fit(train_x, train_y)
m_knn.score(test_x, test_y)    # 0.947

# 2) scaling data 학습
# 2-1) standard 방식
# step 1) scaling data 생성
from sklearn.preprocessing import StandardScaler as standard
m_sc1 = standard()
m_sc1.fit(train_x)
train_x_sc1 = m_sc1.transform(train_x)

m_sc1.fit(test_x)
test_x_sc1 = m_sc1.transform(test_x)

# step 2) 데이터 학습 & 평가
m_knn.fit(train_x_sc, train_y)
m_knn.score(test_x_sc, test_y)    # 0.947 => standadrd 경우 scaling 전이랑 같음

# 2-2) min, max 방식
# step 1) scaling data 생성
from sklearn.preprocessing import MinMaxScaler as minmax
m_sc2 = minmax()
m_sc2.fit(train_x)
train_x_sc2 = m_sc2.transform(train_x)

m_sc2.fit(test_x)
test_x_sc2 = m_sc2.transform(test_x)

# step 2) 데이터 학습 & 평가
m_knn.fit(train_x_sc2, train_y)
m_knn.score(test_x_sc2, test_y)    # 0.959 => min_max 경우 scaling 전보다 상승*

# 3) 전체 interaction 학습 (min_max로 scaling된 상태)
# step 1) 모델 생성
from sklearn.preprocessing import PolynomialFeatures as poly
m_poly = poly(degree = 2)    # 2차항까지
m_poly.fit(train_x_sc2)

train_x_poly = m_poly.transform(train_x_sc2)
test_x_poly = m_poly.transform(test_x_sc2)

col_poly = m_poly.get_feature_names(df_cancer.feature_names)

# step 2) 데이터 학습 & 평가
m_knn.fit(train_x_poly, train_y)
m_knn.score(test_x_poly, test_y)    # 0.964 => 2차항까지 교호작용한 설명변수로 이전 값보다 상승 **

# 4) 선택된 interaction 학습
# 변형된 데이터셋 Rf에 학습 후 feature importance 확인
m_rf = rf_c(random_state = 0)
m_rf.fit(train_x_poly, train_y)
m_rf.score(test_x_poly, test_y)

s1 = Series(m_rf.feature_importances_, index = col_poly)
s1.sort_values(ascending = False)[:9].index.values

# 아래 내용 위의 변수명에 따라 수정해야 할듯 *
train_x_sc_poly_sel = DataFrame(train_x_sc_poly, columns = col_poly).loc[:, col_selected]
test_x_sc_poly_sel = DataFrame(test_x_sc_poly, columns = col_poly).loc[:, col_selected]

m_knn4 = knn(5)
m_knn4.fit(train_x_sc_poly_sel, train_y)
m_knn4.score(test_x_sc_poly_sel, test_y)      # 90

## 전진 선택법
l1 = s1.sort_values(ascending=False).index

collist=[]
df_result=DataFrame()

for i in l1 : 
    collist.append(i)
    train_x_sc_poly_sel = DataFrame(train_x_sc_poly, columns = col_poly).loc[:, collist]
    test_x_sc_poly_sel = DataFrame(test_x_sc_poly, columns = col_poly).loc[:, collist]

    m_knn5 = knn(5)
    m_knn5.fit(train_x_sc_poly_sel, train_y)
    vscore = m_knn5.score(test_x_sc_poly_sel, test_y)  
    
    df1 = DataFrame([Series(collist).str.cat(sep='+'), vscore], index=['column_list', 'score']).T
    df_result = pd.concat([df_result, df1], ignore_index=True)
    
df_result.sort_values(by='score', ascending=False)
# --------------------------------------------------------------------------- #