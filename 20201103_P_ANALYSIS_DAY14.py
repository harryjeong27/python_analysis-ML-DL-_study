import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import keras

# 2.2 단순 선형 회귀의 계수 추정 과정 - 경사하강법
# 1) x, y의 데이터 값
data = [[2, 81], [4, 93], [6, 91], [8, 97]]

x_data = [x_row[0] for x_row in data]   # 공부시간
y_data = [y_row[1] for y_row in data]   # 시험성적

# 2) 초기 계수 생성
a = tf.Variable(tf.random.uniform([1], 0, 10, dtype=tf.float64, seed=0))
b = tf.Variable(tf.random.uniform([1], 0, 100, dtype=tf.float64, seed=0))

# 3) 선형 회귀 식 생성
y = a * x_data + b     # y = predict value

# 4) RMSE 함수 생성
rmse = tf.sqrt(tf.reduce_mean(tf.square(y - y_data)))

# 학습률 값
learning_rate = 0.1

# RMSE 값을 최소로 하는 값 찾기
gradient_decent = tf.compat.v1.train.GradientDescentOptimizer(learning_rate).minimize(rmse)

# 텐서플로를 이용한 학습
with tf.Session() as sess:
 # 변수 초기화
 sess.run(tf.global_variables_initializer())
 
 # 2001번 실행(0번째를 포함하므로)
 for step in range(2001) :
     sess.run(gradient_decent)
     
 # 100번마다 결과 출력
     if step % 100 == 0 :
         print("Epoch: %.f, RMSE = %.04f, 기울기 a = %.4f, y 절편 b= %.4f" % (step,sess.run(rmse), sess.run(a), sess.run(b))) 

# 선형 회귀를 통한 계수의 추정
from sklearn.linear_model import LinearRegression     
m_reg = LinearRegression()
m_reg.fit(np.array(x_data).reshape(-1,1), y_data)

m_reg.coef_        # 2.3
m_reg.intercept_   # 79
# 따라서, y = 2.3 * X + 79 라는 선형 회귀식이 추정됨

# 2.3 로지스틱 회귀
# - 회귀선 추정으로 y의 값을 분류할 수 있음
# - 참, 거짓으로 구성된 factor의 level이 두 개인 종속변수의 형태
# - 로지스틱 회귀식을 sigmoid function이라 함

# 2.4 오차 역전파(backpropagation)
# - 기울기를 추정하는 과정에서 마지막 노드에서나 결정되는 오차를
#   이전 layer들에게 전달하여 해당 layer에서의 오차와 기울기와의 관계를 파악하는 방식
# - 각 가중치에 따라 전달되는 오차도 비례한다는 가정하에
#   오차를 각 가중치에 따라 분해하여 역으로 전달하는 방식

# =============================================================================
# # ANN 모델에서의 Y의 형태에 따른 활성화 함수
# # 1) Y가 연속형
# #   - 활성화 함수 필요 없음 (있어도 오류는 발생 X)
#   
# # 2) Y가 범주형(2개 level)
# #   - 1개의 Y로 학습되는 경우 : 주로 sigmoid function (0 또는 1의 신호로 변환해주므로)  
# #   - 2개의 Y로 분리학습 되는 경우 : softmax function
#   
# # 3) Y가 범주형(3개 level 이상)
# #   - 0, 1, 2로의 신호 변환을 해주는 활성화 함수 없으므로
# #     반드시 레벨의 수 만큼 Y를 분할하여 학습시켜야 함
#     
# # loss 함수
# # 1) 회귀 모델 (Y가 연속형) : MSE 기반 오차 함수 사용
# #    - mean_squared_error : (y-yhat)**2
# #    - mean_absolute_error : |y-yhat|
# #    - mean_absolute_percentage_error 
# #    ....
# 
# # 2) 분류 모델 (Y가 범주형) : crossentropy 계열 함수 (log함수 기반)
# #    - Y의 범주가 2개 : binary_crossentropy
# #    - Y의 범주가 3개 이상 : categorical_crossentropy
# =============================================================================









