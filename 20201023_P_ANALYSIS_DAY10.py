# [ 문자 변수 -> 숫자 변수 변경 ]
run profile1
df1 = DataFrame({'col1' : [1, 2, 3, 4],
                 'col2' : ['M', 'M', 'F', 'M'],
                 'y' : ['N', 'Y', 'N', 'Y']})

# 1) if문으로 직접 치환
np.where(df1.col2 == 'M', 0, 1)

# 2) dummy 변수 치환 함수 사용
pd.get_dummies(df1, drop_first = True)

# 3) 기타 함수 사용
mr = pd.read_csv('mushroom.csv', header = None)

# 문자값가지고 있는 컬럼 숫자로 변경
# 3-1)
def f_char(df) :
    target = []
    data = []
    attr_list = []
    for row_index, row in df.iterrows() :
        target.append(row.iloc[0])     # Y값열 분리
        row_data = []
        for v in row.iloc[1:] :        # X값열 하나씩 v로 전달
            row_data.append(ord(v))    # ord를 사용한 숫자 변환 방식
        data.append(row_data)

f_char(mr)

DataFrame(data)
        
# ord
ord('a')    # 97 => 유니크하게 문자별로 고유의 번호를 불러줌
ord('abc')  # Error => 1개의 length인 문자만 가능

# iterrows
# => row에 대한 index와 각 row별 하나씩 꺼내서 전달함 (반복문에 쓰기 좋음)
for row_index, row in mr.iterrows() :
    print(str(row_index) + ':' + str(list(row)))
# 8119:['e', 'k', 's', 'n', 'f', 'n', 'a', 'c', 'b', 'y', 'e', '?', 's', 's', 'o', 'o', 'p', 'o', 'o', 'p', 'b', 'c', 'l']
# => 8119 row에서 e, k, s 순으로 전달

# target까지 숫자와 하는 함수 만들어 보기*

# 3-2) target 포함해서 변환해주고, ord 한계를 넘어 2자 이상 문자도 변환하는 방법
from sklearn.preprocessing import LabelEncoder

df2 = DataFrame({'col1' : [1, 2, 3, 4],
                 'col2' : ['ABC1', 'BCD1', 'BCD2', 'CDF1'],
                 'y' : ['N', 'N', 'N', 'Y']})

f_char(df2)    # Error => ord length

m_label = LabelEncoder()
m_label.fit(df2.col2)          # 1차원만 학습 가능
m_label.transform(df2.col2)    # 값의 unique value마다 서로 다른숫자 부여
# => 열의 unique value 찾고, 순서대로 정렬한 후 서로 다른 숫자 부여하는 방식
# => 위 mushroom data는 모든 값이 같은 속성이라 열 상관 없이 각 문자에 하나의 숫자를 대입하면 됨
# => 실제로는 열별로 봐야함

# 행별로 진행하면 아래 같은 오류 발생
# a b c  => 0 1 2 
# b c d  => 0 1 2 
# c d e  => 0 1 2 

# 4) LabelEncoder에 의한 변환 기법
def f_char2(df) :
    def f1(x) :
        m_label = LabelEncoder()
        m_label.fit(x)
        return m_label.transform(x)
    return df.apply(f1, axis = 0)

f_char2(mr)
f_char2(df2)    # 숫자 컬럼은 변경하지 않도록 수정, dtype으로 해보기*

# 보스턴 주택 데이터 가격 셋
from sklearn.datasets import load_boston
df_boston = load_boston()

df_boston.data
df_boston.feature_names
df_boston.target        # 종속변수가 연속형

print(df_boston.DESCR)

# 2차 interaction이 추가된 확장된 boston data set
from mglearn.datasets import load_extended_boston
df_boston_et_x, df_boston_et_y = load_extended_boston()
# --------------------------------------------------------------------------- #