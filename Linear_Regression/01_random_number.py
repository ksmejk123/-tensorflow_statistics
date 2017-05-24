import numpy as np

#random seed 위치 설정
np.random.seed(0)
# random 갯수 5개
rn1  = np.random.rand(5)
print(rn1)
rn2 = np.random.rand(10)
print(rn2)
# seed 재설정
np.random.seed(0)
rn3 = np.random.rand(10)
print('rn2 와 rn3 비교 ',rn3)
print('--------shuffle-------')
x = np.arange(10)
# shuffle 은 data의 순서를 변경
print(x)
np.random.shuffle(x)
print(x)
print('------------sampling------------')
#choice 명령어 사용
# random number range 5 size 5 replace:중복x p(확률):x
ch1 = np.random.choice(5,5, replace=False)
print('ch1=', ch1)
# p 는 합이 1이 되도록
ch2 = np.random.choice(5,10,p=[0.1,0.1,0.3,0.3,0.2])
print('ch2=', ch2)
print('----------rand/randn/randint----------')
# 0~1 사이에서 확률 분포로 실수 난수 생성
rand1 = np.random.rand(10)
print('rand = ', rand1)
# 기대값 0이고 표준편차가 1인 가우시안 표준 정규 분포를 따름
randn2 = np.random.randn(10)
print('randn=',randn2)
#균일 분포의 정수 난수
randint3 = np.random.randint(10,size=10)
print('randint=',randint3)
print('------------unique/bincount----------')
# unique 중복되지 않게 값 출력
unique1 = np.unique([10,9,8,10,9,8,5,5])
print('unique=',unique1)

a = np.array(['a','a','c','b','c','b'])
#return_counts True 각 값을 가진 데이터 갯수 출력
index, count = np.unique(a,return_counts = True)
print('index =',index, 'count = ',count)

#0~4 까지 몇번 나왔는지 count하는 함수
bincount = np.bincount([1,2,3,4,1,2,3,4],minlength=4)
print(bincount)