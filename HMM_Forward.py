####HMM估值问题_前向算法###
import numpy as np
# 转移概率：a，观测概率：b，初始概率：pi 观测序列：o 
#A = np.array([[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]]) #行概率和为1
#B = np.array([[0.5, 0.5], [0.4, 0.6], [0.7, 0.3]])
#PI = np.array([0.2, 0.4, 0.4])   #初始状态概率和为1
#O = np.array([0,1,0])
A=np.array([[1,0,0,0],[0.2,0.3,0.1,0.4],[0.2,0.5,0.2,0.1],[0.8,0.1,0.0,0.1]])
B=np.array([[1,0,0,0,0],[0,0.3,0.4,0.1,0.2],[0,0.1,0.1,0.7,0.1],[0,0.5,0.2,0.1,0.2]])
PI=np.array([0.2,0.3,0.1,0.4])
O=np.array([1,3,2,0])
def HMM_forward(A, B, PI, O):
    n = np.shape(PI)[0] # 状态个数，即三个盒子
    m = np.shape(O)[0] # 观测序列长度
    alpha = np.zeros((m, n))  # alpha[i][j]:i表示时间，j表示所处的隐状态
    for i in range(n):   #计算alpha（t=0）
        alpha[0][i] = PI[i] * B[i][O[0]]  #即初始时刻（t=0）时，处于隐状态i并且激发可见符号O(t=0)的概率

    for t in range(1,m): #由alpha[t-1]计算alpha[t]（迭代），t from 1 to m(观测序列（可见符号序列）长度）
        for i in range(n):
            temp = 0.0
            for j in range(n):
                temp = temp + alpha[t-1][j] * A[j][i]  #t-1时刻处于隐状态j，t时刻转移为i的概率和
            temp = temp * B[i][O[t]]
            alpha[t][i] = temp                  #alpha[t][i]={max_j alpha[t-1][j]*a[j][i]}*b[j][O(t)]
    print(alpha)
    return alpha      

if __name__ == '__main__':
    alpha = HMM_forward(A, B, PI, O)
    p = 0.0
    for i in range(4):
        p += alpha[3][i]
    print(p)
