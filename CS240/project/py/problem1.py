# 时间：2021年6月10日15:57:26
# 贡献者：刘
# 前置文件：problem1_5_6
# 文件内容：第一道题的代码
# 文件描述：循环实现
# 实现内容：封装函数
# TBD：优化
# 通过个数：






# 输入部分代码
N=int(input())
bignum=int(1e9+1)
smallnum=int(-1e9-1)
loc_set=[ [bignum,smallnum] for _ in range(N+1)]                                                                    # n
for i in range(N):                                                                                                  # n 
    temp=input().split(' ')
    temp=[int(temp[0]),int(temp[1])]
    if temp[0]<loc_set[temp[1]][0]:
        loc_set[temp[1]][0]=temp[0]
    if temp[0]>loc_set[temp[1]][1]:
        loc_set[temp[1]][1]=temp[0]

loc_set2=[[0,0]]
for i in range(N+1):                                                                                                # n
    if loc_set[i][0]!=bignum:
        loc_set2.append(loc_set[i])
loc_set2.append([0,0])

temp_list=[[0 for _ in range(2)]  for _ in range(len(loc_set2))]                                                    # n

last_leftbound=loc_set2[-1][0]
last_rightbound=loc_set2[-1][1]
for i in range(len(loc_set2)-2,-1,-1):                                                                              # n
    [current_leftbound,current_rightbound]=loc_set2[i]
    temp_list[i][0]=(current_rightbound-current_leftbound)+min(abs(current_rightbound-last_leftbound)+temp_list[i+1][0],abs(current_rightbound-last_rightbound)+temp_list[i+1][1])
    temp_list[i][1]=(current_rightbound-current_leftbound)+min(abs(current_leftbound-last_leftbound)+temp_list[i+1][0],abs(current_leftbound-last_rightbound)+temp_list[i+1][1])
    last_leftbound,last_rightbound=current_leftbound,current_rightbound

final_len=temp_list[0][0]
print(final_len)