# 时间：2021年6月11日11:11:26
# 贡献者：刘
# 前置文件：problem2_1_3
# 文件内容：第二道题的代码
# 文件描述：递归，没wrong answer,剩下1个runtime error，原因是超出递归深度限制
# 实现内容：用delta
# TBD：
# 通过个数：15


import math
# import sys
# sys.setrecursionlimit(4000)


# # 读数据
# temp=input().split(' ')
# input1=[int(temp[0]),int(temp[1])]
# n=input1[0]
# k=input1[1]
# capacity_graph=[[] for _ in range(n)]
# residual_graph=[[] for _ in range(n)]
# for i in range(n):
#     temp=input().split(' ')
#     for j in range(n):
#         capacity_graph[i].append(int(temp[j]))
#         residual_graph[i].append(int(temp[j]))



# 编数据


# input1=[5,7]
# k=input1[1]
# n=input1[0]
# capacity_graph=[[0,1,0,2,0],
#                 [0,0,4,10,0],
#                 [0,0,0,0,5],
#                 [0,0,0,0,10],
#                 [0,0,0,0,0]]
# residual_graph=[[0,1,0,2,0],
#                 [0,0,4,10,0],
#                 [0,0,0,0,5],
#                 [0,0,0,0,10],
#                 [0,0,0,0,0]]



# input1=[4,0]
# k=input1[1]
# n=input1[0]
# capacity_graph=[[0,10,10,0],
# [0,0,0,15],
# [0,5,0,5],
# [0,0,0,0]]
# residual_graph=[[0,10,10,0],
# [0,0,0,15],
# [0,5,0,5],
# [0,0,0,0]]



# input1=[7,1000]
# k=input1[1]
# n=input1[0]
# capacity_graph=[[ 0, 5, 5, 6, 0, 0, 0],
#                 [ 0, 0, 0, 0, 2, 0, 0],
#                 [ 0, 0, 0, 0, 0, 4, 0],
#                 [ 0, 0, 0, 0, 4, 3, 0],
#                 [ 0, 0, 0, 0, 0, 2, 5],
#                 [ 0, 0, 0, 0, 0, 0, 6],
#                 [ 0, 0, 0, 0, 0, 0, 0]]
# residual_graph=[[ 0, 5, 5, 6, 0, 0, 0],
#                 [ 0, 0, 0, 0, 2, 0, 0],
#                 [ 0, 0, 0, 0, 0, 4, 0],
#                 [ 0, 0, 0, 0, 4, 3, 0],
#                 [ 0, 0, 0, 0, 0, 2, 5],
#                 [ 0, 0, 0, 0, 0, 0, 6],
#                 [ 0, 0, 0, 0, 0, 0, 0]]



input1=[10,1000]
k=input1[1]
n=input1[0]
capacity_graph=[[0,96,50,78,7,57,1,14,17,1],
                [85,0,100,56,36,69,66,4,92,3],
                [43,64,0,97,49,16,71,88,7,41],
                [15,51,47,0,24,83,43,40,74,91],
                [77,85,80,68,0,30,45,3,45,11],
                [55,52,5,26,57,0,12,25,50,8],
                [63,35,14,36,78,36,0,38,61,58],
                [40,22,51,47,70,60,55,0,84,63],
                [14,85,69,99,63,37,85,16,0,2],
                [17,64,24,27,25,16,45,81,99,0]]
residual_graph=[[0,96,50,78,7,57,1,14,17,1],
                [85,0,100,56,36,69,66,4,92,3],
                [43,64,0,97,49,16,71,88,7,41],
                [15,51,47,0,24,83,43,40,74,91],
                [77,85,80,68,0,30,45,3,45,11],
                [55,52,5,26,57,0,12,25,50,8],
                [63,35,14,36,78,36,0,38,61,58],
                [40,22,51,47,70,60,55,0,84,63],
                [14,85,69,99,63,37,85,16,0,2],
                [17,64,24,27,25,16,45,81,99,0]]









# 找最大的capacity
max_capacity=0
for i in capacity_graph:
    for j in i:
        if j>max_capacity:
            max_capacity=j

# 确定delta
delta=2**math.floor(math.log(max_capacity,2))


# 开始实现ford算法
# 递归

current_loc_set=[[] for _ in range(n)]
path=[]
passed_loc=[]
def find_a_path_and_flow():
    for i in range(n):                      # 这个地方很慢，可以优化
        current_loc_set[i]=[]
        for j in range(n):
            if residual_graph[i][j]>=delta:
                current_loc_set[i].append(j)
    current_path=[]
    def find_a_path(current_loc):
        if current_loc_set[current_loc]==[]:
            return False
        elif current_loc_set[current_loc][-1]==(n-1):
            current_path.append(n-1)
            return True
        elif current_loc_set[current_loc]!=[]:
            # 用来避免打转
            if current_loc in passed_loc:
                return False
            passed_loc.append(current_loc)
            for j in range(len(current_loc_set[current_loc])):
                label_exist=find_a_path(current_loc_set[current_loc][j])
                if label_exist==True:
                    current_path.insert(0,current_loc_set[current_loc][j])
                    return True
            if label_exist==False:
                return False
    passed_loc=[]
    current_flow=1e6+1
    if find_a_path(0)==True:
        # print('yeah')
        current_path.insert(0,0)
        path.append(current_path)
        for i in range(len(current_path)-1):
            current_flow=min(current_flow,residual_graph[current_path[i]][current_path[i+1]])
        for i in range(len(current_path)-1):
            residual_graph[current_path[i]][current_path[i+1]]-=current_flow
        return current_flow
    else:
        return 0

flow=find_a_path_and_flow()
flow_set=[]
if flow!=0:
    flow_set.append(flow)
while 1:
    if flow==0 and delta==0.5:
        break
    flow=find_a_path_and_flow()
    if flow!=0:
        flow_set.append(flow)
    else:
        delta=delta/2

for interrupt_num in range(1,n):
    if k==0:
        break
    else:
        for i in range(len(path)):
            while 1:
                loc_interrupt=[]
                label_interrupt=0
                for j in range(len(path[i])-1):
                    if residual_graph[path[i][j]][path[i][j+1]]==0:
                        label_interrupt+=1
                        loc_interrupt.append([path[i][j],path[i][j+1]])
                if label_interrupt==interrupt_num:
                    if k>=interrupt_num:
                        k-=interrupt_num
                        flow_set[i]+=1
                        for i2 in range(len(loc_interrupt)):
                            residual_graph[loc_interrupt[i2][0]][loc_interrupt[i2][1]]+=1
                        for j1 in range(len(path[i])-1):
                            residual_graph[path[i][j1]][path[i][j1+1]]-=1
                    else:
                        k=0
                else:
                    break
                if k==0:
                    break


print(sum(flow_set))
print()