# 时间：2021年6月12日22:38:09
# 贡献者：刘
# 前置文件：problem3_5
# 文件内容：第三道题的代码
# 文件描述：广搜
# 实现内容：剩下2个是超时
# TBD：优化加速
# 通过个数：50


# 读数据
temp=input().split(' ')
node_num,edge_num=int(temp[0]),int(temp[1])
graph=[[0 for _ in range(node_num)] for _ in range(node_num)]
for i in range(edge_num):
    temp=input().split(' ')
    graph[int(temp[0])-1][int(temp[1])-1]=1
    graph[int(temp[1])-1][int(temp[0])-1]=1




# # 读atcoder数据
# node_num=int(input())
# graph=[[0 for _ in range(node_num)] for _ in range(node_num)]
# for i in range(node_num):
#     temp=input()
#     for k in range(node_num):
#         graph[i][k]=int(temp[k])
#         graph[k][i]=int(temp[k])



# 编数据


# node_num,edge_num=3,3
# graph=[[0, 1, 1], [1, 0, 1], [1, 1, 0]]


# node_num,edge_num=2,1
# graph=[[0, 1],[1, 0]]

# node_num,edge_num=7,6
# graph=[[0, 1, 1, 0, 0, 0, 0],
#         [1, 0, 0, 1, 1, 0, 0],
#         [1, 0, 0, 0, 0, 1, 1],
#         [0, 1, 0, 0, 0, 0, 0],
#         [0, 1, 0, 0, 0, 0, 0],
#         [0, 0, 1, 0, 0, 0, 0],
#         [0, 0, 1, 0, 0, 0, 0]]


# node_num=6
# graph=[[0,1,0,1,1,0],
#         [1,0,1,0,0,1],
#         [0,1,0,1,0,0],
#         [1,0,1,0,0,0],
#         [1,0,0,0,0,0],
#         [0,1,0,0,0,0]]


def find_next_nodes(current_loc):
    next_nodes_set=[]
    for i in range(len(graph[current_loc])):
        if graph[current_loc][i]==1:
            is_end=False
            next_nodes_set.append(i)
    next_nodes_set_set[current_loc]=next_nodes_set
    return next_nodes_set

next_nodes_set_set={}
for i in range(node_num):
    find_next_nodes(i)



def arrange_the_sets():
    node_queue_pointer=0
    while node_queue_pointer<len(node_queue):
        next_node_set=next_nodes_set_set[node_queue[node_queue_pointer]]
        if next_node_set!=[]:
            for i in next_node_set:
                if i in node_queue:
                    if abs(node_depth_set[i]-node_depth_set[node_queue[node_queue_pointer]])!=1:
                        return False
                    continue
                node_depth_set[i]=node_depth_set[node_queue[node_queue_pointer]]+1
                node_queue.append(i)
                ownership_graph[i]=node_depth_set[i]
        node_queue_pointer+=1
    return True

# def check_error():
#     for i in range(node_num):
#         next_node_set=next_nodes_set_set[i]
#         for j in next_node_set:
#             if abs(ownership_graph[i]-ownership_graph[j])!=1:
#                 return False
#     return True




max_k=0
for i in range(node_num):
    ownership_graph=[-1 for _ in range(node_num)]
    ownership_graph[i]=0
    node_queue=[i]
    node_depth_set=[-1 for _ in range(node_num)]
    node_depth_set[i]=0
    x=arrange_the_sets()
    if x==True:
        max_k=max(max_k,len(set(ownership_graph)))
if max_k!=0:
    print(max_k)
else:
    print(-1)