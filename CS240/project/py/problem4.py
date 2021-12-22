# 时间：2021年6月13日14:17:24
# 贡献者：刘
# 前置文件：none
# 文件内容：第四道题的代码
# 文件描述：
# 实现内容：用插入法
# TBD：
# 通过个数：ALL




# 读数据
# N=int(input())

N=50







sum_set={}
for x in range(1,N+1):
    for i in range(1,x):
        for j in range(x+1,N+1):
            if i+j==2*x:
                if x in sum_set:
                    sum_set[x].append([i,j])
                else:
                    sum_set[x]=[[i,j]]


list=[i for i in range(1,N+1)]


list_pointer=0
list_history=[]
temp2=0
while list_pointer<N-1:
    list_pointer+=1
    x=list[list_pointer]
    if x in sum_set:
        for i in sum_set[x]:
            if (list.index(i[0])<list.index(x) and list.index(i[1])<list.index(x)) or (list.index(i[0])>list.index(x) and list.index(i[1])>list.index(x)):
                pass
            else:
                max_index=max(list.index(i[0]),list.index(i[1]))
                min_index=min(list.index(i[0]),list.index(i[1]))
                if list not in list_history:
                    list_history.append([])
                    for j in list:
                        list_history[temp2].append(j)
                    temp2+=1
                    temp=list[max_index]
                    list.remove(temp)
                    list.insert(0,temp)
                else:
                    temp=list[min_index]
                    list.remove(temp)
                    list.insert(-1,temp)
                list_pointer=0




list_pointer=1
while list_pointer<=N:
    for i in range(0,list_pointer):
        for j in range(list_pointer+1,N):
            if list[i]+list[j]==2*list[list_pointer]:
                print(list[i],list[list_pointer],list[j])
                print('False')
    list_pointer+=1

print(list)