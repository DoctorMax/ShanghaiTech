data=open('./data.txt','w')
for i in range(1,200001):
    data.write(str(i))
    data.write(" ")
    data.write(str(i))
    data.write("\n")