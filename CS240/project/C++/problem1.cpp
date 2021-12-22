#include <iostream>
#include <vector>
#include <string>
#include <stdio.h>

using namespace std;


#define bignum int(1e9+1)
#define smallnum int(-1e9-1)
#define max_N 200001


int min(int a,int b)
{
    if (a>b)
    {
        return b;
    }
    else
    {
        return a;
    }
}





int main()
{

    int N;
    int temp0;
    int temp1;
    int color_num=0;
    int last_leftbound=0;
    int last_rightbound=0;
    int current_leftbound;
    int current_rightbound;
    int final_len;
    scanf("%d",&N);
    long long **loc_set;
    loc_set=new long long* [max_N];
    for (int i=0;i<max_N;i++)
    {
        loc_set[i]=new long long [2];
        loc_set[i][0]=bignum;
        loc_set[i][1]=smallnum;
    }
    long long **loc_set2;
    loc_set2=new long long* [max_N];
    for (int i=0;i<max_N;i++)
    {
        loc_set2[i]=new long long [2];
        loc_set2[i][0]=0;
        loc_set2[i][1]=0;
    }
    long long **temp_list;
    temp_list=new long long* [max_N];
    for (int i=0;i<max_N;i++)
    {
        temp_list[i]=new long long [2];
        temp_list[i][0]=0;
        temp_list[i][1]=0;
    }



    for (int i=0;i<N;i++)
    {
        scanf("%d%d",&temp0,&temp1);
        if (temp0<loc_set[temp1][0])
        {
            loc_set[temp1][0]=temp0;
        }
        if (temp0>loc_set[temp1][1])
        {
            loc_set[temp1][1]=temp0;
        }
        
    }
    for (int i=1;i<N+1;i++)
    {
        if (loc_set[i][0]!=bignum)
        {
            color_num+=1;
            loc_set2[color_num][0]=loc_set[i][0];
            loc_set2[color_num][1]=loc_set[i][1];
            
        }
    }
    for (int i=color_num;i>-1;i--)
    {
        current_leftbound=loc_set2[i][0];
        current_rightbound=loc_set2[i][1];
        temp_list[i][0]=(current_rightbound-current_leftbound)+min(abs(current_rightbound-last_leftbound)+temp_list[i+1][0],abs(current_rightbound-last_rightbound)+temp_list[i+1][1]);
        temp_list[i][1]=(current_rightbound-current_leftbound)+min(abs(current_leftbound-last_leftbound)+temp_list[i+1][0],abs(current_leftbound-last_rightbound)+temp_list[i+1][1]);
        last_leftbound=current_leftbound;
        last_rightbound=current_rightbound;
    }
    final_len=temp_list[0][0];
    printf("%d\n",final_len);
    delete[]loc_set;
    delete[]loc_set2;
    delete[]temp_list;
    return 0;
}