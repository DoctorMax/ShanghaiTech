// 时间：2021年6月13日21:18:18
// 贡献者：刘
// 前置文件：problem_2
// 文件内容：第一道题的C代码
// 文件描述：循环实现
// 实现内容：
// TBD：
// 通过个数：all




#include <iostream>
#include <vector>
#include <string>
#include <stdio.h>

using namespace std;


#define bignum (long long) (1e9+1)
#define smallnum (long long) (-1e9-1)
#define max_N (long long) 200002


long long min(long long a,long long b)
{
    if (a>=b)
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

    long long N;
    long long temp0;
    long long temp1;
    long long color_num=0;
    long long last_leftbound=0;
    long long last_rightbound=0;
    long long current_leftbound;
    long long current_rightbound;
    long long final_len;
    long long i;
    scanf("%lld",&N);
    long long **loc_set;
    loc_set=new long long* [max_N];
    for (i=0;i<max_N;i++)
    {
        loc_set[i]=new long long [2];
        loc_set[i][0]=bignum;
        loc_set[i][1]=smallnum;
    }
    long long **loc_set2;
    loc_set2=new long long* [max_N];
    for (i=0;i<max_N;i++)
    {
        loc_set2[i]=new long long [2];
        loc_set2[i][0]=(long long)0;
        loc_set2[i][1]=(long long)0;
    }
    long long **temp_list;
    temp_list=new long long* [max_N];
    for (i=0;i<max_N;i++)
    {
        temp_list[i]=new long long [2];
        temp_list[i][0]=(long long)0;
        temp_list[i][1]=(long long)0;
    }



    for ( i=0;i<N;i++)
    {
        scanf("%lld%lld",&temp0,&temp1);
        if (temp0<loc_set[temp1][0])
        {
            loc_set[temp1][0]=temp0;
        }
        if (temp0>loc_set[temp1][1])
        {
            loc_set[temp1][1]=temp0;
        }
        
    }

    for ( i=1;i<N+1;i++)
    {
        if (loc_set[i][0]!=bignum)
        {
            color_num+=1;
            loc_set2[color_num][0]=loc_set[i][0];
            loc_set2[color_num][1]=loc_set[i][1];
            
        }
    }





    for ( i=color_num;i>-1;i--)
    {
        current_leftbound=loc_set2[i][0];
        current_rightbound=loc_set2[i][1];

        temp_list[i][0]=(current_rightbound-current_leftbound)+min(abs(current_rightbound-last_leftbound)+temp_list[i+1][0],abs(current_rightbound-last_rightbound)+temp_list[i+1][1]);
        temp_list[i][1]=(current_rightbound-current_leftbound)+min(abs(current_leftbound-last_leftbound)+temp_list[i+1][0],abs(current_leftbound-last_rightbound)+temp_list[i+1][1]);

        last_leftbound=current_leftbound;
        last_rightbound=current_rightbound;

    }
    final_len=temp_list[0][0];
    printf("%lld",final_len);



    delete[]loc_set;
    delete[]loc_set2;
    delete[]temp_list;
    return 0;
}