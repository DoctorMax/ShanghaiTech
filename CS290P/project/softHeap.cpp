// 文件内容描述：软堆soft heap的C++实现
// 参考链接：[1]Chazelle, Bernard. The soft heap: an approximate priority queue with optimal error rate[J]. Journal of the ACM, 2000, 47(6):1012-1027.
// 作者：刘畅:liuchang3@shanghaitech.edu.cn
// 创建时间：2021年12月20日16:16:04
// 解题思路：参考原作者提供的C源码进行C++实现
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <queue>
#include <list>
#include <algorithm>
#include <iomanip>
#include <ctime>
#include <stack>
#define INFTY 666666666
using namespace std;
// 数据结构定义
// 每个node的list中的元素的数据类型，item list cell
class itemListNode
{
public:
    //这个key是真实值，不会变化的真实值
    int key;
    itemListNode *next;
};
// 软堆的节点node的数据类型
class node
{
public:
    //ckey的意思是common key，近似值，是这个node的list中所有元素的key的一个上界，可以变
    int ckey;
    int rank;
    // 这个技巧用于将高度数的节点表示为二叉节点
    // node是一个rank-1的软队列的根节点
    // 软队列是指从root往next走，不管child
    node *next;
    // 子节点的rank比node的rank小1
    node *child;
    // next的ckey要保证比child的ckey小
    // node的list的头一个节点
    itemListNode *il;
    // node的list的尾节点
    itemListNode *il_tail;
    node()
    {

        next = NULL;
        child = NULL;
        il = NULL;
        il_tail = NULL;
        rank = 0;
        ckey = 0;
    }
    bool isLeaf()
    {
        if (next == NULL && child == NULL)
        {
            return true;
        }
        else
        {
            return false;
        }
    }
};
// 优先队列中最顶层的数据结构，各自指向根节点，head之间是双向list。headlist是根据rank值进行从小到大排序的
class head
{
public:
    // 这个rank同根节点的rank
    int rank;
    head *next;
    head *prev;
    // 指向根节点
    node *queue;
    // 指向在head list中，排在当前head后面（和自身）的head中，其根节点ckey最小的head
    head *suffix_min;
    head()
    {

        queue = NULL;
        next = NULL;
        prev = NULL;
        suffix_min = NULL;
        rank = 0;
    }
};
// 软堆的数据结构，指由许多head带领的一系列堆
class softHeap
{
public:
    // headlist因为是双向链表，所以header是链表的头
    head *header;
    // headlist的链表尾
    head *tail;
    // 和错误率相关的一个参数
    int r;
    softHeap(int temp_r)
    {
        header = new head();
        tail = new head();
        tail->rank = INFTY;
        header->next = tail;
        tail->prev = header;
        r = temp_r;
    }

    // 从当前的head h向前一个个head更新suffix_min指针的指向，前提是h后面的head的suffix_min指针都是对的
    void fix_minlist(head *h)
    {
        head *tmp_min;
        // 当前head h后面的head的suffix_min
        if (h->next == tail)
        {
            tmp_min = h;
        }
        else
        {
            tmp_min = h->next->suffix_min;
        }
        // 往前更新
        while (h != header)
        {
            if (h->queue->ckey < tmp_min->queue->ckey)
            {
                // 如果当前head h的ckey比较小，那么h就是新的最小的
                tmp_min = h;
            }
            h->suffix_min = tmp_min;
            // 往前翻
            h = h->prev;
        }
    }

    // q是一个queue的根节点，我们要把这个queue合并进软堆heap中
    void meld(node *q)
    {
        head *h;
        // 前一个
        head *prevhead;
        // 后一个，从第一个head开始往后搜
        head *tohead = header->next;
        node *top;
        node *bottom;

        // 因为软堆的headlist要根据rank从小到大排列，所以我们要先从前往后搜索headlist，找到合适的位置去插入新的queue
        while (q->rank > tohead->rank)
        {
            tohead = tohead->next;
        }

        // 这是应该插入的两个head中间的前一个head
        prevhead = tohead->prev;

        // 如果以q为根节点的queue的rank和原本的headlist中的某一个head具有相同rank，那么开始合并
        while (q->rank == tohead->rank)
        {
            // ckey小的放上面
            if (tohead->queue->ckey > q->ckey)
            {
                top = q;
                bottom = tohead->queue;
            }
            else
            {
                top = tohead->queue;
                bottom = q;
            }

            // 生成一个rank+1的节点，ckey=两个根节点中ckey更大的一个
            q = new node();
            q->ckey = top->ckey;
            q->rank = top->rank + 1;

            // 看起来就像next是左子节点（ckey较小的），child是右子节点
            q->child = bottom;
            q->next = top;
            // 新根节点q的list继承自next（ckey较小的）
            q->il = top->il;
            q->il_tail = top->il_tail;

            // 检查下一个head，如果当前这个rank+1后和下一个head的rank相同，则重复操作
            tohead = tohead->next;
        }
        // 如果不需要进行合并，那么直接创建一个新的head，然后插入两个已有的head之间，并且指向根节点
        if (prevhead == tohead->prev)
        {
            h = new head();
        }

        // 如果合并了，那么直接重复利用之前的那个head
        else
        {
            h = prevhead->next;
        }
        // 将前后链接起来
        h->queue = q;
        h->rank = q->rank;
        h->prev = prevhead;
        h->next = tohead;
        prevhead->next = h;
        tohead->prev = h;

        // 重新计算一次从head h往前的所有head的suffix_min指针的指向
        fix_minlist(h);
    }

    // 筛一遍node v和它的下面所有next的list和ckey，进行重新设置，并且删掉部分node节点
    node *sift(node *v)
    {
        node *tmp;

        // 清空v的list，因为需要重新设置，所以没有用了
        v->il = NULL;
        v->il_tail = NULL;

        // 如果v是叶子节点，那么设置v的ckey为无穷大，这样就可以保证在后续的操作中，v依然在堆的底部，最后才会被弹出
        if (v->isLeaf())
        {
            v->ckey = INFTY;
            return v;
        }

        // 如果v不是叶子节点，那么对v的next进行sift操作
        v->next = sift(v->next);

        // 在上面一行返回后，我们得到了一个新的v->next，但是v->next的ckey可能是一个很大的值，这可能会违反堆的顺序规则（规则是我们定义next的ckey要比child小）
        if (v->next->ckey > v->child->ckey)
        {
            tmp = v->child;
            v->child = v->next;
            v->next = tmp;
        }

        // 把v->next的list直接传给v的list，把v->next的ckey也直接传给v，因为此处v的list已经被清空了。
        // 此处也受益于next的ckey比child的小，所以child作为v的子节点也符合小根堆的规则
        v->il = v->next->il;
        v->il_tail = v->next->il_tail;
        v->ckey = v->next->ckey;
        // 如果v的rank比预设数r大并且满足###条件，那么就再进行一次sift操作。
        if (v->rank > r &&
            (v->rank % 2 == 1 || v->child->rank < v->rank - 1))
        {

            v->next = sift(v->next);

            // 再进行一次调整
            if (v->next->ckey > v->child->ckey)
            {
                tmp = v->child;
                v->child = v->next;
                v->next = tmp;
            }
            // 如果v->next不是叶子节点并且v->next的list不为空
            if (v->next->ckey != INFTY && v->next->il != NULL)
            {
                // 将v->next的list接在v的list前面，然后赋给v
                v->next->il_tail->next = v->il;
                v->il = v->next->il;
                // 如果v本来的list是空的，那么就指定list尾是v->next的list尾。如果不是空的那么就不用变了
                if (v->il_tail == NULL)
                {
                    v->il_tail = v->next->il_tail;
                }
                // 把v->next的ckey给v
                v->ckey = v->next->ckey;
            }
        }
        // 如果v的next和child都是叶子结点，那么把这两个节点删掉，但是我们不改变v的rank，因为此时rank是从上往下定义的
        if (v->child->ckey == INFTY)
        {
            if (v->next->ckey == INFTY)
            {
                v->child = NULL;
                v->next = NULL;
            }
            // 如果child是叶子节点，但next不是（意味着只有child需要被删除时），那么删掉v->next节点和v->child节点，把v->next的next和child作为v的新next和child
            else
            {
                v->child = v->next->child;
                v->next = v->next->next;
            }
        }

        return v;
    }

    // insert的实现方法为：新建一个只有一个node的树，然后挂在一个新的head下，然后合并进软堆
    void insert(int new_key)
    {
        node *q = new node();
        ;
        itemListNode *l = new itemListNode();

        // 生成新node的list，然后把新的key扔进去，作为一个新的list元素
        l->key = new_key;
        l->next = NULL;

        // 生成新node
        q->rank = 0;
        // 因为list里只有一个key，那么ckey就等于key
        q->ckey = new_key;
        q->il = l;
        q->il_tail = l;

        // 进行合并（将一个树合并进软堆）
        meld(q);
    }

    // 返回ckey最小的item
    int popMin()
    {
        node *s, *tmp;
        int min;
        int childcount;

        // h是head list中root节点ckey最小的head
        head *h = header->next->suffix_min;

        // 如果h的root的item list是空的，那么我们就要进行操作
        while (h->queue->il == NULL)
        {

            tmp = h->queue;
            childcount = 0;

            // 统计这个软队列的孩子层数
            while (tmp->next != NULL)
            {
                tmp = tmp->next;
                childcount++;
            }

            // 检查这个软队列的rank是否合法
            if (childcount < h->rank / 2)
            {
                // 如果不合法，把这个软队列和它的head剔除出软堆结构中
                h->prev->next = h->next;
                h->next->prev = h->prev;
                fix_minlist(h->prev);
                tmp = h->queue;
                // 把root节点的child节点合并进软堆中，并且继续往下循环，把软队列中所有node的child一一合并进软堆中
                while (tmp->next != NULL)
                {
                    meld(tmp->child);
                    tmp = tmp->next;
                }
            }
            // 如果软队列的rank合法，通过调用sift把下面的item list提上来，放到root节点中
            else
            {
                h->queue = sift(h->queue);
                // 如果sift操作结束后，root节点本身就是叶子节点的话，那么这个软队列就全都删掉
                if (h->queue->ckey == INFTY)
                {
                    h->prev->next = h->next;
                    h->next->prev = h->prev;
                    h = h->prev;
                }
                fix_minlist(h);
            }

            // 如果suffix_min发生了变化，那么我们要保证新的这个suffix_min不存在item list为空的情况
            h = header->next->suffix_min;
        }

        // 此时h就是整个head list中拥有最小的ckey的head。从这个head的root的item list中获取第一个key的值。
        min = h->queue->il->key;

        // 把这个item从item list中移除
        h->queue->il = h->queue->il->next;

        // 如果这个item是这个item list中唯一的一个item,那么把item list的尾也指向空
        if (h->queue->il == NULL)
        {
            h->queue->il_tail = NULL;
        }

        return min;
    }

    // 删除某一个item
    bool deleteOne(int key_num, int new_key)
    {
        stack<int> temp_stack;
        int value;
        for (int i = 0; i < key_num; i++)
        {
            value = popMin();
            if (value == new_key)
            {
                while (!temp_stack.empty())
                {
                    insert(temp_stack.top());
                    temp_stack.pop();
                }
                return true;
            }else{
                temp_stack.push(value);
            }
        }
        while (!temp_stack.empty())
        {
            insert(temp_stack.top());
            temp_stack.pop();
        }
        return false;
    }
};

int main()
{
    int mode;
    cout << "Choose the mode:" << endl
         << "0:enter manually" << endl
         << "1:automatically generated" << endl
         << "2:read the file" << endl;
    cin >> mode;
    switch (mode)
    {
    case 0:
    {
        // 手动输入模式
        softHeap *heap;
        int r;
        char op;
        int value;
        bool status;
        int node_num = 0;
        cout << "Enter (int)r:" << endl;
        cin >> r;
        heap = new softHeap(r);
        while (scanf("%c", &op) == 1)
        {
            switch (op)
            {
            case 'i':
                scanf(" %d", &value);
                heap->insert(value);
                node_num++;
                break;
            case 'p':
                if (node_num == 0)
                {
                    cout << "The queue is empty!" << endl;
                    break;
                }
                value = heap->popMin();
                cout << value << endl;
                node_num--;
                break;
            case 'd':
                scanf(" %d", &value);
                if (node_num == 0)
                {
                    cout << "The queue is empty!" << endl;
                    break;
                }
                status = heap->deleteOne(node_num, value);
                if (status)
                {
                    cout << "Sucess!" << endl;
                    node_num--;
                }
                else
                {
                    cout << "Fail!" << endl;
                }

                break;
            default:
                break;
            }
        }
        break;
    }
    case 1:
    {
        // 自动生成模式
        clock_t start, end;
        int r;
        int size;
        char op;
        int value;
        static const int max_size = 100000000;
        int *inserts = new int[max_size];
        int *deletes_old = new int[max_size];
        int *deletes_new = new int[max_size];
        int i;
        int error_num = 0;
        softHeap *heap;
        cout << "Enter (int)size:" << endl;
        cin >> size;
        cout << "Enter (int)r:" << endl;
        cin >> r;
        heap = new softHeap(r);
        // 随机数种子
        srand(time(NULL));
        // C++自带的优先队列，使用随机生成的数据，进行对比
        priority_queue<int, vector<int>, greater<int>> pq;
        // 先随机生成数据点
        for (i = 0; i < size; i++)
        {
            inserts[i] = rand();
        }
        // 开始计soft heap时
        start = clock();
        // 使用soft heap
        for (i = 0; i < size; i++)
            heap->insert(inserts[i]);
        for (i = 0; i < size; i++)
            deletes_new[i] = heap->popMin();
        // soft heap结束
        end = clock();
        cout << "soft heap duration: " << end - start << "ms" << endl;
        // 使用传统方法
        start = clock();
        for (i = 0; i < size; i++)
            pq.push(inserts[i]);
        for (i = 0; i < size; i++)
        {
            deletes_old[i] = pq.top();
            pq.pop();
        }
        end = clock();
        cout << "traditional heap duration: " << end - start << "ms" << endl;
        // cout << "old"
        //  << "|"
        //  << "new" << endl;
        // 比较两个输出的错误率
        for (int i = 0; i < size; i++)
        {
            // cout << deletes_old[i] << "|" << deletes_new[i] << endl;
            if (deletes_old[i] != deletes_new[i])
            {
                error_num++;
            }
        }
        cout << "error rate=" << (float)error_num / size << endl;
        delete (inserts);
        delete (deletes_new);
        delete (deletes_old);
        break;
    }
    case 2:
    {
        // 读取文件模式
        int r;
        char op;
        int value;
        int node_num = 0;
        static const int size = 100000;
        int inserts[size];
        int deletes[size];
        int i, z;
        bool status;
        softHeap *heap;
        FILE *fp;
        cout << "Enter (int)r:" << endl;
        cin >> r;
        heap = new softHeap(r);
        fp = fopen("input.txt", "r");
        remove("output.txt");
        ofstream output("output.txt", ios_base::app);
        while (fscanf(fp, "%c", &op) == 1)
        {
            switch (op)
            {
            case 'i':
                fscanf(fp, " %d\n", &value);
                heap->insert(value);
                node_num++;
                break;
            case 'p':
                fscanf(fp, "\n");
                if (node_num == 0)
                {
                    cout << "The queue is empty!" << endl;
                    break;
                }
                value = heap->popMin();
                output << value << "\n";
                cout << value << endl;
                node_num--;
                break;
            case 'd':
                fscanf(fp, " %d\n", &value);
                if (node_num == 0)
                {
                    cout << "The queue is empty!" << endl;
                    break;
                }
                status = heap->deleteOne(node_num, value);
                if (status)
                {
                    output << "Sucess!"
                           << "\n";
                    cout << "Sucess!" << endl;
                    node_num--;
                }
                else
                {
                    output << "Fail!"
                           << "\n";
                    cout << "Fail!" << endl;
                }
                break;
            default:
                break;
            }
        }
        fclose(fp);
        output.close();
        break;
    }
    }

    system("pause");
}
