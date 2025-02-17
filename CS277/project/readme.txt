本组的project代码部分总共一个文件。
使用的数据来源是老师提供的crunchbase数据。
具体使用的数据文件包括：
    cb_acquisitions.csv
    cb_degrees.csv
    cb_ipos.csv
    cb_objects.csv
    cb_people.csv
此处仅保留每个数据的前100项，可能导致程序运行效果极差，但是应该不会产生错误。
代码中有进行相对详细的注释描述和分块标注，此处重新声明各个部分的功能：
    1.初始化，主要作用是调用各种框架和使用到的库。
    2.读取数据，从当前目录下进行数据读取，并且观察前几项数据和特征数量。
    3.数据处理，分为一下几个部分：
        a.洗公司数据，挑选出可能用到的特征，并且进行基本的特征提取和错值处理。
        b.洗创始人数据，挑选出可能用到的特征。
        c.洗上市信息数据，挑选出可能用到的特征。
        d.洗收购数据，挑选出可能用到的特征。
        e.洗学位数据，将非格式化的学位设为未知。
        f.将数据进行融合，并且提取我们设想的特征，确定企业成功的定义。
        g.提取2000年以后的数据。
    4.数据分析，粗略检查数字特征的相关性。
    5.画图
        a.画出不同国家的初创企业数量和成功率。
        b.画出不同城市的初创企业数量和成功率。
        c.画出不同行业的初创企业数量和成功率。
        d.画出企业成功与否与创始人的学位水平的关系。
        e.画出每年的初创企业分布。
    6.为了正确选择特征而进行的数据处理，先对缺省数据项进行删除处理，然后对特征的相关性进行分析，得到相关性高的一些特征。
    7.根据相关性选择特征重新进行数据处理，删去相关性低的特征，然后再对原始数据的缺省数据项进行删除处理。
    8.准备训练用的数据集。
    9.训练4中模型
    10.进行预测
    11.绘制4个混淆矩阵，对训练结果进行衡量和分析
    