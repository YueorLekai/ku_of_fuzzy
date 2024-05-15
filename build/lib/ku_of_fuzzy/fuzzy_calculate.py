import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt


# 第一部分：标准化
# 标准差标准化
def standardize_columns(df):
    """
       标准差标准化：使用每列的均值和标准差对DataFrame的列进行标准化。
       参数:
           df (DataFrame): 需要标准化的pandas DataFrame。
       返回:
           DataFrame: 标准化后的DataFrame。
       """
    # 计算每列的均值和标准差
    means = df.mean()
    stds = df.std(ddof=0)

    # 标准差标准化
    standardized_df = (df - means) / stds
    # 数据需要自行映射到[0,1]之间

    return standardized_df


# 最小-最大标准化(极值标准化)
def min_max_normalize(df):
    """
        最小-最大标准化：对DataFrame的列进行极值标准化。
        参数:
            df (DataFrame): 需要标准化的pandas DataFrame。
        返回:
            DataFrame: 标准化后的DataFrame。
        """
    df_normalized = (df - df.min()) / (df.max() - df.min())
    return df_normalized


# 极差标准化(减去均值再除以最大与最小的差)
def mean_normalization(df):
    """
    极差标准化：减去均值再除以最大与最小的差。
    参数:
        df (DataFrame): 需要标准化的pandas DataFrame。
    返回:
        DataFrame: 标准化后的DataFrame。
    """
    # 计算列的最大值、最小值与均值
    max_value = df.max()
    min_value = df.min()
    mean_value = df.mean()
    # 应用极差标准化
    df = (df - mean_value) / (max_value - min_value)
    # 数据需要自行映射到[0,1]之间
    return df


# 最大值规格化
def max_normalize(df):
    """
    最大值规格化：使用每列的最大值对DataFrame的列进行规格化。
    参数:
        df (DataFrame): 需要规格化的pandas DataFrame。
    返回:
        DataFrame: 规格化后的DataFrame。
    """
    # 使用df.max()获取每列的最大值
    max_values = df.max()
    # 对DataFrame的每个数值列进行最大值规格化
    df = df / max_values
    return df


# 第二部分：建立相似度矩阵

# 余弦相似度法
def cosine_similarity(dataframe1, decimals=3):
    """
    计算dataframe1中每一行与其他行之间的余弦相似度，并返回一个m*m的DataFrame。

    Args:
        dataframe1 (pd.DataFrame): 输入的m*n的DataFrame，元素值在[0, 1]之间。
        decimals (int): 四舍五入到小数点后的位数，默认为3。

    Returns:
        pd.DataFrame: 一个m*m的DataFrame，其中Xij表示dataframe1的第i行和第j行之间的余弦相似度。
    """
    # 将DataFrame转换为NumPy数组
    array1 = dataframe1.values

    # 计算每个向量的模长
    norms = np.linalg.norm(array1, axis=1)

    # 初始化余弦相似度矩阵
    m = len(array1)
    cosine_matrix = np.zeros((m, m))

    # 计算余弦相似度
    for i in range(m):
        for j in range(m):
            dot_product = np.dot(array1[i], array1[j])
            cosine_matrix[i][j] = dot_product / (norms[i] * norms[j])

    # 四舍五入余弦相似度矩阵中的值
    cosine_matrix_rounded = np.round(cosine_matrix, decimals=decimals)

    # 创建一个m*m的DataFrame来存储结果
    dataframe2 = pd.DataFrame(cosine_matrix_rounded, index=dataframe1.index, columns=dataframe1.index)

    return dataframe2


# 皮尔逊相关系数法
def pearson_similarity(dataframe1, decimals=3):
    """
    计算dataframe1中每一行与其他行之间的皮尔逊相关系数，并返回一个m*m的DataFrame。

    Args:
        dataframe1 (pd.DataFrame): 输入的m*n的DataFrame。
        decimals (int): 四舍五入到小数点后的位数，默认为3。

    Returns:
        pd.DataFrame: 一个m*m的DataFrame，其中Xij表示dataframe1的第i行和第j行之间的皮尔逊相关系数。
    """
    # 计算每一行的平均值
    row_means = dataframe1.mean(axis=1)

    # 初始化相关系数矩阵
    m = len(dataframe1)
    pearson_matrix = np.zeros((m, m))

    # 计算皮尔逊相关系数
    for i in range(m):
        for j in range(m):
            numerator = np.sum(abs((dataframe1.iloc[i] - row_means[i]) * (dataframe1.iloc[j] - row_means[j])))
            denominator = np.sqrt(
                np.sum((dataframe1.iloc[i] - row_means[i]) ** 2) * np.sum((dataframe1.iloc[j] - row_means[j]) ** 2))
            if denominator != 0:
                pearson_matrix[i][j] = numerator / denominator
            else:
                pearson_matrix[i][j] = np.nan  # 如果分母为0，则赋值为NaN

    # 四舍五入相关系数矩阵中的值
    pearson_matrix_rounded = np.round(pearson_matrix, decimals=decimals)

    # 创建一个m*m的DataFrame来存储结果
    dataframe2 = pd.DataFrame(pearson_matrix_rounded, index=dataframe1.index, columns=dataframe1.index)

    return dataframe2


# 距离法求相似度r_ij
# r_ij=1-c*d_ij
# 欧式距离
def euclidean_distance(dataframe1, decimals=3):
    """
    计算dataframe1中每一行与其他行之间的欧氏距离，并返回一个m*m的DataFrame。

    Args:
        dataframe1 (pd.DataFrame): 输入的m*n的DataFrame。
        decimals (int): 四舍五入到小数点后的位数，默认为3。

    Returns:
        pd.DataFrame: 一个m*m的DataFrame，其中Xij表示dataframe1的第i行和第j行之间的欧氏距离。
    """
    # 初始化距离矩阵
    m = len(dataframe1)
    euclidean_matrix = np.zeros((m, m))

    # 计算欧氏距离
    for i in range(m):
        for j in range(m):
            euclidean_matrix[i][j] = np.linalg.norm(dataframe1.iloc[i] - dataframe1.iloc[j])

    # 四舍五入距离矩阵中的值
    euclidean_matrix_rounded = np.round(euclidean_matrix, decimals=decimals)

    # 创建一个m*m的DataFrame来存储结果
    dataframe2 = pd.DataFrame(euclidean_matrix_rounded, index=dataframe1.index, columns=dataframe1.index)

    return dataframe2


# Hamming距离
def hamming_distance(dataframe1, decimals=3):
    """
    计算dataframe1中每一行与其他行之间的自定义距离，并返回一个m*m的DataFrame。

    参数:
        dataframe1 (pd.DataFrame): 输入的m*n的DataFrame。
        decimals (int): 四舍五入到小数点后的位数，默认为3。

    返回:
        pd.DataFrame: 一个m*m的DataFrame，其中Xij表示dataframe1的第i行和第j行之间的自定义距离。
    """
    # 初始化距离矩阵
    m = len(dataframe1)
    hamming_matrix = np.zeros((m, m))

    # 计算自定义距离
    for i in range(m):
        for j in range(m):
            hamming_matrix[i][j] = np.sum(np.abs(dataframe1.iloc[i] - dataframe1.iloc[j]))

    # 四舍五入距离矩阵中的值
    hamming_matrix_rounded = np.round(hamming_matrix, decimals=decimals)

    # 创建一个m*m的DataFrame来存储结果
    dataframe2 = pd.DataFrame(hamming_matrix_rounded, index=dataframe1.index, columns=dataframe1.index)

    return dataframe2


# Chebyshev距离
def Chebyshev_distance(dataframe1, decimals=3):
    """
    计算dataframe1中每一行与其他行之间的最大距离，并返回一个m*m的DataFrame。

    参数:
        dataframe1 (pd.DataFrame): 输入的m*n的DataFrame。
        decimals (int): 四舍五入到小数点后的位数，默认为3。

    返回:
        pd.DataFrame: 一个m*m的DataFrame，其中Xij表示dataframe1的第i行和第j行之间的最大距离。
    """
    # 初始化距离矩阵
    m = len(dataframe1)
    Chebyshev_matrix = np.zeros((m, m))

    # 计算最大距离
    for i in range(m):
        for j in range(m):
            Chebyshev_matrix[i][j] = np.max(np.abs(dataframe1.iloc[i] - dataframe1.iloc[j]))

    # 四舍五入距离矩阵中的值
    Chebyshev_matrix_rounded = np.round(Chebyshev_matrix, decimals=decimals)

    # 创建一个m*m的DataFrame来存储结果
    dataframe2 = pd.DataFrame(Chebyshev_matrix_rounded, index=dataframe1.index, columns=dataframe1.index)

    return dataframe2


# 贴近度法求相似矩阵
# 最大最小法
def maximum_minimum(dataframe1, decimals=3):
    """
    计算dataframe1中每一行与其他行之间的比率距离，并返回一个m*m的DataFrame。

    参数:
        dataframe1 (pd.DataFrame): 输入的m*n的DataFrame。
        decimals (int): 四舍五入到小数点后的位数，默认为3。

    返回:
        pd.DataFrame: 一个m*m的DataFrame，其中rij表示dataframe1的第i行和第j行之间的比率距离。
    """
    # 初始化距离矩阵
    m = len(dataframe1)
    maxmin_matrix = np.zeros((m, m))

    # 计算比率距离
    for i in range(m):
        for j in range(m):
            numerator = np.sum(np.minimum(dataframe1.iloc[i], dataframe1.iloc[j]))
            denominator = np.sum(np.maximum(dataframe1.iloc[i], dataframe1.iloc[j]))
            maxmin_matrix[i][j] = numerator / denominator if denominator != 0 else 0

    # 四舍五入距离矩阵中的值
    maxmin_matrix_rounded = np.round(maxmin_matrix, decimals=decimals)

    # 创建一个m*m的DataFrame来存储结果
    dataframe2 = pd.DataFrame(maxmin_matrix_rounded, index=dataframe1.index, columns=dataframe1.index)

    return dataframe2


# 算术平均最小法
def arithmetic_mean_minimum(dataframe1, decimals=3):
    """
    计算dataframe1中每一行与其他行之间的修改后的比率距离，并返回一个m*m的DataFrame。

    参数:
        dataframe1 (pd.DataFrame): 输入的m*n的DataFrame。
        decimals (int): 四舍五入到小数点后的位数，默认为3。

    返回:
        pd.DataFrame: 一个m*m的DataFrame，其中rij表示dataframe1的第i行和第j行之间的修改后的比率距离。
    """
    # 初始化距离矩阵
    m = len(dataframe1)
    arithmetic_mean_minimum_matrix = np.zeros((m, m))

    # 计算修改后的比率距离
    for i in range(m):
        for j in range(m):
            numerator = np.sum(np.minimum(dataframe1.iloc[i], dataframe1.iloc[j]))
            denominator = 0.5 * np.sum(dataframe1.iloc[i] + dataframe1.iloc[j])
            arithmetic_mean_minimum_matrix[i][j] = numerator / denominator if denominator != 0 else 0

    # 四舍五入距离矩阵中的值
    arithmetic_mean_minimum_matrix_rounded = np.round(arithmetic_mean_minimum_matrix, decimals=decimals)

    # 创建一个m*m的DataFrame来存储结果
    dataframe2 = pd.DataFrame(arithmetic_mean_minimum_matrix_rounded, index=dataframe1.index, columns=dataframe1.index)

    return dataframe2


# 几何平均最小法
def geometric_mean_minimum(dataframe1, decimals=3):
    """
    计算dataframe1中每一行与其他行之间的几何平均距离，并返回一个m*m的DataFrame。

    参数:
        dataframe1 (pd.DataFrame): 输入的m*n的DataFrame。
        decimals (int): 四舍五入到小数点后的位数，默认为3。

    返回:
        pd.DataFrame: 一个m*m的DataFrame，其中rij表示dataframe1的第i行和第j行之间的几何平均距离。
    """
    # 初始化距离矩阵
    m = len(dataframe1)
    geometric_mean_matrix = np.zeros((m, m))

    # 计算几何平均距离
    for i in range(m):
        for j in range(m):
            numerator = np.sum(np.minimum(dataframe1.iloc[i], dataframe1.iloc[j]))
            denominator = np.sum(np.sqrt(dataframe1.iloc[i] * dataframe1.iloc[j]))
            geometric_mean_matrix[i][j] = numerator / denominator if denominator != 0 else 0

    # 四舍五入距离矩阵中的值
    geometric_mean_matrix_rounded = np.round(geometric_mean_matrix, decimals=decimals)

    # 创建一个m*m的DataFrame来存储结果
    dataframe2 = pd.DataFrame(geometric_mean_matrix_rounded, index=dataframe1.index, columns=dataframe1.index)

    return dataframe2


# 第三部分：模糊矩阵的聚类

# 矩阵的合成
def fuzzy_matrix_composition(df1, df2):
    """
        矩阵的合成：计算两个DataFrame的模糊矩阵合成。
        参数:
            df1 (DataFrame): 第一个pandas DataFrame。
            df2 (DataFrame): 第二个pandas DataFrame。
        返回:
            DataFrame: 合成后的模糊矩阵。
        """
    # 确保第一个DataFrame的列数等于第二个DataFrame的行数
    if df1.shape[1] != df2.shape[0]:
        raise ValueError("第一个DataFrame的列数必须等于第二个DataFrame的行数。")

    # 初始化结果DataFrame
    result_df = pd.DataFrame(index=df1.index, columns=df2.columns)

    # 遍历DataFrame进行模糊矩阵合成
    for i in df1.index:
        for j in df2.columns:
            # 计算模糊合成的值
            result_df.at[i, j] = max(min(df1.at[i, k], df2.at[k, j]) for k in df1.columns)

    return result_df


# 截矩阵的生成
def threshold_matrix(df, threshold):
    """
    截矩阵的生成：根据给定的阈值生成截矩阵。
    参数:
        df (DataFrame): 需要生成截矩阵的pandas DataFrame。
        threshold (float): 用于生成截矩阵的阈值。
    返回:
        DataFrame: 生成的截矩阵。
    """
    return df.where(df >= threshold, 0).where(df < threshold, 1)


def is_fuzzy_matrix_symmetric(matrix_df, tolerance=1e-8):
    """
    判断模糊矩阵是否具有对称性。

    Args:
        matrix_df (pd.DataFrame): 输入的模糊矩阵，应为方阵。
        tolerance (float): 用于比较浮点数时的容差值，默认为1e-8。

    Returns:
        bool: 如果模糊矩阵具有对称性，则返回True；否则返回False。
    """
    # 确保输入是方阵
    if matrix_df.shape[0] != matrix_df.shape[1]:
        raise ValueError("输入的矩阵应为方阵。")

    # 检查是否对称
    for i in range(matrix_df.shape[0]):
        for j in range(i, matrix_df.shape[1]):  # 只需检查矩阵的上三角部分
            if not np.isclose(matrix_df.iloc[i, j], matrix_df.iloc[j, i], atol=tolerance):
                return False
    return True


# 模糊矩阵自反性判断
def is_fuzzy_matrix_reflexive(df, tolerance=1e-8):
    """
    判断模糊矩阵是否自反。
    参数:
        df (DataFrame): 需要判断的pandas DataFrame。
        tolerance (float): 用于比较浮点数时的容差值，默认为1e-8。
    返回:
        bool: 如果矩阵自反则返回True，否则返回False。
    """
    # 检查主对角线上的元素是否都近似等于1
    for i in range(min(df.shape[0], df.shape[1])):
        if not np.isclose(df.iloc[i, i], 1, atol=tolerance):
            return False
    return True


# 模糊数学传递性判断

def is_fuzzy_matrix_Transmissive(df):
    """
        判断模糊矩阵是否传递。
        参数:
            df (DataFrame): 需要判断的pandas DataFrame。
        返回:
            bool: 如果矩阵传递则返回1，否则返回0。
    """
    # 使用fuzzy_matrix_composition函数合成矩阵A和它自身
    df_composed = fuzzy_matrix_composition(df, df)
    # 比较合成后的矩阵A^2和原矩阵A
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            if df_composed.iloc[i, j] > df.iloc[i, j]:
                return 0
    return 1


# 判读矩阵是否为等价矩阵
def is_fuzzy_matrix_equivalent(df, tolerance=1e-8):
    """
        判断矩阵是否为等价矩阵。
        参数:
            df (DataFrame): 需要判断的pandas DataFrame。
            tolerance (float): 用于比较浮点数时的容差值，默认为1e-8。

        返回:
            bool: 如果矩阵是等价矩阵则返回1，否则返回0。
        """
    if is_fuzzy_matrix_Transmissive(df) and is_fuzzy_matrix_reflexive(df, tolerance) and is_fuzzy_matrix_symmetric(df,
                                                                                                                   tolerance):
        return 1
    else:
        return 0


# 判断矩阵是否为相似矩阵
def is_fuzzy_matrix_similar(df, tolerance=1e-8):
    """
        判断矩阵是否为相似矩阵。
        参数:
            df (DataFrame): 需要判断的pandas DataFrame。
            tolerance (float): 用于比较浮点数时的容差值，默认为1e-8。
        返回:
            bool: 如果矩阵是相似矩阵则返回1，否则返回0。
        """
    if is_fuzzy_matrix_reflexive(df, tolerance) and is_fuzzy_matrix_symmetric(df, tolerance):
        return 1
    else:
        return 0


# 平方法求相似矩阵传递闭包

def fuzzy_matrix_transitive_closure(df, tolerance=1e-8):
    """
        判断矩阵是否为相似矩阵。
        参数:
            df (DataFrame): 需要判断的pandas DataFrame。
            tolerance (float): 用于比较浮点数时的容差值，默认为1e-8。
        返回:
            df (DataFrame): 传递闭包后的等价矩阵
        """
    # 初始化传递闭包矩阵为输入矩阵
    transitive_closure = df.copy()
    # 判读是否为相似矩阵
    if not is_fuzzy_matrix_similar(transitive_closure, tolerance):
        raise ValueError("非相似矩阵或者更改容差tolerance")
    for _ in range(1000000):
        # 计算当前传递闭包矩阵与自身的模糊矩阵合成
        transitive_closure = fuzzy_matrix_composition(transitive_closure, transitive_closure)
        if is_fuzzy_matrix_equivalent(transitive_closure, tolerance):
            return transitive_closure
    raise ValueError("合成次数过多或者更改容差tolerance")


# 模糊矩阵的linkage，用于dendrogram()聚类图的绘制

def fuzzy_linkage(df):
    """
    输入带聚类的等价矩阵
    返回同scipy库linkage()的距离矩阵
    格式一样的
    """
    if not is_fuzzy_matrix_equivalent(df):
        raise ValueError("非等价矩阵")
    # 得到从大到小排序的λ
    values_array = pd.unique(df.values.flatten())
    values_array = np.sort(values_array)[::-1]

    # 得到不同的截矩阵并且生成linkage
    link_matrix = []
    togather = []
    for i in values_array:
        temp_df = threshold_matrix(df, i)
        temp_df = pd.DataFrame(temp_df.values.tolist())
        class_count = df.shape[0] - 1
        for c1, c2 in togather:
            colum = temp_df[c1]
            temp_df.drop([c1, c2], axis=1, inplace=True)
            class_count = class_count + 1
            temp_df[class_count] = colum
        class_count = df.shape[0] - 1
        # 遍历DataFrame的每一行
        for index, row in temp_df.iterrows():
            # 遍历行的每个元素
            num = 2
            while 1:
                link_row = []  # 定义一个空列表
                for col, value in temp_df.iloc[index].items():
                    if value == 1:
                        link_row.append(col)  # 如果元素为1，添加列的位置到link_row
                    if len(link_row) == 2:
                        m = [link_row[0], link_row[1], round(1 - i, 3), num]  # 如果link_row有两个元素，添加i和num
                        togather.append(link_row[:2])
                        num = num + 1
                        link_matrix.append(m)  # 将link_row添加到link_matrix
                        l1, l2 = m[:2]
                        colum = temp_df[l1]
                        class_count = class_count + 1
                        temp_df.drop([l1, l2], axis=1, inplace=True)
                        temp_df[class_count] = colum
                        break
                else:
                    break
    return np.array(link_matrix)


# 绘制加强型动态聚类图
def augmented_dendrogram(*args, **kwargs):
    ddata = dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        for i, d in zip(ddata['icoord'], ddata['dcoord']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            plt.plot(x, y, 'ro')
            plt.annotate("%.3g" % (1 - y), (x, y), xytext=(0, -8),
                         textcoords='offset points',
                         va='top', ha='center')

    return ddata


'''df = pd.DataFrame([[1, 0.8, 0.3, 0.4], [0.8, 1, 0.4, 0.5], [0.3, 0.4, 1, 0.9], [0.4, 0.5, 0.9, 1]])

# 计算传递闭包
transitive_closure = fuzzy_matrix_transitive_closure(df)
Z = fuzzy_linkage(transitive_closure)
print(Z)
augmented_dendrogram(Z)
plt.show()'''


# 根据模糊等价矩阵生成动态聚类图


def draw(df, width=6.4, height=4.8):
    """
    绘制基于模糊链接矩阵的动态聚类树状图。

    参数:
    df (DataFrame): 用于聚类的数据框。
    width (float): 图形的宽度，默认为6.4英寸。
    height (float): 图形的高度，默认为4.8英寸。

    返回:
    None: 该函数不返回任何值，但会显示一个树状图。
    注意:
    - 如果想要个性化定制图形，可以直接使用augmented_dendrogram()和matplotlib库的其他功能。
    - 别忘了在函数调用后使用plt.show()来显示图形。
    """
    Z = fuzzy_linkage(df)
    plt.figure(figsize=(width, height))
    augmented_dendrogram(Z, labels=df.columns)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.gca().get_yaxis().set_visible(False)
    plt.text(-0.05, 0.5, 'λ', transform=plt.gca().transAxes, ha='left', va='center', fontsize=14)
    plt.title('模糊矩阵动态聚类图', fontsize=16)


'''df = pd.DataFrame([[1, 0.8, 0.3, 0.4], [0.8, 1, 0.4, 0.5], [0.3, 0.4, 1, 0.9], [0.4, 0.5, 0.9, 1]])
df = fuzzy_matrix_transitive_closure(df)
print(df)
draw(df)
plt.show()
'''


# 聚类和阈值寻找函数
def num_clusters(df, num):
    """
    寻找合适的阈值并进行聚类。
    参数:
        df (DataFrame): 需要聚类的pandas DataFrame。
        num (int): 期望的聚类数量。
    返回:
        tuple: (找到的阈值, 聚类结果的列表)。
    """
    # 获取矩阵中的所有唯一值并降序排序
    unique_values = np.sort(pd.unique(df.values.flatten()))[::-1]

    # 遍历所有可能的阈值
    for threshold in unique_values:
        # 生成截矩阵
        binary_matrix = threshold_matrix(df, threshold)
        # 初始化聚类列表
        cluster_list = []
        # 遍历截矩阵的每一行
        for i, row in binary_matrix.iterrows():
            # 找到数值为1的元素对应的列名
            connected_points = list(row[row == 1].index)
            # 如果当前行的元素还没有被分配到任何聚类中
            if not any(i in sublist for sublist in cluster_list):
                # 创建新的聚类
                new_cluster = [i] + connected_points
                # 添加到聚类列表中
                cluster_list.append(new_cluster)

        # 清理聚类列表，合并重叠的聚类
        cleaned_cluster_list = []
        while len(cluster_list) > 0:
            first, *rest = cluster_list
            first = set(first)
            lf = -1
            while len(first) > lf:
                lf = len(first)
                rest2 = []
                for r in rest:
                    if len(first.intersection(set(r))) > 0:
                        first |= set(r)
                    else:
                        rest2.append(r)
                rest = rest2
            cleaned_cluster_list.append(list(first))
            cluster_list = rest

        # 检查聚类数量是否符合要求
        if len(cleaned_cluster_list) == num:
            # 返回找到的阈值和聚类结果
            return threshold, cleaned_cluster_list

    # 如果没有找到符合条件的阈值和聚类
    raise ValueError("没有找到符合条件的阈值和聚类")


# 模糊统计量与最佳阈值的判读
# 模糊统计量F的求解
def fuzzy_statistic(df0, df1, lambda_level):
    """
    df0是原始元素-属性矩阵
    df1是归一化、相似后的矩阵
    lambda_level表示对应的截矩阵选择的界限
    返回的是对应λ的F值
    """
    # 计算传递闭包
    df1 = fuzzy_matrix_transitive_closure(df1)
    if not is_fuzzy_matrix_equivalent(df1):
        raise ValueError("非等价矩阵")
    # 计算给定 lambda 水平的阈值矩阵
    threshold_df = threshold_matrix(df1, lambda_level)

    # 计算唯一行的数量，表示不同的类别
    unique_rows = len(threshold_df.drop_duplicates(keep='first'))

    # 查找类别的数量以及每个类别中的对象数量
    r = unique_rows  # 类别的数量
    n = df0.shape[0]  # 总对象数
    n_i = threshold_df.sum(axis=0)  # 每个类别中的对象数量

    # 如果类别数为1或总对象数等于类别数，则F值为0
    if r == 1 or n == r:
        return 0

    # 计算全局中心
    global_center = df0.mean(axis=0)  # 所有对象的均值

    # 初始化分子和分母
    numerator = 0
    denominator = 0

    # 计算模糊统计量的分子和分母
    for i in range(r):
        # 获取属于类别 i 的对象
        class_objects = df0.iloc[threshold_df.iloc[:, i].astype(bool).values]
        # 计算类别 i 的中心
        class_center = class_objects.mean(axis=0)
        # 在分子中累积
        numerator += n_i.iloc[i] * np.linalg.norm(class_center - global_center) ** 2

        # 对类别 i 中的每个对象进行累积
        for j in class_objects.index:
            # 在分母中累积
            denominator += np.linalg.norm(class_objects.loc[j] - class_center) ** 2

    # 调整自由度
    numerator /= (r - 1) if r > 1 else 1  # 避免除以0
    denominator /= (n - r) if n > r else 1  # 避免除以0

    # 计算模糊统计量 F
    F = numerator / denominator

    return F


# 最优的分类λ值
def best_F(df0, df1):
    """
    df0是原始元素-属性矩阵
    df1是归一化、相似后的矩阵
    返回的一个列表
    第一个是最优F时的λ值
    第二个是最优的F值
    """
    # 计算传递闭包
    df1 = fuzzy_matrix_transitive_closure(df1)
    # 得到从大到小排序的λ
    values_array = pd.unique(df1.values.flatten())
    values_array = np.sort(values_array)[::-1]
    F_value = {}
    for lamuda in values_array:
        F_value[lamuda] = fuzzy_statistic(df0, df1, lamuda)
    return [max(F_value, key=F_value.get), max(F_value.values())]


# 第四部分：模糊识别


# 模糊集内积
def fuzzy_inner_product(A, B):
    """
        A和B为两个pd.Series表示的模糊集
        返回两个的内积
    """
    if not isinstance(A, pd.Series) or not isinstance(B, pd.Series):
        raise ValueError("必须是pd.Series对象")
    n = len(A)
    if n != len(B):
        raise ValueError("必须具有相同的长度")
    # 计算模糊集的内积
    return A.combine(B, min).max()


# 模糊集外积
def fuzzy_outer_product(A, B):
    """
        A和B为两个pd.Series表示的模糊集
        返回两个的外积
    """
    if not isinstance(A, pd.Series) or not isinstance(B, pd.Series):
        raise ValueError("必须是pd.Series对象")
    n = len(A)
    if n != len(B):
        raise ValueError("必须具有相同的长度")
    # 计算模糊集的外积
    return A.combine(B, max).min()


# 计算两个模糊集的格贴近度
def lattice_proximity(A, B):
    """
        A和B为两个pd.Series表示的模糊集
        返回的时格贴近度
    """
    if not isinstance(A, pd.Series) or not isinstance(B, pd.Series):
        raise ValueError("必须是pd.Series对象")
    n = len(A)
    if n != len(B):
        raise ValueError("必须具有相同的长度")
    # 计算格贴近度
    inner_prod = fuzzy_inner_product(A, B)
    outer_prod = fuzzy_outer_product(A, B)
    similarity = 0.5 * (inner_prod + 1 - outer_prod)
    return similarity


# 最小最大贴近度
def min_max_proximity(A, B):
    """
        A和B为两个pd.Series表示的模糊集
        返回最小最大贴近度
    """
    # 确保A和B是pd.Series对象
    if not isinstance(A, pd.Series) or not isinstance(B, pd.Series):
        raise ValueError("必须是pd.Series对象")
    n = len(A)
    if n != len(B):
        raise ValueError("必须具有相同的长度")
    # 计算分子：模糊集的 "取小" 操作
    numerator = A.combine(B, min).sum()

    # 计算分母：模糊集的 "取大" 操作
    denominator = A.combine(B, max).sum()

    # 计算相似度
    if denominator == 0:
        raise ValueError("分母不能为0")
    similarity = numerator / denominator

    return similarity


# 计算最小平均贴近度
def min_mean_proximity(A, B):
    """
        A和B为两个pd.Series表示的模糊集
        返回最小平均贴近度
    """
    # 确保A和B是pd.Series对象
    if not isinstance(A, pd.Series) or not isinstance(B, pd.Series):
        raise ValueError("必须是pd.Series对象")
    n = len(A)
    if n != len(B):
        raise ValueError("必须具有相同的长度")
    # 计算分子：模糊集的 "AND" 操作
    numerator = 2 * A.combine(B, min).sum()

    # 计算分母：简单的算术加法
    denominator = A.sum() + B.sum()

    # 计算相似度
    if denominator == 0:
        raise ValueError("分母不能为0")
    similarity = numerator / denominator

    return similarity


# 海明贴近度
def hamming_proximity(A, B):
    """
        A和B为两个pd.Series表示的模糊集
        返回海明贴近度
    """
    # 确保A和B是pd.Series对象
    if not isinstance(A, pd.Series) or not isinstance(B, pd.Series):
        raise ValueError("必须是pd.Series对象")
    # 计算海明贴近度
    n = len(A)
    if n != len(B):
        raise ValueError("必须具有相同的长度")

    # 计算每个元素差的绝对值，然后求和
    sum_of_differences = sum(abs(A - B))

    # 计算海明贴近度
    proximity = 1 - (sum_of_differences / n)

    return proximity


# 欧几里得贴近度
def euclidean_proximity(A, B):
    """
        A和B为两个pd.Series表示的模糊集
        返回欧几里得贴近度
    """
    # 确保A和B是pd.Series对象
    if not isinstance(A, pd.Series) or not isinstance(B, pd.Series):
        raise ValueError("必须是pd.Series对象")
    # 计算欧几里得贴近度
    n = len(A)
    if n != len(B):
        raise ValueError("必须具有相同的长度")

    # 计算平方差的和
    sum_of_squares = np.sum((A - B) ** 2)

    # 计算欧几里得贴近度
    proximity = 1 - (np.sqrt(sum_of_squares) / np.sqrt(n))

    return proximity


# 第五部分：模糊综合评判

# 主因素决定型
def principal_factor_determination_evaluation(A, R):
    """
        A为pd.Series表示的权重向量
        R为pd.Dataframe表示的综合评价矩阵
        返回主因素决定型评价向量
    """
    # 确保A是一个Series，R是一个DataFrame
    if not isinstance(A, pd.Series) or not isinstance(R, pd.DataFrame):
        raise ValueError("A必须是pd.Series对象，R必须是pd.DataFrame对象")
    # 创建一个空的Series来保存内积结果
    inner_products = pd.Series(dtype=float)

    # 遍历DataFrame R的每一列
    for column in R:
        # 计算A和当前列的内积
        inner_product = fuzzy_inner_product(A, R[column])
        # 将内积结果添加到Series中
        inner_products[column] = inner_product

    # 返回包含所有内积结果的Series
    return inner_products


# 主因素突出型
def principal_factor_prominent_evaluation(A, R):
    """
        A为pd.Series表示的权重向量
        R为pd.Dataframe表示的综合评价矩阵
        返回主因素突出型评价向量
    """
    # 确保A是一个Series，R是一个DataFrame
    if not isinstance(A, pd.Series) or not isinstance(R, pd.DataFrame):
        raise ValueError("A必须是pd.Series对象，R必须是pd.DataFrame对象")

    # 创建一个空的Series来保存每个对象的综合评价值
    B = pd.Series(dtype=float)

    # 遍历R的每一列，即每个对象
    for j in R.columns:
        # 计算每个因素的权重与评价值的乘积
        products = A * R[j]
        # 取乘积的最大值作为对象的综合评价值
        B[j] = products.max()

    # 返回包含所有对象综合评价值的Series
    return B


# 加权平均型
def weighted_sum_evaluation(A, R):
    """
        A为pd.Series表示的权重向量
        R为pd.Dataframe表示的综合评价矩阵
        返回加权平均型评价向量
    """
    # 确保A是一个Series，R是一个DataFrame
    if not isinstance(A, pd.Series) or not isinstance(R, pd.DataFrame):
        raise ValueError("A必须是pd.Series对象，R必须是pd.DataFrame对象")

    # 创建一个空的Series来保存每个对象的综合评价值
    B = pd.Series(dtype=float)

    # 遍历R的每一列，即每个对象
    for j in R.columns:
        # 计算每个因素的权重与评价值的乘积之和
        weighted_sum = (A * R[j]).sum()
        # 将计算结果作为对象的综合评价值
        B[j] = weighted_sum

    # 返回包含所有对象综合评价值的Series
    return B


# 取小上界和型
def min_sum_evaluation(A, R):
    """
        A为pd.Series表示的权重向量
        R为pd.Dataframe表示的综合评价矩阵
        返回取小上界和型评价向量
    """
    # 确保A是一个Series，R是一个DataFrame
    if not isinstance(A, pd.Series) or not isinstance(R, pd.DataFrame):
        raise ValueError("A必须是pd.Series对象，R必须是pd.DataFrame对象")

    # 创建一个空的Series来保存每个选项的综合评价值
    B = pd.Series(dtype=float)

    # 遍历R的每一列，即每个选项
    for j in R.columns:
        # 计算每个因素的权重与评价值的最小值之和
        min_sum = (A.combine(R[j], min)).sum()
        # 取和与1之间的最小值作为选项的综合评价值
        B[j] = min(min_sum, 1)

    # 返回包含所有选项综合评价值的Series
    return B


# 均衡平均型
def balanced_average_evaluation(A, R):
    """
        A为pd.Series表示的权重向量
        R为pd.Dataframe表示的综合评价矩阵
        返回均衡平均型向量
    """
    # 确保A是一个Series，R是一个DataFrame
    if not isinstance(A, pd.Series) or not isinstance(R, pd.DataFrame):
        raise ValueError("A必须是pd.Series对象，R必须是pd.DataFrame对象")

    # 创建一个空的Series来保存每个选项的综合评价值
    B = pd.Series(dtype=float)

    # 计算每个选项的评价值之和
    r_j_sum = R.sum()

    # 遍历R的每一列，即每个选项
    for j in R.columns:
        # 计算每个因素的权重与评价值比值的最小值之和
        normalized_weighted_sum = (A.combine(R[j] / r_j_sum[j], min)).sum()
        # 将计算结果作为选项的综合评价值
        B[j] = normalized_weighted_sum

    # 返回包含所有选项综合评价值的Series
    return B
