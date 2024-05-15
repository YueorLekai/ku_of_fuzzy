# ku_of_fuzzy
# 目录
- 序
- 简介
- 安装
- 使用示例
- 功能列表
- 贡献指南
- 许可证
- 联系方式
- 致谢
- 更新日志
## 序
鄙人本科在校生，利用课余时间写的一个简单的基本模糊数学计算的库\
以便数模竞赛，课程项目\
才疏学浅，偶有错误之处，尊请指出\
若有建议及其他想法，欢迎联系1977269004@qq.com
## 简介
这是一个用于模糊数学计算的Python库。\
包含数据多种方式标准化和归一化、相似度矩阵的多种方法建立、\
模糊矩阵矩阵的合成、截矩阵的生成、模糊矩阵自反性等性质的判断、\
及平方法求相似矩阵传递闭包、模糊矩阵的linkage求解和动态聚类图的绘制、\
模糊统计量F的求解和最佳F值的求解、模糊集内积与外积、两模糊集多种方法求解贴近度、\
以及多种模型求解综合评价矩阵。

## 安装
```bash
pip install ku_of_fuzzy
```
## 使用示例

以下是`ku_of_fuzzy`库的一些基本使用示例：

```python
from ku_of_fuzzy import fuzzy_calculate
import pandas as pd
import matplotlib.pyplot as plt
# 示例：内积求解
A = pd.Series([0.1, 0.2, 0.5, 0.4])
B = pd.Series([0.3, 0.1, 0.5, 0.4])
inner_product = fuzzy_calculate.fuzzy_inner_product(A,B)

# 示例：画出最佳聚类图
df = pd.DataFrame([[1, 0.8, 0.3, 0.4], [0.8, 1, 0.4, 0.5], [0.3, 0.4, 1, 0.9], [0.4, 0.5, 0.9, 1]])
# 传递闭包求等价矩阵
df = fuzzy_calculate.fuzzy_matrix_transitive_closure(df)
fuzzy_calculate.draw(df)
plt.show()
```
# 功能列表

## 第一部分：标准化
`ku_of_fuzzy`库提供了多种数据标准化方法，以便在模糊数学计算中使用。以下是可用的标准化函数：

### 标准差标准化
- **函数**: `standardize_columns(df)`
- **描述**: 使用每列的均值和标准差对DataFrame的列进行标准化。
- **参数**:
  - `df` (DataFrame): 需要标准化的pandas DataFrame。
- **返回**: 标准化后的DataFrame。

### 最小-最大标准化(极值标准化)
- **函数**: `min_max_normalize(df)`
- **描述**: 对DataFrame的列进行极值标准化。
- **参数**:
  - `df` (DataFrame): 需要标准化的pandas DataFrame。
- **返回**: 标准化后的DataFrame。

### 极差标准化
- **函数**: `mean_normalization(df)`
- **描述**: 减去均值再除以最大与最小的差，进行极差标准化。
- **参数**:
  - `df` (DataFrame): 需要标准化的pandas DataFrame。
- **返回**: 标准化后的DataFrame。

### 最大值规格化
- **函数**: `max_normalize(df)`
- **描述**: 使用每列的最大值对DataFrame的列进行规格化。
- **参数**:
  - `df` (DataFrame): 需要规格化的pandas DataFrame。
- **返回**: 规格化后的DataFrame。
## 第二部分：建立相似度矩阵
`ku_of_fuzzy`库提供了多种方法来建立相似度矩阵。以下是可用的建立相似度矩阵的函数：

### 余弦相似度法
- **函数**: `cosine_similarity(dataframe1, decimals=3)`
- **描述**: 计算DataFrame中每一行与其他行之间的余弦相似度，并四舍五入到指定的小数位数。
- **参数**:
  - `dataframe1` (pd.DataFrame): 输入的m*n的DataFrame，元素值在[0, 1]之间。
  - `decimals` (int, 可选): 四舍五入到小数点后的位数，默认为5。
- **返回**: 一个m*m的DataFrame，其中Xij表示第i行和第j行之间的余弦相似度。

### 皮尔逊相关系数法
- **函数**: `pearson_similarity(dataframe1, decimals=3)`
- **描述**: 计算DataFrame中每一行与其他行之间的皮尔逊相关系数，并四舍五入到指定的小数位数。
- **参数**:
  - `dataframe1` (pd.DataFrame): 输入的m*n的DataFrame。
  - `decimals` (int, 可选): 四舍五入到小数点后的位数，默认为5。
- **返回**: 一个m*m的DataFrame，其中Xij表示第i行和第j行之间的皮尔逊相关系数。

### 欧氏距离法
- **函数**: `euclidean_distance(dataframe1, decimals=3)`
- **描述**: 计算DataFrame中每一行与其他行之间的欧氏距离，并四舍五入到指定的小数位数。
- **参数**:
  - `dataframe1` (pd.DataFrame): 输入的m*n的DataFrame。
  - `decimals` (int, 可选): 四舍五入到小数点后的位数，默认为5。
- **返回**: 一个m*m的DataFrame，其中Xij表示第i行和第j行之间的欧氏距离。

### Hamming距离法
- **函数**: `hamming_distance(dataframe1, decimals=3)`
- **描述**: 计算DataFrame中每一行与其他行之间的Hamming距离，并四舍五入到指定的小数位数。
- **参数**:
  - `dataframe1` (pd.DataFrame): 输入的m*n的DataFrame。
  - `decimals` (int, 可选): 四舍五入到小数点后的位数，默认为5。
- **返回**: 一个m*m的DataFrame，其中Xij表示第i行和第j行之间的Hamming距离。

### Chebyshev距离法
- **函数**: `Chebyshev_distance(dataframe1, decimals=3)`
- **描述**: 计算DataFrame中每一行与其他行之间的Chebyshev距离，并四舍五入到指定的小数位数。
- **参数**:
  - `dataframe1` (pd.DataFrame): 输入的m*n的DataFrame。
  - `decimals` (int, 可选): 四舍五入到小数点后的位数，默认为5。
- **返回**: 一个m*m的DataFrame，其中Xij表示第i行和第j行之间的Chebyshev距离。

### 最大最小法
- **函数**: `maximum_minimum(dataframe1, decimals=3)`
- **描述**: 计算DataFrame中每一行与其他行之间的最大最小比率距离，并四舍五入到指定的小数位数。
- **参数**:
  - `dataframe1` (pd.DataFrame): 输入的m*n的DataFrame。
  - `decimals` (int, 可选): 四舍五入到小数点后的位数，默认为5。
- **返回**: 一个m*m的DataFrame，其中rij表示第i行和第j行之间的最大最小比率距离。

### 算术平均最小法
- **函数**: `arithmetic_mean_minimum(dataframe1, decimals=3)`
- **描述**: 计算DataFrame中每一行与其他行之间的算术平均最小比率距离，并四舍五入到指定的小数位数。
- **参数**:
  - `dataframe1` (pd.DataFrame): 输入的m*n的DataFrame。
  - `decimals` (int, 可选): 四舍五入到小数点后的位数，默认为5。
- **返回**: 一个m*m的DataFrame，其中rij表示第i行和第j行之间的算术平均最小比率距离。

### 几何平均最小法
- **函数**: `geometric_mean_minimum(dataframe1, decimals=3)`
- **描述**: 计算DataFrame中每一行与其他行之间的几何平均最小比率距离，并四舍五入到指定的小数位数。
- **参数**:
  - `dataframe1` (pd.DataFrame): 输入的m*n的DataFrame。
  - `decimals` (int, 可选): 四舍五入到小数点后的位数，默认为5。
- **返回**: 一个m*m的DataFrame，其中rij表示第i行和第j行之间的几何平均最小比率距离。


## 第三部分：模糊矩阵的聚类
`ku_of_fuzzy`库提供了一系列函数来处理模糊矩阵的聚类问题。以下是可用的聚类相关函数：

### 矩阵的合成
- **函数**: `fuzzy_matrix_composition(df1, df2)`
- **描述**: 计算两个DataFrame的模糊矩阵合成。
- **参数**:
  - `df1` (DataFrame): 第一个pandas DataFrame。
  - `df2` (DataFrame): 第二个pandas DataFrame。
- **返回**: 合成后的模糊矩阵。

### 截矩阵的生成
- **函数**: `threshold_matrix(df, threshold)`
- **描述**: 根据给定的阈值生成截矩阵。
- **参数**:
  - `df` (DataFrame): 需要生成截矩阵的pandas DataFrame。
  - `threshold` (float): 用于生成截矩阵的阈值。
- **返回**: 生成的截矩阵。

### 模糊矩阵对称性判断
- **函数**: `is_fuzzy_matrix_symmetric(matrix_df)`
- **描述**: 判断模糊矩阵是否具有对称性。
- **参数**:
  - `matrix_df` (pd.DataFrame): 输入的模糊矩阵，应为方阵。
  - `tolerance`(float): 判断接近的容差，默认为1e-8
- **返回**: 如果模糊矩阵具有对称性，则返回True；否则返回False。

### 模糊矩阵自反性判断
- **函数**: `is_fuzzy_matrix_reflexive(df)`
- **描述**: 判断模糊矩阵是否自反。
- **参数**:
  - `df` (DataFrame): 需要判断的pandas DataFrame。
  - `tolerance`(float): 判断接近的容差，默认为1e-8
- **返回**: 如果矩阵自反则返回True，否则返回False。

### 模糊矩阵传递性判断
- **函数**: `is_fuzzy_matrix_Transmissive(df)`
- **描述**: 判断模糊矩阵是否传递。
- **参数**:
  - `df` (DataFrame): 需要判断的pandas DataFrame。
- **返回**: 如果矩阵传递则返回True，否则返回False。

### 模糊矩阵等价性判断
- **函数**: `is_fuzzy_matrix_equivalent(df)`
- **描述**: 判断矩阵是否为等价矩阵。
- **参数**:
  - `df` (DataFrame): 需要判断的pandas DataFrame。
  - `tolerance`(float): 判断接近的容差，默认为1e-8
- **返回**: 如果矩阵是等价矩阵则返回True，否则返回False。

### 模糊矩阵相似性判断
- **函数**: `is_fuzzy_matrix_similar(df)`
- **描述**: 判断矩阵是否为相似矩阵。
- **参数**:
  - `df` (DataFrame): 需要判断的pandas DataFrame。
  - `tolerance`(float): 判断接近的容差，默认为1e-8。
- **返回**: 如果矩阵是相似矩阵则返回True，否则返回False。

### 相似矩阵传递闭包求解
- **参数**:
  - `df` (DataFrame): 需要求解的相似矩阵pandas DataFrame。
  - `tolerance`(float): 判断接近的容差，默认为1e-8。
- **函数**: `fuzzy_matrix_transitive_closure(df)`
- **描述**: 输入相似矩阵，返回合成后的等价矩阵。

### 模糊矩阵的linkage求解
- **函数**: `fuzzy_linkage(df)`
- **描述**: 输入带聚类的等价矩阵，返回同scipy库linkage()的距离矩阵格式一样的结果。

### 动态聚类图绘制
- **函数**: `augmented_dendrogram(*args, **kwargs)`
- **描述**: 绘制加强型动态聚类图，额外标出了λ的值。

### 动态聚类图绘制
- **函数**: `draw(df)`
- **描述**: 提供一个常用的聚类图绘制方法，额外标出了λ的值。如果需要个性化定制，可以使用`augmented_dendrogram()`和`matplotlib`库。
- **参数**:
  - `df` (DataFrame): 输入的模糊等价矩阵。
  - `width` (float): 图形的宽度，以英寸为单位，默认为6.4。
  - `height` (float): 图形的高度，以英寸为单位，默认为4.8。
- **注意**: 在绘制完聚类图后，使用`plt.show()`来显示图像。

### 模糊统计量F的求解
- **函数**: `fuzzy_statistic(df0, df1, lambda_level)`
- **描述**: 输入原始元素-属性矩阵和归一化、相似后的矩阵，返回对应λ的F值。
### 聚类数量确定阈值

- **函数**: `num_clusters(df, num)`
- **描述**: 根据给定的相似度矩阵和期望的聚类数量，找到一个阈值，使得在该阈值下，矩阵中相似度高于阈值的元素被划分为同一类，从而达到指定的聚类数量。
- **参数**:
  - `df` (DataFrame): 需要聚类的pandas DataFrame。
  - `num` (int): 期望的聚类数量。
- **返回**: 
  - `threshold` (float): 确定的聚类阈值。
  - `cluster_list` (list of lists): 每个子列表包含属于同一类的元素名。
### 阈值确定聚类数量

- **函数**: `classify_by_threshold(df, lamda)`
- **描述**: 给定一个阈值和相似度矩阵，返回在该阈值下的聚类数量和具体的分类情况。
- **参数**:
  - `df` (DataFrame): 需要聚类的pandas DataFrame。
  - `lamda` (float): 给定的聚类阈值。
- **返回**: 
  - `num_classes` (int): 在阈值 `lamda` 下的聚类数量。
  - `classification` (list of lists): 每个子列表包含属于同一类的元素名。
### 最优分类λ值求解
- **函数**: `best_F(df0, df1)`
- **描述**: 输入原始元素-属性矩阵和归一化、相似后的矩阵，返回最优F时的λ值和最优的F值。
## 第四部分：模糊识别
`ku_of_fuzzy`库提供了一系列函数来处理模糊集的识别问题。以下是可用的模糊识别相关函数：

### 模糊集内积
- **函数**: `fuzzy_inner_product(A, B)`
- **描述**: 计算两个模糊集的内积。
- **参数**:
  - `A`, `B` (pd.Series): 两个表示模糊集的pandas Series。

### 模糊集外积
- **函数**: `fuzzy_outer_product(A, B)`
- **描述**: 计算两个模糊集的外积。
- **参数**:
  - `A`, `B` (pd.Series): 两个表示模糊集的pandas Series。

### 格贴近度
- **函数**: `lattice_proximity(A, B)`
- **描述**: 计算两个模糊集的格贴近度。
- **参数**:
  - `A`, `B` (pd.Series): 两个表示模糊集的pandas Series。

### 最小最大贴近度
- **函数**: `min_max_proximity(A, B)`
- **描述**: 计算两个模糊集的最小最大贴近度。
- **参数**:
  - `A`, `B` (pd.Series): 两个表示模糊集的pandas Series。

### 最小平均贴近度
- **函数**: `min_mean_proximity(A, B)`
- **描述**: 计算两个模糊集的最小平均贴近度。
- **参数**:
  - `A`, `B` (pd.Series): 两个表示模糊集的pandas Series。

### 海明贴近度
- **函数**: `hamming_proximity(A, B)`
- **描述**: 计算两个模糊集的海明贴近度。
- **参数**:
  - `A`, `B` (pd.Series): 两个表示模糊集的pandas Series。

### 欧几里得贴近度
- **函数**: `euclidean_proximity(A, B)`
- **描述**: 计算两个模糊集的欧几里得贴近度。
- **参数**:
  - `A`, `B` (pd.Series): 两个表示模糊集的pandas Series。

## 第五部分：模糊综合评判
`ku_of_fuzzy`库提供了一系列函数来进行模糊综合评判。以下是可用的模糊综合评判相关函数：

### 主因素决定型评价
- **函数**: `principal_factor_determination_evaluation(A, R)`
- **描述**: 返回主因素决定型评价向量。
- **参数**:
  - `A` (pd.Series): 权重向量。
  - `R` (pd.DataFrame): 综合评价矩阵。

### 主因素突出型评价
- **函数**: `principal_factor_prominent_evaluation(A, R)`
- **描述**: 返回主因素突出型评价向量。
- **参数**:
  - `A` (pd.Series): 权重向量。
  - `R` (pd.DataFrame): 综合评价矩阵。

### 加权平均型评价
- **函数**: `weighted_sum_evaluation(A, R)`
- **描述**: 返回加权平均型评价向量。
- **参数**:
  - `A` (pd.Series): 权重向量。
  - `R` (pd.DataFrame): 综合评价矩阵。

### 取小上界和型评价
- **函数**: `min_sum_evaluation(A, R)`
- **描述**: 返回取小上界和型评价向量。
- **参数**:
  - `A` (pd.Series): 权重向量。
  - `R` (pd.DataFrame): 综合评价矩阵。

### 均衡平均型评价
- **函数**: `balanced_average_evaluation(A, R)`
- **描述**: 返回均衡平均型评价向量。
- **参数**:
  - `A` (pd.Series): 权重向量。
  - `R` (pd.DataFrame): 综合评价矩阵。
## 贡献指南

我们欢迎并感谢任何形式的贡献。如果您想为项目贡献代码、文档或其他形式的内容，请遵循以下步骤：

### 报告问题
- 使用GitHub Issues报告bug或请求新功能。
- 在创建新问题之前，请先检查是否已有相关的问题。

### 提交更改
- Fork项目仓库。
- 在您自己的分支上进行更改。
- 编写清晰的提交信息。
- 推送您的更改并创建一个Pull Request。
- 在Pull Request中描述您的更改和动机。

### 开发指南
- 确保您的代码遵循PEP 8编码规范。
- 添加单元测试以验证您的代码的功能。
- 确保所有测试都能通过。

### 文档
- 如果您添加了新功能，请更新README.md或相应的文档。
- 提供示例代码以帮助用户理解如何使用新功能。

### 许可
- 提交您的贡献意味着您同意根据本项目的许可证分发您的代码。

感谢您考虑为此项目做出贡献！
## 许可证

本项目采用MIT许可证。有关许可证的完整文本，请参见项目中的LICENSE文件。
## 联系方式

如果您在使用本库时遇到任何问题，或者有任何反馈和建议，请通过以下方式联系我：

- 邮箱：1977269004@qq.com
- GitHub Issues：https://github.com/YueorLekai/ku_of_fuzzy/issues
- QQ：1977269004(注明来意)
## 致谢

感谢所有为本项目做出贡献的人员，无论是通过提交代码、提供反馈还是其他方式的支持。

## 更新日志

### 0.1.0 - 2024-05-13
- 初始版本发布，提供基础的模糊数学计算功能。

更多更新信息，请查看项目的CHANGELOG.md文件。
