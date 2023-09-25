from itertools import product 
 
# 定义要检查的两个数组 
array1 = [(0, 0, 0), (0, 0, 0), (1, 0, 0),(0, 1, 0),(0, 0, 1),(0, 0, 0),(0, 0, 0),(0, 0, 0)] 
 
# 创建一个空字典，用于保存不同种类的组合和对应的索引位置列表 
combinations_dict = {} 
 
# 进行两两组合 
combined = list(product(array1,array1))
print(combined)
 
# 遍历组合并记录不同种类的组合和索引位置
for index, combo in enumerate(combined):
    combination = tuple(sorted(combo))
    if combination not in combinations_dict:
        combinations_dict[combination] = []
    combinations_dict[combination].append(index)
 
# 打印结果
print(combinations_dict)
# 分别打印个数
for k, v in combinations_dict.items(): 
    print(k, len(v))