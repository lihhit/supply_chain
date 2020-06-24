# coding=utf-8

"""
背景：

假定F公司需要采购某一种产品，总的需求量D是200-250,
目前有5个供应商可供选择，忽略这5个供应商的优劣等级，
5个供应商的供货能力如下表所示：

供应商i	运输费用o_i	产品单价p_i	废品率q_i(%)	最大供货量d_i	交货期
1	15	5	5	50	10
2	10	5	5	90	5
3	6	4	10	70	8
4	5	10	10	80	3
5	4	2	10	70	8



假设从5家供应商购买产品的数量分别为：x_1,x_2,x_3,x_4,x_5,可以得到3个目标函数：
    成本目标函数：Z_1=15y_1+10y_2+6y_3+5y_4+4y_5+ 5x_1+5x_2+4x_3+10x_4+10x_5
    废品率目标函数： Z_2= 0.05x_1+0.05x_2+0.1x_3+0.1x_4+0.1x_5
    交货期目标函数：Z_3= 10x_1+5x_2+8x_3+3x_4+8x_5
约束条件：
    200≤x_1+x_2+x_3+x_4+x_5≤250
    x_1≤50,x_2≤90,x_3≤70,x_4≤80,x_5≤70

NSGA2算法实现：
1、随机产生初始种群，做二进制编码
2、通过选择、交叉、变异，扩张种群
3、基于帕累托前沿理论，做非支配排序
4、拥挤距离计算
5、得到最优解集
"""

import random as rn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D as p3d


def individual_from_values_to_gene(values):
    gene = []
    for i in range(len(values)):
        gene = gene + [int(x) for x in '{:08b}'.format(values[i])]
    return gene

def individual_from_gene_to_value(gene):
    values=[]
    values.append(int("".join(str(i) for i in gene[:8]), 2))
    values.append(int("".join(str(i) for i in gene[8:16]), 2))
    values.append(int("".join(str(i) for i in gene[16:24]), 2))
    values.append(int("".join(str(i) for i in gene[24:32]), 2))
    values.append(int("".join(str(i) for i in gene[32:40]), 2))
    return values


# 产生初始种群
def creat_initial_population(population_size):
    population = []
    while len(population)<population_size:
        x1 = np.random.random_integers(0,50)
        x2 = np.random.random_integers(0,90)
        x3 = np.random.random_integers(0,70)
        x4 = np.random.random_integers(0, 80)
        x5 = np.random.random_integers(0, 70)

        temp = x1+x2+x3+x4+x5
        if temp>=200 and temp<=250:
            population.append(([x1,x2,x3,x4,x5]))

    return np.array(population)

# 计算目标函数的值
# Z_1=15y_1+10y_2+6y_3+5y_4+4y_5+ 5x_1+5x_2+4x_3+10x_4+10x_5
# Z_2= 0.05x_1+0.05x_2+0.1x_3+0.1x_4+0.1x_5
# Z_3= 10x_1+5x_2+8x_3+3x_4+8x_5
def calculate_objects_value(population):
    objects = []
    size=population.shape[0]
    for i in range(size):
        x = population[i]
        y = []
        for j in range(len(x)):
            if x[j]>0:y.append(1)
            else:y.append(0)
        Z1 = 15*y[0]+10*y[1]+6*y[2]+5*y[3]+4*y[4]+5*x[0]+5*x[1]+4*x[2]+10*x[3]+10*x[4]
        Z2 = 0.05*x[0]+0.05*x[1]+0.1*x[2]+0.1*x[3]+0.1*x[4]
        Z3 = 10*x[0]+5*x[1]+8*x[2]+3*x[3]+8*x[4]
        objects.append([Z1,Z2,Z3])
    return np.array(objects)

# 基于目标函数的值，计算拥挤距离
def calculate_crowding_distance(objects):
    
    population_size = len(objects[:, 0])
    number_of_objects = len(objects[0, :])
    crowding_matrix = np.zeros((population_size, number_of_objects))
    # 对目标函数的值，做归一化处理 (ptp is max-min)
    normed_objects = (objects - objects.min(0)) / objects.ptp(0)
    # 
    for col in range(number_of_objects):
        crowding = np.zeros(population_size)
        # 设定前后端的点，拥挤距离最大
        crowding[0] = 1
        crowding[population_size - 1] = 1
        # 对目标函数值从小到大排序，为了计算相邻两个个体的拥挤距离 
        sorted_objects = np.sort(normed_objects[:, col])

        sorted_objects_index = np.argsort(
            normed_objects[:, col])

        # 对每个个体左右相减得到该个体的拥挤距离
        crowding[1:population_size - 1] = \
            (sorted_objects[2:population_size] -
             sorted_objects[0:population_size - 2])

        # 重新得到原来个体的index
        re_sort_order = np.argsort(sorted_objects_index)
        sorted_crowding = crowding[re_sort_order]
        crowding_matrix[:, col] = sorted_crowding

    # 对两个目标函数得到的拥挤距离做相加，得到一个表示拥挤距离的值
    crowding_distances = np.sum(crowding_matrix, axis=1)

    return crowding_distances

def reduce_by_crowding(objects, number_to_select):
    
    population_idx = np.arange(objects.shape[0])
    crowding_distances = calculate_crowding_distance(objects)
    picked_population_idx = np.zeros((number_to_select))
    picked_objects = np.zeros((number_to_select, len(objects[0, :])))

    for i in range(number_to_select):
        population_size = population_idx.shape[0]
        index1 = rn.randint(0, population_size - 1)
        index2 = rn.randint(0, population_size - 1)
        #保留拥挤距离大的个体
        if crowding_distances[index1] >= crowding_distances[
            index2]:
            picked_population_idx[i] = population_idx[
                index1]
            picked_objects[i, :] = objects[index1, :]
            population_idx = np.delete(population_idx, (index1),axis=0)
            objects = np.delete(objects, (index1), axis=0)
            crowding_distances = np.delete(crowding_distances, (index1),axis=0)
        else:
            picked_population_idx[i] = population_idx[index2]
            picked_objects[i, :] = objects[index2, :]
            population_idx = np.delete(population_idx, (index2), axis=0)
            objects = np.delete(objects, (index2), axis=0)
            crowding_distances = np.delete(crowding_distances, (index2), axis=0)

    picked_population_idx = np.asarray(picked_population_idx, dtype=int)

    return picked_population_idx

def pickup_pareto_index(objects, population_idx):
    population_size = objects.shape[0]
    pareto_front = np.ones(population_size, dtype=bool)
    for i in range(population_size):
        for j in range(population_size):
            if all(objects[j] <= objects[i]) and any(objects[j] < objects[i]):
                pareto_front[i] = 0
                break
    return population_idx[pareto_front]

def build_pareto_population(population, objects, minimum_population_size, maximum_population_size):
    unselected_population_idx = np.arange(population.shape[0])
    all_population_idx = np.arange(population.shape[0])
    pareto_front = []
    while len(pareto_front) < minimum_population_size:
        temp_pareto_front = pickup_pareto_index(
            objects[unselected_population_idx, :], unselected_population_idx)
        combined_pareto_size = len(pareto_front) + len(temp_pareto_front)
        # 检查种群大小是否已超过预设最大值，如果超过需要通过拥挤距离，进行挑选
        if combined_pareto_size > maximum_population_size:
            number_to_select = combined_pareto_size - maximum_population_size
            selected_individuals = (reduce_by_crowding(
                objects[temp_pareto_front], number_to_select))
            temp_pareto_front = temp_pareto_front[selected_individuals]
        pareto_front = np.hstack((pareto_front, temp_pareto_front))
        unselected_set = set(all_population_idx) - set(pareto_front)
        unselected_population_idx = np.array(list(unselected_set))
    population = population[pareto_front.astype(int)]
    return population

# 父代交叉产生子代
def crossover(parent_1, parent_2):
    gene_parent_1 = individual_from_values_to_gene(parent_1)
    gene_parent_2 = individual_from_values_to_gene(parent_2)
    chromosome_length = len(gene_parent_1)
    crossover_point = rn.randint(1, chromosome_length - 1)
    gene_child_1 = np.hstack((gene_parent_1[0:crossover_point],
                         gene_parent_2[crossover_point:]))
    gene_child_2 = np.hstack((gene_parent_2[0:crossover_point],
                         gene_parent_1[crossover_point:]))
    return individual_from_gene_to_value(gene_child_1), individual_from_gene_to_value(gene_child_2)

# 种群变异
def mutate(population, mutation_probability):
    random_mutation_array = np.random.random(size=(population.shape))
    random_mutation_boolean = random_mutation_array <= mutation_probability
    population[random_mutation_boolean] = np.logical_not(population[random_mutation_boolean])
    return population

def is_right_individual(values):
    temp = sum(values)
    if values[0]<=50 and values[1]<=90 and values[2]<=70 and values[3]<=80 and values[4]<=70 and temp>200 and temp<250:
        return True
    return False

# 实属编码，基因数量太小，只通过变异来扩张种群
def breed_population(population):
    new_population = []
    population_size = population.shape[0]
    for i in range(int(population_size / 2)):
        parent_1 = population[rn.randint(0, population_size - 1)]
        parent_2 = population[rn.randint(0, population_size - 1)]
        child_1, child_2 = crossover(parent_1, parent_2)
        if is_right_individual(child_1):
            new_population.append(child_1)
        if is_right_individual(child_2):
            new_population.append(child_2)
    population = np.vstack((population, np.array(new_population)))
    population = np.unique(population, axis=0)
    return population

# main
start_population_size = 500
maximum_generation = 100
minimum_end_population_size = 250
maximum_end_population_size = 300

population = creat_initial_population(start_population_size)

objects = calculate_objects_value(population)
ax = plt.figure().add_subplot(111, projection = '3d')
x = objects[:, 0]
y = objects[:, 1]
z = objects[:, 2]
ax.scatter(x, y, z, marker='.')
ax.set_xlabel('Cost')
ax.set_ylabel('Rejection')
ax.set_zlabel('Lead time')

for generation in range(maximum_generation):
    population = breed_population(population)
    objects = calculate_objects_value(population)
    population = build_pareto_population(
        population, objects, minimum_end_population_size, maximum_end_population_size)


objects = calculate_objects_value(population)
ax = plt.figure().add_subplot(111, projection = '3d')
x = objects[:, 0]
y = objects[:, 1]
z = objects[:, 2]
ax.scatter(x, y, z, marker='.')

ax.set_xlabel('Cost')
ax.set_ylabel('Rejection')
ax.set_zlabel('Lead time')

plt.show()


