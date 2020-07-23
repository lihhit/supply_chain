# coding:utf-8
"""
假定F公司采用定期订货法采购某一种生产原料，月需求量D是150吨（含±10吨的允差），目前有五个供应商可供选择，忽略这五个供应商服务水平的优劣等级，五个供应商的供货能力如下表所示：
表4.1 单产品的各供应商的供货能力
供应商i	产品FOB单价p_i(USD/吨)	废品率q_i(%)	最小起订量M_i(吨)	最大供货量S_i(吨)	延迟交货率
（%）
1	200	1	10	80	9
2	270	0.8	10	50	7
3	330	0.6	5	20	4
4	400	0.2	10	50	2
5	350	0.4	5	30	3

将参数代入目标模型，
MinZ_1=200x_1 y_1+270x_2 y_2+330x_3 y_3+400x_4 y_4+350x_5 y_5
Min Z_2= 0.01x_1 y_1+0.008x_2 y_2+0.006x_3 y_3+0.002x_4 y_4+0.004x_5 y_5
Min Z_3= 0.09x_1 y_1+0.07x_2 y_2+0.04x_3 y_3+0.02x_4 y_4+0.03x_5 y_5
约束条件:
140≤x_1 y_1+x_2 y_2+x_3 y_3+x_4 y_4+x_5 y_5≤160
10≤x_1≤80,10≤x_2≤50,5≤x_3≤20,10≤x_4≤50,5≤x_5≤30
y_1,y_2,y_3,y_4,y_5∈[0,1]

"""

import random as rn
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
# 有中文出现的情况，需要u'内容'
from mpl_toolkits.mplot3d import Axes3D as p3d




def individual_from_values_to_gene(values):
    gene = []
    for i in range(len(values)):
        gene = gene + [int(x) for x in '{:08b}'.format(values[i])]
    return gene


def individual_from_gene_to_value(gene):
    values = []
    values.append(int("".join(str(i) for i in gene[:8]), 2))
    values.append(int("".join(str(i) for i in gene[8:16]), 2))
    values.append(int("".join(str(i) for i in gene[16:24]), 2))
    values.append(int("".join(str(i) for i in gene[24:32]), 2))
    values.append(int("".join(str(i) for i in gene[32:40]), 2))
    return values


# 产生初始种群
def creat_initial_population(population_size):
    population = []
    while len(population) < population_size:
        x1 = rn.randint(10,80)
        x2 = rn.randint(10,50)
        x3 = rn.randint(5,20)
        x4 = rn.randint(10,50)
        x5 = rn.randint(5,30)

        y1 = rn.randint(0, 1)
        y2 = rn.randint(0, 1)
        y3 = rn.randint(0, 1)
        y4 = rn.randint(0, 1)
        y5 = rn.randint(0, 1)

        temp = x1*y1+x2*y2+x3*y3+x4*y4+x5*y5
        if 140 <= temp <= 150:
            population.append(([x1*y1, x2*y2, x3*y3, x4*y4, x5*y5]))

    return np.array(population)


# 计算目标函数的值
# MinZ_1=420x_1 y_1+427x_2 y_2+433x_3 y_3+440x_4 y_4+435x_5 y_5
# Min Z_2= 0.0005x_1 y_1+0.0004x_2 y_2+0.0003x_3 y_3+0.0002x_4 y_4+0.0003x_5 y_5
# Min Z_3= 15y_1+12y_2+10y_3+14y_4+10y_5


def calculate_objects_value(population):
    objects = []
    size = population.shape[0]
    for i in range(size):
        x = population[i]
        y = []
        for j in range(0,5):
            if x[j]>0: y.append(1)
            else: y.append(0)
        Z1 = 200*x[0]+270*x[1]+330*x[2]+400*x[3]+350*x[4]
        Z2 = 0.01*x[0]+0.008*x[1]+0.006*x[2]+0.002*x[3]+0.004*x[4]
        Z3 = 0.09*x[0]+0.07*x[1]+0.04*x[2]+0.02*x[3]+0.03*x[4]
        objects.append([Z1, Z2, Z3])
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

        sorted_objects_index = np.argsort(normed_objects[:, col])

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
        # 保留拥挤距离大的个体
        if crowding_distances[index1] >= crowding_distances[index2]:
            picked_population_idx[i] = population_idx[index1]
            picked_objects[i, :] = objects[index1, :]
            population_idx = np.delete(population_idx, (index1), axis=0)
            objects = np.delete(objects, (index1), axis=0)
            crowding_distances = np.delete(crowding_distances, (index1),
                                           axis=0)
        else:
            picked_population_idx[i] = population_idx[index2]
            picked_objects[i, :] = objects[index2, :]
            population_idx = np.delete(population_idx, (index2), axis=0)
            objects = np.delete(objects, (index2), axis=0)
            crowding_distances = np.delete(crowding_distances, (index2),
                                           axis=0)

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


def build_pareto_population(population, objects, minimum_population_size,
                            maximum_population_size):
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
    gene_child_1 = np.hstack(
        (gene_parent_1[0:crossover_point], gene_parent_2[crossover_point:]))
    gene_child_2 = np.hstack(
        (gene_parent_2[0:crossover_point], gene_parent_1[crossover_point:]))
    return individual_from_gene_to_value(
        gene_child_1), individual_from_gene_to_value(gene_child_2)


# 种群变异
def mutate(population, mutation_probability):
    random_mutation_array = np.random.random(size=(population.shape))
    random_mutation_boolean = random_mutation_array <= mutation_probability
    population[random_mutation_boolean] = np.logical_not(
        population[random_mutation_boolean])
    return population

# 5000≤x_1≤8000,3000≤x_2≤5000,1000≤x_3≤2000,3000≤x_4≤5000,1500≤x_5≤3000
# 14925≤x_1 y_1+x_2 y_2+x_3 y_3+x_4 y_4+x_5 y_5≤15075

def is_right_individual(values):
    temp = sum(values)
    if 10<=values[0] <= 80 and 10<=values[1] <= 50 and 5<=values[2] <= 20 \
            and 10<=values[3] <= 50 and 5<=values[4] <= 30 and 140 <= temp <= 160:
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
    if new_population:
        population = np.vstack((population, np.array(new_population)))
    population = np.unique(population, axis=0)
    return population


# main
start_population_size = 500
maximum_generation = 50
minimum_end_population_size = 450
maximum_end_population_size = 550

population = creat_initial_population(start_population_size)

objects = calculate_objects_value(population)

ax = plt.figure().add_subplot(111, projection='3d')
x = objects[:, 0]
y = objects[:, 1]
z = objects[:, 2]
ax.scatter(x, y, z, marker='.')
ax.set_xlabel(u"成本(USD)")
ax.set_ylabel(u'废品数(吨)')
ax.set_zlabel(u'延迟交货数(吨)')

for generation in range(maximum_generation):
    population = breed_population(population)
    objects = calculate_objects_value(population)
    population = build_pareto_population(population, objects,
                                         minimum_end_population_size,
                                         maximum_end_population_size)

objects = calculate_objects_value(population)

import xlsxwriter
size = population.shape[0]
wb = xlsxwriter.Workbook("/Users/bryli/PycharmProjects/supply_chain/single_data.xlsx")
ws = wb.add_worksheet()
header = wb.add_format({'bold': True})
header.set_align('center')
header.set_border(1)
header.set_bottom(5)
normal = wb.add_format()
normal.set_align('center')
normal.set_border(1)


ws.write(0,0,u"供应商1",header)
ws.write(0,1,u"供应商2",header)
ws.write(0,2,u"供应商3",header)
ws.write(0,3,u"供应商4",header)
ws.write(0,4,u"供应商5",header)
ws.write(0,5,u"成本(USD)",header)
ws.write(0,6,u"废品数(吨)",header)
ws.write(0,7,u"延迟交货数(吨)",header)

for idx in range(0,size):
    raw = idx+1
    ws.write(raw, 0, population[idx][0],normal)
    ws.write(raw, 1, population[idx][1],normal)
    ws.write(raw, 2, population[idx][2],normal)
    ws.write(raw, 3, population[idx][3],normal)
    ws.write(raw, 4, population[idx][4],normal)
    ws.write(raw, 5, objects[idx][0],normal)
    ws.write(raw, 6, objects[idx][1],normal)
    ws.write(raw, 7, objects[idx][2],normal)

widths = [8, 8, 8, 8, 8, 16, 16,16]
for i in range(len(widths)):
    ws.set_column('%c:%c' % (chr(65 + i), chr(65 + i)), widths[i])

wb.close()
ax = plt.figure().add_subplot(111, projection='3d')
x = objects[:, 0]
y = objects[:, 1]
z = objects[:, 2]
ax.scatter(x, y, z, marker='.')

ax.set_xlabel(u"成本(USD)")
ax.set_ylabel(u'废品数(吨)')
ax.set_zlabel(u'延迟交货数(吨)')

plt.show()

print("end")
