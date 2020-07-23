# coding=utf-8
"""
假设F公司需要购买3种产品，需求的正态分布参数如下：
	产品A	产品B	产品C
期望μ_(D_j )	220	240	260
方差σ_(D_j)^2	49	36	64
F_(D_j)^(-1) (α_(D_j )=0.05)	208.486	230.131	246.841
F_(D_j)^(-1) (β_(D_j )=0.95)	231.514	249.869	273.159

假设有5个供应商可以提供这3种产品，供货能力的正态分布N(μ_(C_ij ), σ_(C_ij)^2)参数如下：
供应商	产品A	产品B	产品C
1	N (55,5)	N (40,4)	N (125,25)
2	N (80,15)	N (120,25)	N (25,1)
3	N (65, 10)	N (60, 8)	N (150,49)
4	N (85,6.25)	N (250, 121)	N (60,8)
5	N (80,16)	N (100,20)	N (75,12)
假设所有供应商i能满足j产品供应的概率为α_(C_ij )=0.95，计算从供应商i处购买j产品的数量为x_ij的上限F_(C_ij)^(-1) (1-α_(C_ij ) )=F_(C_ij)^(-1) (0.05)的值，如下图所示：
供应商	产品A	产品B	产品C
1	51.322	36.710	116.776
2	73.630	111.776	23.355
3	59.799	55.348	138.486
4	80.888	231.907	55.348
5	73.421	92.644	69.302

供应商提供的产品单价（USD）服从正态分布：
供应商	产品A	产品B	产品C
1	N(200, σ_(p_11)^2)	N(180, σ_(p_12)^2)	N(240, σ_(p_13)^2)
2	N(270, σ_(p_21)^2)	N(240, σ_(p_22)^2)	N(300, σ_(p_23)^2)
3	N(330, σ_(p_31)^2)	N(300, σ_(p_32)^2)	N(400, σ_(p_33)^2)
4	N(400, σ_(p_41)^2)	N(360, σ_(p_42)^2)	N(450, σ_(p_43)^2)
5	N(350, σ_(p_51)^2)	N(350, σ_(p_52)^2)	N(420, σ_(p_53)^2)
供应商提供产品的废品率（%）：
供应商	产品A	产品B	产品C
1	1	0.9	1.1
2	0.8	0.7	1
3	0.6	0.6	0.8
4	0.2	0.3	0.5
5	0.4	0.5	0.7

供应商提供的产品延迟交货率(%)：
供应商	产品A	产品B	产品C
1	10	9	8
2	8	7	7
3	6	4	5
4	3	2	1
5	5	3	4
将所有参数代入目标模型，
Min Z_1= 200x_11+180x_12+240x_13+270x_21+240x_22+300x_23+330x_31+300x_32+400x_33+400x_41+360x_42+450x_43+350x_51+350x_52+420x_53
Min Z_2=0.01x_11+0.09x_12+0.11x_13+0.008x_21+0.007x_22+0.01x_23+0.006x_31+0.006x_32+0.008x_33+0.002x_41+0.003x_42+0.005x_43+0.004x_51+0.005x_52+0.007x_53
Min Z_3=0.1x_11+0.09x_12+0.08x_13+0.08x_21+0.07x_22+0.07x_23+0.06x_31+0.04x_32+0.05x_33+0.03x_41+0.02x_42+0.01x_43+0.05x_51+0.03x_52+0.04x_53
约束条件，
208.486≤x_11+x_21+x_31+x_41+x_51≤231.514
230.131≤x_12+x_22+x_32+x_42+x_52≤249.869
246.841≤x_13+x_23+x_33+x_43+x_53≤273.159
x_11≤51.322，x_12≤36.710，x_13≤116.776
x_21≤73.630，x_22≤111.776，x_23≤23.355
x_31≤59.799，x_32≤55.348，x_33≤138.486
x_41≤80.888，x_42≤231.907，x_43≤55.348
x_51≤73.421，x_52≤92.644，x_53≤69.302

"""

import random as rn
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
# 有中文出现的情况，需要u'内容'
from mpl_toolkits.mplot3d import Axes3D as p3d


# 产生初始种群
def creat_initial_population(population_size):
    population = []
    while len(population) < population_size:
        x1 = [
            rn.randint(0, round(51.322)),
            rn.randint(0, round(36.710)),
            rn.randint(0, round(116.776))
        ]
        x2 = [
            rn.randint(0, round(73.630)),
            rn.randint(0, round(111.776)),
            rn.randint(0, round(23.355))
        ]
        x3 = [
            rn.randint(0, round(59.799)),
            rn.randint(0, round(55.348)),
            rn.randint(0, round(138.486))
        ]
        x4 = [
            rn.randint(0, round(80.888)),
            rn.randint(0, round(231.907)),
            rn.randint(0, round(55.348))
        ]
        x5 = [
            rn.randint(0, round(73.421)),
            rn.randint(0, round(92.644)),
            rn.randint(0, round(69.302))
        ]

        if 208.486 <= x1[0] + x2[0] + x3[0] + x4[0] + x5[
                0] <= 231.514 and 230.131 <= x1[1] + x2[1] + x3[1] + x4[
                    1] + x5[1] <= 249.869 and 246.841 <= x1[2] + x2[2] + x3[
                        2] + x4[2] + x5[2] <= 273.159:
            population.append([x1, x2, x3, x4, x5])

    return np.array(population)


# 计算目标函数的值
#Min Z_1= 200x_11+180x_12+240x_13+270x_21+240x_22+300x_23+330x_31+300x_32+400x_33+400x_41+360x_42+450x_43+350x_51+350x_52+420x_53
#Min Z_2=0.01x_11+0.09x_12+0.11x_13+0.008x_21+0.007x_22+0.01x_23+0.006x_31+0.006x_32+0.008x_33+0.002x_41+0.003x_42+0.005x_43+0.004x_51+0.005x_52+0.007x_53
#Min Z_3=0.1x_11+0.09x_12+0.08x_13+0.08x_21+0.07x_22+0.07x_23+0.06x_31+0.04x_32+0.05x_33+0.03x_41+0.02x_42+0.01x_43+0.05x_51+0.03x_52+0.04x_53

def calculate_objects_value(population):
    objects = []
    size = population.shape[0]
    for i in range(size):
        x = population[i]
        Z1 =  200*x[0][0]+180*x[0][1]+240*x[0][2] + \
                  270*x[1][0]+240*x[1][1]+300*x[1][2] + \
                  330*x[2][0]+300*x[2][1]+400*x[2][2] + \
                  400*x[3][0]+360*x[3][1]+450*x[3][2] +\
                  350*x[4][0]+350*x[4][1]+420*x[4][2]
        Z2 = 0.01*x[0][0]+0.09*x[0][1]+0.11*x[0][2] + \
             0.008*x[1][0]+0.007*x[1][1]+0.01*x[1][2] + \
             0.006*x[2][0]+0.006*x[2][1]+0.008*x[2][2] +  \
             0.002*x[3][0]+0.003*x[3][1]+0.005*x[3][2] +  \
             0.004*x[4][0]+0.005*x[4][1]+0.007*x[4][2]
        Z3 = 0.1*x[0][0]+0.09*x[0][1]+0.08*x[0][2] + \
             0.08*x[1][0]+0.07*x[1][1]+0.07*x[1][2] + \
             0.06*x[2][0]+0.04*x[2][1]+0.05*x[2][2] + \
             0.03*x[3][0]+0.02*x[3][1]+0.01*x[3][2] +\
             0.05*x[4][0]+0.03*x[4][1]+0.04*x[4][2]
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
    rate = rn.random()
    child_1 = np.rint(rate * parent_1 + (1 - rate) * parent_2)
    child_2 = np.rint(rate * parent_2 + (1 - rate) * parent_1)
    return child_1, child_2


# 种群变异
def mutate(population, mutation_probability):
    random_mutation_array = np.random.random(size=(population.shape))
    random_mutation_boolean = random_mutation_array <= mutation_probability
    population[random_mutation_boolean] = np.logical_not(
        population[random_mutation_boolean])
    return population


# 约束条件，
#208.486≤x_11+x_21+x_31+x_41+x_51≤231.514
#230.131≤x_12+x_22+x_32+x_42+x_52≤249.869
#246.841≤x_13+x_23+x_33+x_43+x_53≤273.159
#x_11≤51.322，x_12≤36.710，x_13≤116.776
#x_21≤73.630，x_22≤111.776，x_23≤23.355
#x_31≤59.799，x_32≤55.348，x_33≤138.486
#x_41≤80.888，x_42≤231.907，x_43≤55.348
#x_51≤73.421，x_52≤92.644，x_53≤69.302
def is_right_individual(x):
    if  0<=x[0][0]<=51.322 and 0<=x[0][1]<=36.710 and 0<=x[0][2]<=116.776 \
        and 0<=x[1][0]<=73.630 and 0<=x[1][1]<=111.776 and 0<=x[1][2]<=23.355 \
        and 0<=x[2][0]<=59.799 and 0<=x[2][1]<=55.348 and 0<=x[2][2]<=138.486 \
        and 0<=x[3][0]<=80.888 and 0<=x[3][1]<=231.907 and 0<=x[3][2]<=55.348 \
        and 0<=x[4][0]<=73.421 and 0<=x[4][1]<=92.644 and 0<=x[4][2]<69.302 \
        and 208.486 <= x[0][0]+x[1][0]+x[2][0]+x[3][0]+x[4][0] <= 231.514 \
        and 230.131 <= x[0][1]+x[1][1]+x[2][1]+x[3][1]+x[4][1] <=249.869 \
        and 246.841 <= x[0][2]+x[1][2]+x[2][2]+x[3][2]+x[4][2] <= 273.159:
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
wb = xlsxwriter.Workbook("/Users/bryli/PycharmProjects/supply_chain/multi_data.xlsx")
ws = wb.add_worksheet()
header = wb.add_format({'bold': True})
header.set_align('center')
header.set_border(1)
header.set_bottom(5)
normal = wb.add_format()
normal.set_align('center')
normal.set_border(1)


ws.write(0,0,u"供应商1产品A",header)
ws.write(0,1,u"供应商1产品B",header)
ws.write(0,2,u"供应商1产品C",header)

ws.write(0,3,u"供应商2产品A",header)
ws.write(0,4,u"供应商2产品B",header)
ws.write(0,5,u"供应商2产品C",header)

ws.write(0,6,u"供应商3产品A",header)
ws.write(0,7,u"供应商3产品B",header)
ws.write(0,8,u"供应商3产品C",header)

ws.write(0,9,u"供应商4产品A",header)
ws.write(0,10,u"供应商4产品B",header)
ws.write(0,11,u"供应商4产品C",header)

ws.write(0,12,u"供应商5产品A",header)
ws.write(0,13,u"供应商5产品B",header)
ws.write(0,14,u"供应商5产品C",header)

ws.write(0,15,u"成本(USD)",header)
ws.write(0,16,u"废品数(吨)",header)
ws.write(0,17,u"延迟交货数(吨)",header)

for idx in range(0,size):
    raw = idx+1
    ws.write(raw, 0, population[idx][0][0],normal)
    ws.write(raw, 1, population[idx][0][1], normal)
    ws.write(raw, 2, population[idx][0][2], normal)

    ws.write(raw, 3, population[idx][1][0],normal)
    ws.write(raw, 4, population[idx][1][1], normal)
    ws.write(raw, 5, population[idx][1][2], normal)

    ws.write(raw, 6, population[idx][2][0],normal)
    ws.write(raw, 7, population[idx][2][1], normal)
    ws.write(raw, 8, population[idx][2][2], normal)

    ws.write(raw, 9, population[idx][3][0],normal)
    ws.write(raw, 10, population[idx][3][1], normal)
    ws.write(raw, 11, population[idx][3][2], normal)

    ws.write(raw, 12, population[idx][4][0],normal)
    ws.write(raw, 13, population[idx][4][1], normal)
    ws.write(raw, 14, population[idx][4][2], normal)

    ws.write(raw, 15, objects[idx][0],normal)
    ws.write(raw, 16, objects[idx][1],normal)
    ws.write(raw, 17, objects[idx][2],normal)

widths = [16, 16,16, 16,16, 16,16, 16,16, 16,16, 16,16, 16,16, 16,16, 16]
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
