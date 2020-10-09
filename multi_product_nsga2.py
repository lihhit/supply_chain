# coding=utf-8
"""
假设F公司需要购买3种产品，需求量如下（含±10吨的允差）：
	产品A	产品B	产品C
需求量	220	240	260
假设有5个供应商可以提供这3种产品，供货能力如下：
供应商	产品A	产品B	产品C	最小起订量
1	55	40	125	30
2	80	120	25	25
3	65	60	150	40
4	85	250	60	50
5	80	100	75	45
供应商提供的产品单价（USD）如下：
供应商	产品A	产品B	产品C
1	200	180	240
2	270	240	300
3	330	300	400)
4	400	360	450
5	350	350	420
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
Min Z_1= 200x_11 y_1+180x_12 y_1+240x_13 y_1+270x_21 y_2+240x_22 y_2+300x_23 y_2+330x_31 y_3+300x_32 y_3+400x_33 y_3+400x_41 y_4+360x_42 y_4+450x_43 y_4+350x_51 y_5+350x_52 y_5+420x_53 y_5
Min Z_2=0.01x_11 y_1+0.09x_12 y_1+0.11x_13 y_1+0.008x_21 y_2+0.007x_22 y_2+0.01x_23 y_2+0.006x_31 y_3+0.006x_32 y_3+0.008x_33 y_3+0.002x_41 y_4+0.003x_42 y_4+0.005x_43 y_4+0.004x_51 y_5+0.005x_52 y_5+0.007x_53 y_5
Min Z_3=0.1x_11 y_1+0.09x_12 y_1+0.08x_13 y_1+0.08x_21 y_2+0.07x_22 y_2+0.07x_23 y_2+0.06x_31 y_3+0.04x_32 y_3+0.05x_33 y_3+0.03x_41 y_4+0.02x_42 y_4+0.01x_43 y_4+0.05x_51 y_5+0.03x_52 y_5+0.04x_53 y_5
约束条件，
210≤x_11 y_1+x_21 y_2+x_31 y_3+x_41 y_4+x_51 y_5≤230
230≤x_12 y_1+x_22 y_2+x_32 y_3+x_42 y_4+x_52 y_5≤250
250≤x_13 y_1+x_23 y_2+x_33 y_3+x_43 y_4+x_53 y_5≤270
x_11≤55，x_12≤40，x_13≤125
x_21≤80，x_22≤120，x_23≤25
x_31≤65，x_32≤60，x_33≤150
x_41≤85，x_42≤250，x_43≤60
x_51≤80，x_52≤100，x_53≤75
x_11+x_12+x_13≥30
x_21+x_22+x_23≥25
x_31+x_32+x_33≥40
x_41+x_42+x_43≥50
x_51+x_52+x_53≥45
y_1,y_2,y_3,y_4,y_5∈[0,1]

"""

import random as rn
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
# 有中文出现的情况，需要u'内容'
from mpl_toolkits.mplot3d import Axes3D as p3d

def plot_3D_figure(objects):
    ax = plt.figure().add_subplot(111, projection='3d')
    x = objects[:, 0]
    y = objects[:, 1]
    z = objects[:, 2]
    ax.scatter(x, y, z, c='b', marker='.')

    ax.set_xlabel(u"成本(USD)")
    ax.set_ylabel(u'废品数(吨)')
    ax.set_zlabel(u'延迟交货数(吨)')

def make_xlsx(filename, population, objects):
    import xlsxwriter
    size = population.shape[0]
    wb = xlsxwriter.Workbook(filename)
    ws = wb.add_worksheet()
    header = wb.add_format({'bold': True})
    header.set_align('center')
    header.set_border(1)
    header.set_bottom(5)
    normal = wb.add_format()
    normal.set_align('center')
    normal.set_border(1)

    ws.write(0, 0, u"供应商1产品A", header)
    ws.write(0, 1, u"供应商1产品B", header)
    ws.write(0, 2, u"供应商1产品C", header)

    ws.write(0, 3, u"供应商2产品A", header)
    ws.write(0, 4, u"供应商2产品B", header)
    ws.write(0, 5, u"供应商2产品C", header)

    ws.write(0, 6, u"供应商3产品A", header)
    ws.write(0, 7, u"供应商3产品B", header)
    ws.write(0, 8, u"供应商3产品C", header)

    ws.write(0, 9, u"供应商4产品A", header)
    ws.write(0, 10, u"供应商4产品B", header)
    ws.write(0, 11, u"供应商4产品C", header)

    ws.write(0, 12, u"供应商5产品A", header)
    ws.write(0, 13, u"供应商5产品B", header)
    ws.write(0, 14, u"供应商5产品C", header)

    ws.write(0, 15, u"成本(USD)", header)
    ws.write(0, 16, u"废品数(吨)", header)
    ws.write(0, 17, u"延迟交货数(吨)", header)

    for idx in range(0, size):
        raw = idx + 1
        ws.write(raw, 0, population[idx][0][0], normal)
        ws.write(raw, 1, population[idx][0][1], normal)
        ws.write(raw, 2, population[idx][0][2], normal)

        ws.write(raw, 3, population[idx][1][0], normal)
        ws.write(raw, 4, population[idx][1][1], normal)
        ws.write(raw, 5, population[idx][1][2], normal)

        ws.write(raw, 6, population[idx][2][0], normal)
        ws.write(raw, 7, population[idx][2][1], normal)
        ws.write(raw, 8, population[idx][2][2], normal)

        ws.write(raw, 9, population[idx][3][0], normal)
        ws.write(raw, 10, population[idx][3][1], normal)
        ws.write(raw, 11, population[idx][3][2], normal)

        ws.write(raw, 12, population[idx][4][0], normal)
        ws.write(raw, 13, population[idx][4][1], normal)
        ws.write(raw, 14, population[idx][4][2], normal)

        ws.write(raw, 15, objects[idx][0], normal)
        ws.write(raw, 16, objects[idx][1], normal)
        ws.write(raw, 17, objects[idx][2], normal)

    widths = [16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16]
    for i in range(len(widths)):
        ws.set_column('%c:%c' % (chr(65 + i), chr(65 + i)), widths[i])

    wb.close()
# 产生初始种群

def creat_initial_population(population_size):
    population = []
    while len(population) < population_size:
        x1 = [
            rn.randint(0, round(55)),
            rn.randint(0, round(40)),
            rn.randint(0, round(125))
        ]
        x2 = [
            rn.randint(0, round(80)),
            rn.randint(0, round(120)),
            rn.randint(0, round(25))
        ]
        x3 = [
            rn.randint(0, round(65)),
            rn.randint(0, round(60)),
            rn.randint(0, round(150))
        ]
        x4 = [
            rn.randint(0, round(85)),
            rn.randint(0, round(250)),
            rn.randint(0, round(60))
        ]
        x5 = [
            rn.randint(0, round(80)),
            rn.randint(0, round(100)),
            rn.randint(0, round(75))
        ]

        y1 = rn.randint(0, 1)
        y2 = rn.randint(0, 1)
        y3 = rn.randint(0, 1)
        y4 = rn.randint(0, 1)
        y5 = rn.randint(0, 1)

        if 210 <= x1[0]*y1 + x2[0]*y2 + x3[0]*y3 + x4[0]*y4 + x5[0]*y5 <= 230 and \
           230 <= x1[1]*y1 + x2[1]*y2 + x3[1]*y3 + x4[1]*y4 + x5[1]*y5 <= 250 and \
           250 <= x1[2]*y1 + x2[2]*y2 + x3[2]*y3 + x4[2]*y4 + x5[2]*y5 <= 270 and \
           x1[0]+x1[1]+x1[2] >= 30 and \
           x2[0]+x2[1]+x2[2] >= 25 and \
           x3[0]+x3[1]+x3[2] >= 40 and \
           x4[0]+x4[1]+x4[2] >= 50 and \
           x5[0]+x5[1]+x5[2] >= 45:

            population.append([[x*y1 for x in x1], [x*y2 for x in x2], [x*y3 for x in x3], [x*y4 for x in x4], [x*y5 for x in x5]])

    return np.array(population)


# 计算目标函数的值
#Min Z_1= 200x_11+180x_12+240x_13+270x_21+240x_22+300x_23+330x_31+300x_32+400x_33+400x_41+360x_42+450x_43+350x_51+350x_52+420x_53
#Min Z_2=0.01x_11+0.009x_12+0.011x_13+0.008x_21+0.007x_22+0.01x_23+0.006x_31+0.006x_32+0.008x_33+0.002x_41+0.003x_42+0.005x_43+0.004x_51+0.005x_52+0.007x_53
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
        Z2 = 0.01*x[0][0]+0.009*x[0][1]+0.011*x[0][2] + \
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
    """
    plot_3D_figure(objects)
    ax = plt.figure().add_subplot(111, projection='3d')
    x = objects[pareto_front.astype(int)][:, 0]
    y = objects[pareto_front.astype(int)][:, 1]
    z = objects[pareto_front.astype(int)][:, 2]
    ax.scatter(x, y, z, c='b', marker='.')

    x = objects[unselected_population_idx][:, 0]
    y = objects[unselected_population_idx][:, 1]
    z = objects[unselected_population_idx][:, 2]
    ax.scatter(x, y, z, c='r', marker='.')

    ax.set_xlabel(u"成本(USD)")
    ax.set_ylabel(u'废品数(吨)')
    ax.set_zlabel(u'延迟交货数(吨)')
    """
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

def is_right_individual(x):
    if((0<=x[0][0]<=55 and 0<=x[0][1]<=40 and 0<=x[0][2]<=125 and x[0][0]+x[0][1]+x[0][2]>=30) or any(x[0])==False) \
        and ((0<=x[1][0]<=80 and 0<=x[1][1]<=120 and 0<=x[1][2]<=25 and x[1][0]+x[1][1]+x[1][2]>=25) or any(x[1])==False) \
        and ((0<=x[2][0]<=65 and 0<=x[2][1]<=60 and 0<=x[2][2]<=150 and x[2][0]+x[2][1]+x[2][2]>=40) or any(x[2])==False)\
        and ((0<=x[3][0]<=85 and 0<=x[3][1]<=250 and 0<=x[3][2]<=60 and x[3][0]+x[3][1]+x[3][2]>=50) or any(x[3])==False) \
        and ((0<=x[4][0]<=80 and 0<=x[4][1]<=100 and 0<=x[4][2]<75 and x[4][0]+x[4][1]+x[4][2]>=45) or any(x[4])==False) \
        and 210 <= x[0][0]+x[1][0]+x[2][0]+x[3][0]+x[4][0] <= 230 \
        and 230 <= x[0][1]+x[1][1]+x[2][1]+x[3][1]+x[4][1] <=250 \
        and 250 <= x[0][2]+x[1][2]+x[2][2]+x[3][2]+x[4][2] <= 270:
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
    print("breed population size:"+str(population.shape[0]))
    return population


# main
start_population_size = 500
maximum_generation = 30
minimum_end_population_size = 450
maximum_end_population_size = 550

population = creat_initial_population(start_population_size)

objects = calculate_objects_value(population)
plot_3D_figure(objects)

for generation in range(maximum_generation):
    population = breed_population(population)
    objects = calculate_objects_value(population)
    population = build_pareto_population(population, objects,
                                         minimum_end_population_size,
                                         maximum_end_population_size)

objects = calculate_objects_value(population)
make_xlsx("/Users/bryli/PycharmProjects/supply_chain/multi_data.xlsx", population,objects)
plot_3D_figure(objects)
plt.show()
