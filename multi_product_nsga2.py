# coding=utf-8

"""
"""

import random as rn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D as p3d



# 产生初始种群
def creat_initial_population(population_size):
    population = []
    while len(population) < population_size:
        x1 = [rn.uniform(0,51.322),rn.uniform(0,36.710),rn.uniform(0,116.776)]
        x2 = [rn.uniform(0,73.630),rn.uniform(0,111.776),rn.uniform(0,23.355)]
        x3 = [rn.uniform(0,59.799),rn.uniform(0,55.348),rn.uniform(0,138.486)]
        x4 = [rn.uniform(0,80.888),rn.uniform(0,231.907),rn.uniform(0,55.348)]
        x5 = [rn.uniform(0,73.421),rn.uniform(0,92.644),rn.uniform(0,69.302)]


        if 208.486 <= x1[0]+x2[0]+x3[0]+x4[0]+x5[0] <= 231.514 and 230.131 <= x1[1]+x2[1]+x3[1]+x4[1]+x5[1] <=249.869 and 246.841 <= x1[2]+x2[2]+x3[2]+x4[2]+x5[2] <= 273.159:
            population.append([x1, x2, x3, x4, x5])

    return np.array(population)


# 计算目标函数的值
# Min Z_1=5x_11+6x_12+4x_13 + 5x_21+4x_22+5x_23 + 4x_31+3x_32+2x_33 + 11x_41+6x_42+5x_43 + 3x_51+5x_52+9x_53
# Min Z_2=0.05x_11+0.06x_12+0.07x_13+0.05x_21+0.03x_22+0.01x_23+0.1x_31+0.09x_32+0.09x_33 + 0.11x_41+0.07x_42+0.1x_43 + 0.1x_51+0.07x_52+0.04x_53
# Min Z_3=10x_11+8x_12+2x_13 + 6x_21+3x_22+8x_23 + 8x_31+3x_32+9x_33 + 3x_41+4x_42+7x_43 + 8x_51+3x_52+4x_53


def calculate_objects_value(population):
    objects = []
    size = population.shape[0]
    for i in range(size):
        x = population[i]
        Z1 =  5*x[0][0]+6*x[0][1]+4*x[0][2] + \
                  5*x[1][0]+4*x[1][1]+5*x[1][2] + \
                  4*x[2][0]+3*x[2][1]+2*x[2][2] + \
                  11*x[3][0]+6*x[3][1]+5*x[3][2] +\
                  3*x[4][0]+5*x[4][1]+9*x[4][2]
        Z2 = 0.05*x[0][0]+0.06*x[0][1]+0.07*x[0][2] + \
             0.05*x[1][0]+0.03*x[1][1]+0.01*x[1][2] + \
             0.1*x[2][0]+0.09*x[2][1]+0.09*x[2][2] +  \
             0.11*x[3][0]+0.07*x[3][1]+0.1*x[3][2] +  \
             0.1*x[4][0]+0.07*x[4][1]+0.04*x[4][2]
        Z3 = 10*x[0][0]+8*x[0][1]+2*x[0][2] + \
             6*x[1][0]+3*x[1][1]+8*x[1][2] + \
             8*x[2][0]+3*x[2][1]+9*x[2][2] + \
             3*x[3][0]+4*x[3][1]+7*x[3][2] +\
             8*x[4][0]+3*x[4][1]+4*x[4][2]
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
        # 保留拥挤距离大的个体
        if crowding_distances[index1] >= crowding_distances[
            index2]:
            picked_population_idx[i] = population_idx[
                index1]
            picked_objects[i, :] = objects[index1, :]
            population_idx = np.delete(population_idx, (index1), axis=0)
            objects = np.delete(objects, (index1), axis=0)
            crowding_distances = np.delete(crowding_distances, (index1), axis=0)
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
    rate = rn.random()
    child_1 = rate*parent_1+(1-rate)*parent_2
    child_2 = rate*parent_2+(1-rate)*parent_1
    return child_1,child_2

# 种群变异
def mutate(population, mutation_probability):
    random_mutation_array = np.random.random(size=(population.shape))
    random_mutation_boolean = random_mutation_array <= mutation_probability
    population[random_mutation_boolean] = np.logical_not(population[random_mutation_boolean])
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
maximum_generation = 100
minimum_end_population_size = 450
maximum_end_population_size = 550

population = creat_initial_population(start_population_size)

objects = calculate_objects_value(population)
ax = plt.figure().add_subplot(111, projection='3d')
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
print(objects)
print(population)
ax = plt.figure().add_subplot(111, projection='3d')
x = objects[:, 0]
y = objects[:, 1]
z = objects[:, 2]
ax.scatter(x, y, z, marker='.')

ax.set_xlabel('Cost')
ax.set_ylabel('Rejection')
ax.set_zlabel('Lead time')

plt.show()