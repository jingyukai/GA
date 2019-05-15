import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
import random
import matplotlib.pyplot as plt
import copy

# 读取 questionnaire.csv, 存到 df 中，并将第一列（uid）设置为索引
### CODE HERE

df = pd.read_csv('questionnaire.csv',header=0,index_col=0)

initial_population = []

for x in range(10):
    user_mat = np.random.permutation(range(1, 101)).reshape((10,10))
    initial_population.append(user_mat)


def calculate_interest_distance(person, *others):
    people = [df.ix[person, 0:3]]
    for other in others:
        if other:
            people.append(df.ix[other, 0:3])

    return np.average(pdist(people, 'cityblock' ) if len(people) > 1 else 0)

def get_element_safe(sequence, position):
    if not position[0] in range(0,9) or not position[1] in range(0,9):
        return None
    else:
        return sequence.item(position)



def calculate_chromosome_fitness(chromosome):
    fitnesses = []
    for index, gene in np.ndenumerate(chromosome):
        next_to  = calculate_interest_distance(gene, get_element_safe(chromosome, (index[0], index[1] - 1)),
                                                     get_element_safe(chromosome, (index[0], index[1] + 1))) #左右

        in_front = calculate_interest_distance(gene, get_element_safe(chromosome, (index[0]-1, index[1] )))

        behind   = calculate_interest_distance(gene, get_element_safe(chromosome, (index[0]+1, index[1])))

        fitnesses.append(sum([next_to, in_front, behind]) / 3)
    return np.average(fitnesses)

def recombine(mother, father):
    child = np.ones((10,10)) * -1
    # 遍历每个用户
    for index, uid in enumerate(df.index):
        if uid <= 50:
            # 如果uid <=50，则position为mother中该uid的位置
            position = np.where(mother == uid)
        else:
            # 如果uid > 50，则position为father中该uid的位置
            position = np.where(father == uid)

        if child[position] != -1:
            # 处理冲突
            filtered = np.where(np.equal(child, -1))
            position = random.choice(list(zip(filtered[0], filtered[1])))

        child[position] = uid

    return child

def mutate_chromosome(chromosome):
    first = (random.randrange(0, 10, 1), random.randrange(0, 10, 1))
    second = (random.randrange(0, 10, 1), random.randrange(0, 10, 1))

    chromosome[first[0], first[1]], chromosome[second[0], second[1]] = chromosome[second[0], second[1]], chromosome[first[0], first[1]]
    return chromosome

def draw_population(population, generation):
    #get_ids = np.vectorize(get_id)
    figure, axes = plt.subplots(1, 10, figsize=(20, 3))

    for index, chromosome in enumerate(population):
        ids = chromosome
        axes[index].imshow(ids, interpolation="nearest", cmap=plt.cm.plasma)
        subplot_title = "Chromosome " + str(index)
        axes[index].set_title(subplot_title)

    plot_title = "Generation " + str(generation)
    figure.suptitle(plot_title)
    plt.show()

most_fit = []

# 当前的适应度
max_fitness = None

# 适应度最高的染色体
fittest_chromosome = None

### 从generations取出某一代的所有染色体，并排序
def get_chromosomes_in_generation(generations, generation_num):
    # Get only the chromosomes in this generation
    generation = list(filter(lambda k: k['generation'] == generation_num,
                             generations))
    return sorted(generation, key=lambda k: k['fitness'])


### 更新 most_fit / max_fitness / fittest_chromosome
def sort_chromosomes(generation, most_fit):
    max_fitness = generation[0]['fitness']
    most_fit.append(max_fitness)
    fittest_chromosome = generation[0]['chromosome']
    return most_fit, fittest_chromosome


# 请将以下Comment 添加到代码的合适位置

def copy_chromosomes(generation, generations):
    new_generation = copy.deepcopy(generation)

    new_generation[-1]['active'] = False
    new_generation[-2]['active'] = False# C. 将适应度最差的两个染色体杀掉

    active = list(filter(lambda k: k['active'] == True, new_generation))# D. 找到仍存活的染色体，并更新代数
    for chromosome in active:
        chromosome['generation'] += 1

    generations.extend(active)# B. 将新一代染色体加入到generations中并返回
    return generations


def mate_chromosomes(generation, generations, generation_num):

    child = recombine(generation[0]['chromosome'], generation[1]['chromosome'])# A. 将适应度最高的两条染色体交配
    child = mutate_chromosome(child)
    generations.append({
      "active": True,
      "chromosome": child,
      "generation": generation_num + 1,
      "fitness": calculate_chromosome_fitness(child)
    })

    other = recombine(generation[2]['chromosome'], generation[3]['chromosome'])# E. 将适应度第三和第四高的两条染色体交配
    other = mutate_chromosome(other)
    generations.append({
      "active": True,
      "chromosome": other,
      "generation": generation_num + 1,
      "fitness": calculate_chromosome_fitness(other)
    })
    return generations


def run_ga(input_generations):
    fittest = []
    max_fitness = None
    fittest_chromo = None
    for generation_num in range(500):
        if max_fitness and max_fitness < 5:
            break

        generation = get_chromosomes_in_generation(input_generations, generation_num)
        fittest, fittest_chromo = sort_chromosomes(generation, most_fit)

        input_generations = copy_chromosomes(generation, input_generations)
        input_generations = mate_chromosomes(generation, input_generations, generation_num)

        #draw_population(map(lambda d: d['chromosome'],
        #                get_chromosomes_in_generation(input_generations, generation_num)), generation_num)

    return fittest, fittest_chromo


generations = []

for chromosome in initial_population:
    generations.append({
        "generation": 0,
        "fitness": calculate_chromosome_fitness(chromosome),
        "chromosome": chromosome,
        "active": True
    })

generations=sorted(generations,key=lambda d:d['fitness'], reverse=False)



most_fit, fittest_chromosome = run_ga(generations)

print(most_fit, fittest_chromosome)

plt.plot(most_fit)
plt.title("Most Fit Trend")
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.show()
