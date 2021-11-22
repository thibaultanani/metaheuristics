import utility.utility as utility

import multiprocessing
import numpy as np
import sys
import random
import os
import heapq
import psutil
from operator import itemgetter
import time
from datetime import timedelta

import warnings

warnings.filterwarnings('ignore')

np.set_printoptions(threshold=sys.maxsize)


class Genetic:

    def __init__(self, dataset, capacity, list_exp, seed, pop, gen, mut):
        self.path1 = os.path.dirname(os.getcwd()) + '/in/'
        self.path2 = os.path.dirname(os.getcwd()) + '/out'
        self.dataset = dataset
        self.data = utility.read(filename=(self.path1 + dataset))
        self.n_objects = len(self.data.index)
        self.capacity = capacity
        self.list_exp = list_exp
        self.seed = seed
        self.values, self.weights = self.data['Profit'], self.data['Weight']
        self.n_pop = pop
        self.n_gen = gen
        self.n_mut = mut
        utility.cleanOut()

    def crossover(self, p1, p2):
        decomp = utility.getNumberdecomposition(int(self.n_objects / 2))
        part1 = p1[0:decomp[0]]
        part2 = p2[decomp[0]:-decomp[1]]
        part3 = p1[-decomp[1]:]
        b = bool(random.randint(0, 3))
        if b == 1:
            c1 = list(part1) + list(part2) + list(part3)
            c2 = list(part3) + list(part2) + list(part1)
        elif b == 2:
            c1 = list(part2) + list(part1) + list(part3)
            c2 = list(part2) + list(part3) + list(part1)
        else:
            c1 = list(part1) + list(part3) + list(part2)
            c2 = list(part3) + list(part1) + list(part2)
        c1 = np.array(c1)
        c2 = np.array(c2)
        if np.all(c1 is False):
            c1[random.randint(0, len(c1) - 1)] = True
        if np.all(c2 is False):
            c2[random.randint(0, len(c2) - 1)] = True
        return c1, c2

    def mutate(self, pop):
        mutate_index = random.sample(range(0, len(pop[0])), random.randint(1, self.n_mut))
        for ind in pop:
            for x in mutate_index:
                ind[x] = not ind[x]
        return pop

    def write_res(self, folderName, y1, y2, bestScorePro, bestWeightPro, bestScore, bestWeight, bestInd,
                  debut, out, yTps):
        a = os.path.join(os.path.join(self.path2, folderName), 'resultat.txt')
        f = open(a, "w")
        string = "heuristique: Algorithme génétique" + os.linesep + \
                 "population: " + str(self.n_pop) + os.linesep + \
                 "générations: " + str(self.n_gen) + os.linesep + "mutations: " + str(self.n_mut) + os.linesep + \
                 "moyenne: " + str(y1) + os.linesep + "meilleur: " + str(y2) + os.linesep + \
                 "temps: " + str(yTps) + os.linesep + \
                 "scores:" + str(bestScorePro) + os.linesep + \
                 "poids:" + str(bestWeightPro) + os.linesep + \
                 "meilleur score: " + str(bestScore) + os.linesep + \
                 "meilleur poids: " + str(bestWeight) + os.linesep + \
                 "meilleur individu: " + str(bestInd) + os.linesep + \
                 "temps total: " + str(timedelta(seconds=(time.time() - debut))) + os.linesep +\
                 "mémoire: " + str(psutil.virtual_memory())
        f.write(string)
        f.close()
        a = os.path.join(os.path.join(self.path2, folderName), 'print.txt')
        f = open(a, "w")
        f.write(out)

    def selection(self, pop, i):
        try:
            return pop[i], pop[i + 1]
        except IndexError:
            pass

    def merge(self, pop, new):
        pop_list = []
        for ind, score, weight in pop:
            pop_list.append(list([list(ind), score, weight]))
        new_list = []
        for ind, score, weight in new:
            new_list.append(list([list(ind), score, weight]))
        newpop = []
        scores = []
        weights = []
        for i in range(len(pop_list)):
            newpop.append((pop_list[i][0]))
            scores.append((pop_list[i][1]))
            weights.append((pop_list[i][2]))
        for j in range(len(new_list)):
            newpop.append((new_list[j][0]))
            scores.append((new_list[j][1]))
            weights.append((new_list[j][2]))
        return np.array(newpop), scores, weights

    def natural_selection(self, part, besties, names, iters, times, names2):

        debut = time.time()
        print_out = ""

        for mode in part:

            np.random.seed(None)

            folderName = mode.upper()

            utility.createDirectory(path=self.path2, folderName=folderName)

            # Les axes pour le graphique
            x1 = []
            y1 = []
            y2 = []

            yTps = []

            scoreMax = 0
            weightMax = 0
            indMax = 0

            # Progression des meilleurs éléments
            bestScorePro = []
            bestWeightPro = []
            bestIndsPro = []

            # Mesurer le temps d'execution
            instant = time.time()

            # Initialise la population
            pop = utility.create_population_knapsack(inds=self.n_pop, size=self.n_objects)

            scores, weights, inds = utility.fitness_knapsack(self=self, pop=pop)

            bestScore, bestWeight, bestInd, bestScorePro, bestWeightPro, bestIndsPro = \
                utility.knapsack_add_list(scores=scores, weights=weights, inds=pop, bestScorePro=bestScorePro,
                                          bestWeightPro=bestWeightPro, bestIndsPro=bestIndsPro)

            generation = 0

            mean_scores = np.mean(heapq.nlargest(int(self.n_pop / 2), scores))
            time_instant = timedelta(seconds=(time.time() - instant))
            time_debut = timedelta(seconds=(time.time() - debut))

            x1, y1, y2, yTps = utility.add_axis(bestScore=bestScore, meanScore=mean_scores, iter=generation,
                                                time_debut=time_debut, x1=x1, y1=y1, y2=y2, yTps=yTps)

            print_out = utility.my_print_knapsack(print_out=print_out, mode=mode, mean=mean_scores,
                                                  bestScore=bestScore, bestWeight=bestWeight, time_exe=time_instant,
                                                  time_total=time_debut, iter=generation + 1)

            print_out = print_out + "\n"

            for generation in range(self.n_gen):

                instant = time.time()

                # Sélectionne les n_pop/2 meilleurs de la population
                i_val = heapq.nlargest(int(self.n_pop / 2), enumerate(scores), key=itemgetter(1))
                scores = [val for (i, val) in sorted(i_val)]
                indexes = [i for (i, val) in sorted(i_val)]
                pop = [pop[x] for x in indexes]
                weights = [weights[x] for x in indexes]

                # Création des enfants
                children = []
                for i in range(0, int(self.n_pop / 2), 2):
                    try:
                        p1, p2 = self.selection(pop, i)
                        c1, c2 = self.crossover(p1, p2)
                        children.append(c1)
                        children.append(c2)
                    except:
                        pass

                # Calcul du score pour l'ensemble des enfants
                scores_c, weights_c, inds_c = utility.fitness_knapsack(self=self, pop=children)

                pop_score = zip(pop, scores, weights)
                children_score = zip(children, scores_c, weights_c)

                pop, scores, weights = self.merge(pop_score, children_score)

                # Création des mutants
                tmp = np.copy(pop)
                mutants = self.mutate(tmp)

                # Calcul du score pour l'ensemble des mutants
                scores_m, weights_m, inds_m = utility.fitness_knapsack(self=self, pop=mutants)

                pop_score = zip(pop, scores, weights)
                mut_score = zip(mutants, scores_m, weights_m)

                pop, scores, weights = self.merge(pop_score, mut_score)

                bestScore, bestWeight, bestInd, bestScorePro, bestWeightPro, bestIndsPro = \
                    utility.knapsack_add_list(scores=scores, weights=weights, inds=pop, bestScorePro=bestScorePro,
                                              bestWeightPro=bestWeightPro, bestIndsPro=bestIndsPro)

                generation = generation + 1

                mean_scores = np.mean(heapq.nlargest(int(self.n_pop / 2), scores))
                time_instant = timedelta(seconds=(time.time() - instant))
                time_debut = timedelta(seconds=(time.time() - debut))

                print_out = utility.my_print_knapsack(print_out=print_out, mode=mode, mean=mean_scores,
                                                      bestScore=bestScore, bestWeight=bestWeight, time_exe=time_instant,
                                                      time_total=time_debut, iter=generation)

                print_out = print_out + "\n"

                x1, y1, y2, yTps = utility.add_axis(bestScore=bestScore, meanScore=mean_scores, iter=generation,
                                                    time_debut=time_debut, x1=x1, y1=y1, y2=y2, yTps=yTps)

                utility.plot_knapsack(x1=x1, y1=y1, y2=y2, yTps=yTps, n_pop=self.n_pop, n_gen=self.n_gen,
                                      heuristic="Algorithme génétique", folderName=folderName, path=self.path2,
                                      bestScore=bestScore, mean_scores=mean_scores,
                                      time_total=time_debut.total_seconds())

                if bestScore > scoreMax:
                    scoreMax = bestScore
                    weightMax = bestWeight
                    indMax = bestInd

                self.write_res(folderName=folderName, y1=y1, y2=y2, bestScorePro=bestScorePro,
                               bestWeightPro=bestWeightPro, bestScore=bestScore, bestWeight=weightMax, bestInd=indMax,
                               debut=debut, out=print_out, yTps=yTps)

            besties, names, iters, times, names2 = \
                utility.queues_put_knapsack(y2=y2, folderName=folderName, scoreMax=scoreMax, iter=generation, yTps=yTps,
                                            time=time_debut.total_seconds(), besties=besties, names=names,
                                            names2=names2, iters=iters, times=times)

    def init(self):

        print("######################")
        print("#ALGORITHME GENETIQUE#")
        print("######################")
        print()

        besties, names, iters, times, names2 = utility.queues_init()

        mods = self.list_exp

        n = len(mods)
        mods = [mods[i::n] for i in range(n)]

        processes = []

        for part in mods:
            process = multiprocessing.Process(target=self.natural_selection,
                                              args=(part, besties, names, iters, times, names2))
            processes.append(process)
            process.start()

        bestiesLst, namesLst, itersLst, timesLst, names2Lst =\
            utility.queues_get(n_process=len(processes), besties=besties, names=names, names2=names2, iters=iters,
                               times=times)

        for process in processes:
            process.join()

        return utility.res(heuristic="Algorithme génétique", besties=bestiesLst, names=namesLst, iters=itersLst,
                           times=timesLst, names2=names2Lst, path=self.path2, dataset=self.dataset)

if __name__ == '__main__':

    gen = Genetic(dataset='knapsack_test3', capacity=519570967,
                  list_exp=["EXP1", "EXP2", "EXP3", "EXP4", "EXP5", "EXP6", "EXP7", "EXP8", "EXP9", "EXP10"],
                  seed=42, pop=100, gen=1000, mut=5)


    gen.init()
