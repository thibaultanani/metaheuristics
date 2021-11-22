import utility.de as de
import utility.utility as utility
import utility.strategy as strategy

import multiprocessing
import numpy as np
import sys
import random
import os
import heapq
import psutil
import time
from datetime import timedelta

import warnings

warnings.filterwarnings('ignore')

np.set_printoptions(threshold=sys.maxsize)


class Differential:

    def __init__(self, dataset, capacity, list_exp, seed, pop, gen, t1, t2, Fl, Fu):
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
        self.t1 = t1
        self.t2 = t2
        self.Fl = Fl
        self.Fu = Fu
        utility.cleanOut()

    def crossover(self, ind, mutant, cross_proba):
        cross_points = np.random.rand(self.n_objects) <= cross_proba

        trial = np.where(cross_points, mutant, ind)

        idxs = [idx for idx in range(len(ind))]
        selected = np.random.choice(idxs, 1, replace=False)

        trial[selected] = not trial[selected]

        return trial

    def mutate(self, F, pop, bestInd, ind_pos, strat):
        try:
            mut_strategy = eval("strategy." + strat)
        except:
            mut_strategy = eval("strategy.de_rand_1")

        mutant = mut_strategy(F, pop, bestInd, ind_pos)

        mutant = mutant.astype(bool)

        return mutant

    def write_res(self, folderName, y1, y2, bestScorePro, bestWeightPro, bestScore, bestWeight, bestInd,
                  debut, out, yTps):
        a = os.path.join(os.path.join(self.path2, folderName), 'resultat.txt')
        f = open(a, "w")
        string = "heuristique: Evolution différentielle (JDE)" + os.linesep + \
                 "population: " + str(self.n_pop) + os.linesep + \
                 "générations: " + str(self.n_gen) + os.linesep + \
                 "t1: " + str(self.t1) + os.linesep + \
                 "t2: " + str(self.t2) + os.linesep + \
                 "Fl: " + str(self.Fl) + os.linesep + \
                 "Fu: " + str(self.Fu) + os.linesep + \
                 "moyenne: " + str(y1) + os.linesep + "meilleur: " + str(y2) + os.linesep + \
                 "temps: " + str(yTps) + os.linesep + \
                 "scores:" + str(bestScorePro) + os.linesep + \
                 "poids:" + str(bestWeightPro) + os.linesep + \
                 "meilleur score: " + str(bestScore) + os.linesep + \
                 "meilleur poids: " + str(bestWeight) + os.linesep + \
                 "meilleur individu: " + str(bestInd) + os.linesep + \
                 "temps total: " + str(timedelta(seconds=(time.time() - debut))) + os.linesep + \
                 "mémoire: " + str(psutil.virtual_memory())
        f.write(string)
        f.close()
        a = os.path.join(os.path.join(self.path2, folderName), 'print.txt')
        f = open(a, "w")
        f.write(out)

    def merge(self, pop, mutants):
        pop_list = []
        for ind, score, weight in pop:
            pop_list.append(list([list(ind), score, weight]))
        mut_list = []
        for ind, score, weight in mutants:
            mut_list.append(list([list(ind), score, weight]))
        newpop = []
        scores = []
        weights = []
        for i in range(len(pop_list)):
            if pop_list[i][1] > mut_list[i][1]:
                newpop.append((pop_list[i][0]))
                scores.append((pop_list[i][1]))
                weights.append((pop_list[i][2]))
            else:
                newpop.append((mut_list[i][0]))
                scores.append((mut_list[i][1]))
                weights.append((mut_list[i][2]))
        return np.array(newpop), scores, weights

    def differential_evolution(self, part, besties, names, iters, times, names2):

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

            # Initialise les paramètres
            F = [0.5]*self.n_pop
            CR = [0.5]*self.n_pop

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

                # Liste des mutants
                # mutants = []

                # Création des mutants
                for i in range(self.n_pop):

                    r1 = random.uniform(0, 1)
                    r2 = random.uniform(0, 1)
                    r3 = random.uniform(0, 1)
                    r4 = random.uniform(0, 1)

                    if r2 < self.t1:
                        F[i] = self.Fl + r1 * self.Fu

                    if r4 < self.t2:
                        CR[i] = r3

                    # mutation
                    mutant = self.mutate(F[i], pop, bestInd, i, 'de_best_1')

                    # croisement
                    trial = self.crossover(pop[i], mutant, CR[i])

                    score_m, weight_m = utility.fitness_ind_knapsack(self=self, ind=trial)
                    if scores[i] < score_m:
                        pop[i] = trial
                        scores[i] = score_m
                        weights[i] = weight_m

                        bestScore, bestWeight, bestInd, bestScorePro, bestWeightPro, bestIndsPro = \
                            utility.knapsack_add_list(scores=scores, weights=weights, inds=pop,
                                                      bestScorePro=bestScorePro, bestWeightPro=bestWeightPro,
                                                      bestIndsPro=bestIndsPro)

                    # mutants.append(trial)

                '''
                # Calcul du score pour l'ensemble des mutants
                scores_m, weights_m, inds_m = utility.fitness_knapsack(self=self, pop=mutants)

                pop_score = zip(pop, scores, weights)
                mut_score = zip(mutants, scores_m, weights_m)

                pop, scores, weights = de.merge_knapsack(pop=pop_score, mutants=mut_score)

                bestScore, bestWeight, bestInd, bestScorePro, bestWeightPro, bestIndsPro = \
                    utility.knapsack_add_list(scores=scores, weights=weights, inds=pop, bestScorePro=bestScorePro,
                                              bestWeightPro=bestWeightPro, bestIndsPro=bestIndsPro)
                '''

                generation = generation + 1

                mean_scores = np.mean(heapq.nlargest(int(self.n_pop / 2), scores))
                time_instant = timedelta(seconds=(time.time() - instant))
                time_debut = timedelta(seconds=(time.time() - debut))

                print_out = utility.my_print_knapsack(print_out=print_out, mode=mode, mean=mean_scores,
                                                      bestScore=bestScore, bestWeight=bestWeight, time_exe=time_instant,
                                                      time_total=time_debut, iter=generation+1)

                print_out = print_out + "\n"

                x1, y1, y2, yTps = utility.add_axis(bestScore=bestScore, meanScore=mean_scores, iter=generation,
                                                    time_debut=time_debut, x1=x1, y1=y1, y2=y2, yTps=yTps)

                utility.plot_knapsack(x1=x1, y1=y1, y2=y2, yTps=yTps, n_pop=self.n_pop, n_gen=self.n_gen,
                                      heuristic="Evolution différentielle (JDE)", folderName=folderName,
                                      path=self.path2, bestScore=bestScore, mean_scores=mean_scores,
                                      time_total=time_debut.total_seconds())

                generation = generation + 1

                if bestScore > scoreMax:
                    scoreMax = bestScore
                    weightMax = bestWeight
                    indMax = bestInd

                self.write_res(folderName=folderName, y1=y1, y2=y2, bestScorePro=bestScorePro,
                               bestWeightPro=bestWeightPro, bestScore=bestScore, bestWeight=weightMax, bestInd=indMax,
                               debut=debut, out=print_out, yTps=yTps)

            besties.put(y2)
            names.put(folderName + ": " + "{:.0f}".format(scoreMax))
            iters.put(generation)
            times.put(yTps)
            names2.put(folderName + ": " + "{:.0f}".format(time_debut.total_seconds()))

    def init(self):

        print("#############################################")
        print("#ALGORITHME A EVOLUTION DIFFERENTIELLE (JDE)#")
        print("#############################################")
        print()

        besties, names, iters, times, names2 = utility.queues_init()

        mods = self.list_exp

        n = len(mods)
        mods = [mods[i::n] for i in range(n)]

        processes = []

        for part in mods:
            process = multiprocessing.Process(target=self.differential_evolution,
                                              args=(part, besties, names, iters, times, names2))
            processes.append(process)
            process.start()

        bestiesLst, namesLst, itersLst, timesLst, names2Lst =\
            utility.queues_get(n_process=len(processes), besties=besties, names=names, names2=names2, iters=iters,
                               times=times)

        for process in processes:
            process.join()

        return utility.res(heuristic="Evolution différentielle (JDE)",
                           besties=bestiesLst, names=namesLst, iters=itersLst,
                           times=timesLst, names2=names2Lst, path=self.path2,
                           dataset=self.dataset)


if __name__ == '__main__':

    diff = Differential(dataset='knapsack_test3', capacity=519570967,
                        list_exp=["EXP1", "EXP2", "EXP3", "EXP4", "EXP5", "EXP6", "EXP7", "EXP8", "EXP9", "EXP10"],
                        seed=42, pop=100, gen=1000, t1=0.1, t2=0.1, Fl=0.1, Fu=0.9)

    diff.init()
