import utility.de as de
import utility.utility as utility

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

    def __init__(self, dataset, capacity, list_exp, seed, pop, gen):
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
        utility.cleanOut()

    def write_res(self, folderName, y1, y2, bestScorePro, bestWeightPro,
                  bestScore, bestWeight, bestInd, debut, out, yTps):
        a = os.path.join(os.path.join(self.path2, folderName), 'resultat.txt')
        f = open(a, "w")
        string = "heuristique: Evolution différentielle (CODE)" + os.linesep + \
                 "population: " + str(self.n_pop) + os.linesep + \
                 "générations: " + str(self.n_gen) + os.linesep + \
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

            # Choix de paramétrage
            params_pool = [[1.0, 0.1], [1.0, 0.9], [0.8, 0.2], [1.0, 0.5]]
            strats_pool = ['de_rand_1', 'de_rand_2', 'de_current_to_rand_1', 'de_best_1']
            n_pool = len(params_pool)

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
                                                  bestScore=bestScore, bestWeight=bestWeight,
                                                  time_exe=time_instant, time_total=time_debut, iter=generation)

            print_out = print_out + "\n"

            for generation in range(self.n_gen):

                instant = time.time()

                # Liste des mutants
                # mutants = []

                # Création des mutants
                for i in range(self.n_pop):
                    trials = []
                    for j in range(n_pool):
                        strat = strats_pool[j]
                        params = params_pool[random.randint(0, n_pool-1)]
                        mutant = de.mutate(F=params[0], pop=pop, bestInd=bestInd, ind_pos=i, strat=strat)
                        if j != 2:
                            trial = de.crossover(n_ind=self.n_objects, ind=pop[i], mutant=mutant, cross_proba=params[1])
                        else:
                            trial = mutant
                        trials.append(trial)
                    scores_t, weights_t, inds_t = utility.fitness_knapsack(self=self, pop=trials)
                    argmax_t = np.argmax(scores_t)
                    bestInd_t = inds_t[argmax_t]

                    # mutants.append(bestInd_t)

                    if scores[i] < scores_t[argmax_t]:
                        pop[i] = bestInd_t
                        scores[i] = scores_t[argmax_t]
                        weights[i] = weights_t[argmax_t]

                        bestScore, bestWeight, bestInd, bestScorePro, bestWeightPro, bestIndsPro = \
                            utility.knapsack_add_list(scores=scores, weights=weights, inds=pop,
                                                      bestScorePro=bestScorePro, bestWeightPro=bestWeightPro,
                                                      bestIndsPro=bestIndsPro)

                '''
                # Calcul du score pour l'ensemble des mutants
                scores_m, weights_m, inds_m = \
                    utility.fitness_knapsack(self=self, pop=mutants)

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
                                                      time_total=time_debut, iter=generation)

                print_out = print_out + "\n"

                x1, y1, y2, yTps = utility.add_axis(bestScore=bestScore, meanScore=mean_scores, iter=generation,
                                                    time_debut=time_debut, x1=x1, y1=y1, y2=y2, yTps=yTps)

                utility.plot_knapsack(x1=x1, y1=y1, y2=y2, yTps=yTps, n_pop=self.n_pop, n_gen=self.n_gen,
                                      heuristic="Evolution différentielle (CODE)", folderName=folderName,
                                      path=self.path2, bestScore=bestScore, mean_scores=mean_scores,
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

        print("##############################################")
        print("#ALGORITHME A EVOLUTION DIFFERENTIELLE (CODE)#")
        print("##############################################")
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

        return utility.res(heuristic="Evolution différentielle (CODE)",
                           besties=bestiesLst, names=namesLst, iters=itersLst,
                           times=timesLst, names2=names2Lst, path=self.path2, dataset=self.dataset)


if __name__ == '__main__':

    diff = Differential(dataset='knapsack_test3', capacity=519570967,
                        list_exp=["EXP1", "EXP2", "EXP3", "EXP4", "EXP5", "EXP6", "EXP7", "EXP8", "EXP9", "EXP10"],
                        seed=42, pop=100, gen=1000)

    diff.init()
