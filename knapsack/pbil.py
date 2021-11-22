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


class Pbil:

    def __init__(self, dataset, capacity, list_exp, seed, pop, gen, learning_rate, mut_proba, mut_shift):
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
        self.learning_rate = learning_rate
        self.mut_proba = mut_proba
        self.mut_shift = mut_shift
        utility.cleanOut()

    def create_population(self, probas):
        pop = np.zeros((self.n_pop, self.n_objects), dtype=bool)
        for i in range(self.n_pop):
            pop[i] = np.random.rand(self.n_objects) <= probas
        return pop

    def create_proba(self):
        return np.repeat(0.5, self.n_objects)

    def update_proba(self, maxi, probas):
        for i in range(len(probas)):
            probas[i] = probas[i]*(1.0-self.learning_rate)+maxi[i]*self.learning_rate
        return probas

    def mutate_proba(self, probas):
        for i in range(len(probas)):
            if random.uniform(0, 1) < self.mut_proba:
                probas[i] = probas[i]*(1.0-self.mut_shift)+random.choice([0, 1])*self.mut_shift
        return probas

    def write_res(self, folderName, probas, y1, y2, bestScorePro, bestWeightPro, bestScore, bestWeight, bestInd,
                  debut, out, yTps):
        a = os.path.join(os.path.join(self.path2, folderName), 'resultat.txt')
        f = open(a, "w")
        string = "heuristique: Apprentissage incrémental à base de population" + os.linesep + \
                 "population: " + str(self.n_pop) + os.linesep + \
                 "générations: " + str(self.n_gen) + os.linesep + \
                 "taux d'apprentissage: " + str(self.learning_rate) + os.linesep + \
                 "probabilité de mutation: " + str(self.mut_proba) + os.linesep + \
                 "magnitude de mutation: " + str(self.mut_shift) + os.linesep + \
                 "vecteur de probabilité final: " + str(probas) + os.linesep + \
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

            # Initialise le vecteur de probabilité
            probas = self.create_proba()

            # Initialise la population
            pop = self.create_population(probas=probas)

            scores, weights, inds = utility.fitness_knapsack(self=self, pop=pop)

            bestScore, bestWeight, bestInd, bestScorePro, bestWeightPro, bestIndsPro = \
                utility.knapsack_add_list(scores=scores, weights=weights, inds=pop, bestScorePro=bestScorePro,
                                          bestWeightPro=bestWeightPro, bestIndsPro=bestIndsPro)

            generation = 0

            mean_scores = np.mean(heapq.nlargest(int(self.n_pop / 2), scores))
            time_instant = timedelta(seconds=(time.time() - instant))
            time_debut = timedelta(seconds=(time.time() - debut))

            x1, y1, y2, yTps = utility.add_axis_max(maxScore=bestScore, meanScore=mean_scores, iter=generation,
                                                    time_debut=time_debut, x1=x1, y1=y1, y2=y2, yTps=yTps)

            print_out = utility.my_print_knapsack(print_out=print_out, mode=mode, mean=mean_scores,
                                                  bestScore=bestScore, bestWeight=bestWeight, time_exe=time_instant,
                                                  time_total=time_debut, iter=generation + 1)

            print_out = print_out + "\n"

            # Met à jour le vecteur de probabilité
            probas = self.update_proba(maxi=bestInd, probas=probas)

            # Mutation sur le vecteur de probabilité
            probas = self.mutate_proba(probas=probas)

            for generation in range(self.n_gen):

                instant = time.time()

                pop = self.create_population(probas=probas)

                scores, weights, inds = utility.fitness_knapsack(self=self, pop=pop)

                bestScore, bestWeight, bestInd, bestScorePro, bestWeightPro, bestIndsPro = \
                    utility.knapsack_add_list(scores=scores, weights=weights, inds=pop, bestScorePro=bestScorePro,
                                              bestWeightPro=bestWeightPro, bestIndsPro=bestIndsPro)

                generation = generation + 1

                mean_scores = np.mean(heapq.nlargest(int(self.n_pop / 2), scores))
                time_instant = timedelta(seconds=(time.time() - instant))
                time_debut = timedelta(seconds=(time.time() - debut))

                print_out = utility.my_print_knapsack(print_out=print_out, mode=mode, mean=mean_scores,
                                                      bestScore=bestScore, bestWeight=bestWeight, time_exe=time_instant,
                                                      time_total=time_debut, iter=generation+1)

                print_out = print_out + "\n"

                if generation == 1:
                    x1, y1, y2, yTps = utility.add_axis_max(maxScore=bestScore, meanScore=mean_scores, iter=generation,
                                                            time_debut=time_debut, x1=x1, y1=y1, y2=y2, yTps=yTps)
                else:
                    x1, y1, y2, yTps = utility.add_axis_max(maxScore=scoreMax, meanScore=mean_scores, iter=generation,
                                                            time_debut=time_debut, x1=x1, y1=y1, y2=y2, yTps=yTps)

                utility.plot_knapsack(x1=x1, y1=y1, y2=y2, yTps=yTps, n_pop=self.n_pop, n_gen=self.n_gen,
                                      heuristic="Apprentissage incrémental à base de population", folderName=folderName,
                                      path=self.path2, bestScore=bestScore, mean_scores=mean_scores,
                                      time_total=time_debut.total_seconds())

                probas = self.update_proba(maxi=bestInd, probas=probas)

                probas = self.mutate_proba(probas=probas)

                if bestScore > scoreMax:
                    scoreMax = bestScore
                    weightMax = bestWeight
                    indMax = bestInd

                self.write_res(folderName=folderName, probas=probas, y1=y1, y2=y2, bestScorePro=bestScorePro,
                               bestWeightPro=bestWeightPro, bestScore=bestScore, bestWeight=weightMax, bestInd=indMax,
                               debut=debut, out=print_out, yTps=yTps)

            besties, names, iters, times, names2 = \
                utility.queues_put_knapsack(y2=y2, folderName=folderName, scoreMax=scoreMax, iter=generation, yTps=yTps,
                                            time=time_debut.total_seconds(), besties=besties, names=names,
                                            names2=names2, iters=iters, times=times)

    def init(self):

        print("################################################")
        print("#APPRENTISSAGE INCREMENTAL A BASE DE POPULATION#")
        print("################################################")
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

        return utility.res(heuristic="Apprentissage incrémental à base de population",
                           besties=bestiesLst, names=namesLst, iters=itersLst,
                           times=timesLst, names2=names2Lst, path=self.path2,
                           n_gen=self.n_gen, self=self)


if __name__ == '__main__':

    pbil = Pbil(dataset='knapsack_test3', capacity=519570967,
                list_exp=["EXP1", "EXP2", "EXP3", "EXP4", "EXP5", "EXP6", "EXP7", "EXP8", "EXP9", "EXP10"],
                seed=42, pop=100, gen=1000, learning_rate=0.1, mut_proba=0.2, mut_shift=0.05)

    pbil.init()

