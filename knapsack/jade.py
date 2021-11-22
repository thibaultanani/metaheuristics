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
from scipy import stats

import warnings

warnings.filterwarnings('ignore')

np.set_printoptions(threshold=sys.maxsize)


class Differential:

    def __init__(self, dataset, capacity, list_exp, seed, pop, gen, c, p):
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
        self.c = c
        self.p = p
        utility.cleanOut()

    @staticmethod
    def update_param(muCR, muF, c, SCR, SF):
        try:
            muCR = (1 - c) * muCR + c * (sum(SCR)/len(SCR))
        except ZeroDivisionError:
            pass
        try:
            muF = (1 - c) * muF + c * (sum([F**2 for F in SF])/sum(SF))
        except ZeroDivisionError:
            pass
        return muCR, muF

    def write_res(self, folderName, y1, y2, bestScorePro, bestWeightPro, bestScore, bestWeight, bestInd, debut, out,
                  yTps):
        a = os.path.join(os.path.join(self.path2, folderName), 'resultat.txt')
        f = open(a, "w")
        string = "heuristique: Evolution différentielle (JADE)" + os.linesep + \
                 "population: " + str(self.n_pop) + os.linesep + \
                 "générations: " + str(self.n_gen) + os.linesep + \
                 "c: " + str(self.c) + os.linesep + \
                 "p: " + str(self.p) + os.linesep + \
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

    def merge(self, pop, mutants, archive, cross_probas, F_probas):
        pop_list = []
        for ind, score, weight in pop:
            pop_list.append(list([list(ind), score, weight]))
        mut_list = []
        for ind, score, weight, cross_proba, F in mutants:
            mut_list.append(list([list(ind), score, weight, cross_proba, F]))
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
                cross_probas.append((mut_list[i][3]))
                F_probas.append((mut_list[i][4]))
                archive.append((pop_list[i][0]))

        while len(archive) > self.n_objects:
            archive.pop(random.randint(0, self.n_objects-1))

        return np.array(newpop), scores, weights, archive, cross_probas, F_probas

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

            # Initalise les paramètres
            muCR = 0.5
            muF = 0.5

            # Initialiser l'archive
            archive = []

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

            pbest = int(self.n_pop * self.p)

            for generation in range(self.n_gen):

                instant = time.time()

                # Liste des bons croisements et du facteur F
                cross_probas = []
                F_probas = []

                # Liste des mutants
                # mutants = []
                # cross_probas_m = []
                # F_probas_m = []

                # Les indices des individus avec les scores les plus élevés
                indices = (-np.array(scores)).argsort()[:pbest]

                try:
                    pop_archive = np.vstack((pop, archive))
                except ValueError:
                    pop_archive = pop

                # Création des mutants
                for i in range(self.n_pop):

                    cross_proba = -1
                    while cross_proba > 1 or cross_proba < 0:
                        cross_proba = stats.norm.rvs(loc=muCR, scale=0.1)

                    F = -1
                    while F < 0:
                        F = stats.cauchy.rvs(loc=muF, scale=0.1)
                        if F > 1:
                            F = 1

                    pInd = pop[indices[random.randint(0, len(indices)-1)]]

                    # mutation
                    mutant = de.mutate_jade(n_ind=self.n_objects, F=F, pop=pop, pInd=pInd, ind_pos=i,
                                            pop_archive=pop_archive)

                    # croisement
                    trial = de.crossover(n_ind=self.n_objects, ind=pop[i], mutant=mutant, cross_proba=cross_proba)

                    score_m, weight_m = utility.fitness_ind_knapsack(self=self, ind=trial)
                    if scores[i] < score_m:
                        archive.append((pop[i]))

                        pop[i] = trial
                        scores[i] = score_m
                        weights[i] = weight_m

                        cross_probas.append(cross_proba)
                        F_probas.append(F)

                        bestScore, bestWeight, bestInd, bestScorePro, bestWeightPro, bestIndsPro = \
                            utility.knapsack_add_list(scores=scores, weights=weights, inds=pop,
                                                      bestScorePro=bestScorePro, bestWeightPro=bestWeightPro,
                                                      bestIndsPro=bestIndsPro)
                    # mutants.append(trial)

                    # cross_probas_m.append(cross_proba)
                    # F_probas_m.append(F)

                '''
                # Calcul du score pour l'ensemble des mutants
                scores_m, weights_m, inds_m = utility.fitness_knapsack(self=self, pop=mutants)

                pop_score = zip(pop, scores, weights)
                mut_score = zip(mutants, scores_m, weights_m, cross_probas_m, F_probas_m)

                pop, scores, weights, archive, SCR, SF = \
                    self.merge(pop=pop_score, mutants=mut_score, archive=archive, cross_probas=cross_probas,
                               F_probas=F_probas)
                '''

                # Enlever les doublons de nos listes
                cross_probas = utility.f7(cross_probas)
                F_probas = utility.f7(F_probas)

                while len(archive) > self.n_pop:
                    archive.pop(random.randint(0, self.n_pop - 1))

                muCR, muF = self.update_param(muCR=muCR, muF=muF, c=self.c, SCR=cross_probas, SF=F_probas)

                '''
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
                                      heuristic="Evolution différentielle (JADE)", folderName=folderName,
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
        print("#ALGORITHME A EVOLUTION DIFFERENTIELLE (JADE)#")
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

        return utility.res(heuristic="Evolution différentielle (JADE)",
                           besties=bestiesLst, names=namesLst, iters=itersLst,
                           times=timesLst, names2=names2Lst, path=self.path2,
                           dataset=self.dataset)


if __name__ == '__main__':

    genetic = Differential(dataset='knapsack_test2', capacity=9906309440,
                           list_exp=["EXP1"],
                           seed=42, pop=100, gen=1000, c=0.1, p=0.05)

    genetic.init()
