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


class Pbil:

    def __init__(self, n_objects, capacity, seed):
        self.n_objects = n_objects
        self.capacity = capacity
        self.seed = seed
        random.seed(seed)
        self.values = [random.choice(range(1, 100)) for i in range(n_objects)]
        self.weights = [random.choice(range(1, 50)) for i in range(n_objects)]
        while sum(self.weights) <= capacity:
            random.seed(seed+1)
            self.weights = [random.choice(range(1, 50)) for i in range(n_objects)]
        self.warmstart = np.random.rand(n_objects) <= 0.5
        total_weight = capacity
        while total_weight >= capacity:
            random.seed(seed+1)
            self.warmstart = np.random.rand(n_objects) <= 0.1
            total_weight = 0
            for i in range(self.n_objects):
                if self.warmstart[i]:
                    total_weight = total_weight + self.weights[i]
        random.seed()
        self.path2 = os.path.dirname(os.getcwd()) + '/out'

    def create_population(self, inds, size, probas):
        pop = np.zeros((inds, size), dtype=bool)
        for i in range(inds):
            pop[i] = np.random.rand(size) <= probas
        return pop

    def create_proba(self, size):
        return np.repeat(0.5, size)

    def update_proba(self, maxi, probas, learningRate):
        for i in range(len(probas)):
            probas[i] = probas[i]*(1.0-learningRate)+maxi[i]*learningRate
        return probas

    def mutate_proba(self, probas, mutProba, mutShift):
        for i in range(len(probas)):
            if random.uniform(0, 1) < mutProba:
                probas[i] = probas[i]*(1.0-mutShift)+random.choice([0, 1])*mutShift
        return probas

    def crossover(self, ind, mutant, cross_proba):
        cross_points = np.random.rand(self.n_objects) <= cross_proba

        trial = np.where(cross_points, mutant, ind)

        idxs = [idx for idx in range(len(ind))]
        selected = np.random.choice(idxs, 1, replace=False)

        trial[selected] = not trial[selected]

        return trial

    def mutate(self, F, pop, bestInd, ind_pos, probas):

        idxs = [idx for idx in range(len(pop)) if idx != ind_pos]
        selected = np.random.choice(idxs, 2, replace=False)
        xr1, xr2 = pop[selected]

        xr3 = np.random.rand(len(probas)) <= probas

        mutant = bestInd.astype(np.float32) + F * (xr2.astype(np.float32) - xr3.astype(np.float32))

        for i in range(self.n_objects):
            if mutant[i] < 0.5:
                mutant[i] = 0
            else:
                mutant[i] = 1

        mutant = mutant.astype(bool)

        return mutant

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

    def write_res(self, folderName, n_pop, n_gen, cross_proba, F, strat,
                  learning_rate, mut_proba, mut_shift, probas,
                  y1, y2, bestScorePro, bestWeightPro, bestScore, bestWeight, debut, out, yTps):
        a = os.path.join(os.path.join(self.path2, folderName), 'resultat.txt')
        f = open(a, "w")
        string = "heuristique: Apprentissage incrémental à base de population différentiel" + os.linesep + \
                 "population: " + str(n_pop) + os.linesep + \
                 "générations: " + str(n_gen) + os.linesep + \
                 "probabilité de croisement: " + str(cross_proba) + os.linesep + "F: " + str(F) + os.linesep + \
                 "stratégie de mutation: " + str(strat) + os.linesep + \
                 "taux d'apprentissage: " + str(learning_rate) + os.linesep + \
                 "probabilité de mutation: " + str(mut_proba) + os.linesep + \
                 "magnitude de mutation: " + str(mut_shift) + os.linesep + \
                 "vecteur de probabilité final: " + str(probas) + os.linesep + \
                 "moyenne: " + str(y1) + os.linesep + "meilleur: " + str(y2) + os.linesep + \
                 "temps: " + str(yTps) + os.linesep + \
                 "scores:" + str(bestScorePro) + \
                 "poids:" + str(bestWeightPro) + \
                 "meilleur score: " + str(bestScore) + os.linesep + \
                 "meilleur poids: " + str(bestWeight) + os.linesep + "temps total: " + \
                 str(timedelta(seconds=(time.time() - debut))) + os.linesep + "mémoire: " + \
                 str(psutil.virtual_memory())
        f.write(string)
        f.close()
        a = os.path.join(os.path.join(self.path2, folderName), 'print.txt')
        f = open(a, "w")
        f.write(out)

    def natural_selection(self, part, n_pop, n_gen, cross_proba, F, strat, learning_rate, mut_proba, mut_shift,
                          besties, names, iters, times, names2):

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
            probas = self.create_proba(size=self.n_objects)

            # Initialise la population
            pop = utility.create_population(inds=n_pop, size=self.n_objects)
            pop[n_pop - 1] = self.warmstart

            scores, weights, inds, obj = utility.fitness(self=self, pop=pop)

            bestScore = np.max(scores)
            argmax = np.argmax(scores)
            bestWeight = weights[argmax]
            bestInd = inds[argmax]

            bestScorePro.append(bestScore)
            bestWeightPro.append(bestWeight)
            bestIndsPro.append(bestInd)

            # Met à jour le vecteur de probabilité
            probas = self.update_proba(bestInd, probas, learning_rate)

            # Mutation sur le vecteur de probabilité
            probas = self.mutate_proba(probas, mut_proba, mut_shift)

            x1.append(0)
            y1.append(np.mean(heapq.nlargest(int(n_pop / 2), scores)))
            y2.append(bestScore)

            time_debut = timedelta(seconds=(time.time() - debut))
            yTps.append(time_debut.total_seconds())

            generation = 0

            utility.my_print(print_out=print_out, mode=mode, mean=np.mean(heapq.nlargest(int(n_pop / 2), scores)),
                             bestScore=bestScore, bestWeight=bestWeight,
                             time_exe=timedelta(seconds=(time.time() - instant)), time_total=time_debut,
                             iter=generation)

            for generation in range(n_gen):

                instant = time.time()

                # Liste des mutants
                mutants = []

                n_pbil = int(n_pop/3)
                n_diff = n_pop - n_pbil

                # Les mutants de l'évolution différentielle
                for i in range(n_pop):

                    # mutation
                    mutant = self.mutate(F, pop, bestInd, i, probas)

                    # croisement
                    trial = self.crossover(pop[i], mutant, cross_proba)

                    mutants.append(trial)

                '''
                # Les mutants du pbil
                mut_pbil = self.create_population(inds=n_pbil, size=self.n_objects, probas=probas)

                for j in range(n_pbil):

                    trial = self.crossover(pop[n_diff+j], mut_pbil[j], cross_proba)

                    mutants.append(trial)
                '''

                # Calcul du score pour l'ensemble des mutants
                scores_m, weights_m, inds_m, obj = \
                    utility.fitness(self=self, pop=mutants)

                pop_score = zip(pop, scores, weights)
                mut_score = zip(mutants, scores_m, weights_m)

                pop, scores, weights = self.merge(pop_score, mut_score)

                bestScore = np.max(scores)
                argmax = np.argmax(scores)
                bestWeight = weights[argmax]
                bestInd = pop[argmax]

                bestScorePro.append(bestScore)
                bestWeightPro.append(bestWeight)
                bestIndsPro.append(bestInd)

                probas = self.update_proba(bestInd, probas, learning_rate)

                probas = self.mutate_proba(probas, mut_proba, mut_shift)

                x1.append(generation + 1)

                mean_scores = np.mean(heapq.nlargest(int(n_pop / 2), scores))
                time_instant = timedelta(seconds=(time.time() - instant))
                time_debut = timedelta(seconds=(time.time() - debut))
                yTps.append(time_debut.total_seconds())

                print_out = utility.my_print(print_out=print_out, mode=mode,
                                             mean=np.mean(heapq.nlargest(int(n_pop / 2), scores)),
                                             bestScore=bestScore, bestWeight=bestWeight,
                                             time_exe=time_instant, time_total=time_debut,
                                             iter=generation+1)

                print_out = print_out + "\n"

                # La moyenne sur les n_pop/2 premiers de la population
                y1.append(np.mean(heapq.nlargest(int(n_pop / 2), scores)))
                y2.append(bestScore)

                utility.plot(x1=x1, y1=y1, y2=y2, yTps=yTps, n_pop=n_pop, n_gen=n_gen,
                             heuristic="Apprentissage incrémental à base de population différentiel",
                             folderName=folderName, path=self.path2,
                             bestScore=bestScore, mean_scores=mean_scores, time_total=time_debut.total_seconds())

                generation = generation + 1

                if bestScore > scoreMax:
                    scoreMax = bestScore
                    weightMax = bestWeight
                    indMax = bestInd

                self.write_res(folderName=folderName, n_pop=n_pop, n_gen=n_gen, cross_proba=cross_proba, F=F,
                               strat=strat,
                               learning_rate=learning_rate, mut_proba=mut_proba, mut_shift=mut_shift, probas=probas,
                               y1=y1, y2=y2, bestScorePro=bestScorePro,
                               bestWeightPro=bestWeightPro, bestScore=bestScore, bestWeight=bestWeight, debut=debut,
                               out=print_out, yTps=yTps)

            besties.put(y2)
            names.put(folderName + ": " + "{:.0f}".format(scoreMax))
            iters.put(generation)
            times.put(yTps)
            names2.put(folderName + ": " + "{:.0f}".format(time_debut.total_seconds()))

    def init(self, n_pop, n_gen, cross_proba, F, strat, learning_rate, mut_proba, mut_shift):

        print("#############################################################")
        print("#APPRENTISSAGE INCREMENTAL A BASE DE POPULATION DIFFERENTIEL#")
        print("#############################################################")
        print()

        besties = multiprocessing.Queue()
        names = multiprocessing.Queue()
        iters = multiprocessing.Queue()
        times = multiprocessing.Queue()
        names2 = multiprocessing.Queue()

        mods = ["EXP1", "EXP2", "EXP3", "EXP4", "EXP5", "EXP6", "EXP7", "EXP8", "EXP9", "EXP10"]

        n = len(mods)
        mods = [mods[i::n] for i in range(n)]

        processes = []

        bestiesLst = []
        namesLst = []
        itersLst = []
        timesLst = []
        names2Lst = []

        for part in mods:
            process = multiprocessing.Process(target=self.natural_selection,
                                              args=(part, n_pop, n_gen, cross_proba, F, strat,
                                                    learning_rate, mut_proba, mut_shift, besties,
                                                    names, iters, times, names2))
            processes.append(process)
            process.start()

        for i in range(len(processes)):
            bestiesLst.append(besties.get())
            namesLst.append(names.get())
            names2Lst.append(names2.get())
            itersLst.append(iters.get())
            timesLst.append(times.get())

        for process in processes:
            process.join()

        return utility.res(heuristic="Apprentissage incrémental à base de population différentielle",
                           besties=bestiesLst, names=namesLst, iters=itersLst,
                           times=timesLst, names2=names2Lst, path=self.path2,
                           n_gen=n_gen, self=self)


if __name__ == '__main__':

    genetic = Pbil(n_objects=300, capacity=1000, seed=42)
    # pop = 30
    # gen = 1000
    pop = 30
    gen = 1000
    learning_rate = 0.1
    mut_proba = 0.2
    mut_shift = 0.05
    cross_proba = 0.5
    F = 2.0
    strat = "de_best_1"

    genetic.init(n_pop=pop, n_gen=gen, cross_proba=cross_proba, F=F, strat=strat,
                 learning_rate=learning_rate, mut_proba=mut_proba, mut_shift=mut_shift)
