import utility.utility as utility

import multiprocessing
import numpy as np
import sys
import os
import heapq
import psutil
import time
import random
import math
from datetime import timedelta

import warnings

warnings.filterwarnings('ignore')

np.set_printoptions(threshold=sys.maxsize)


class Pbil:
    """
    Parameters
    ----------
    list_exp: [list: string] List of experiments running in parallel
    pop: [int] Number of solutions evaluated per generation
    gen: [int] Number of generations/iterations for the algorithm
    learning_rate: [float (0.0:1.0)] Speed at which the crossover probability is going to converge toward 0.0 or 1.0
    mut_proba: [float (0.0:1.0)] Probability to do a mutation
    mut_shift: [float (0.0:1.0)] Impact of the mutation on the probability vectors
    """
    def __init__(self, list_exp, pop, gen, learning_rate, mut_proba, mut_shift):

        self.heuristic = "PBIL"
        self.list_exp = list_exp
        self.n_pop = pop
        self.n_gen = gen
        self.learning_rate = learning_rate
        self.mut_proba = mut_proba
        self.mut_shift = mut_shift
        self.path1 = os.path.dirname(os.getcwd()) + '/in/'
        self.path2 = os.path.dirname(os.getcwd()) + '/out/' + self.heuristic + '/conjecture'
        self.vertices = 19
        self.n_ind = int(self.vertices*(self.vertices-1)/2)
        utility.cleanOut(path=self.path2)

    def create_population(self, probas):
        pop = np.zeros((self.n_pop, self.n_ind), dtype=bool)
        for i in range(self.n_pop):
            pop[i] = np.random.rand(self.n_ind) <= probas
        return pop

    def create_proba(self):
        return np.repeat(0.5, self.n_ind)

    def update_proba(self, maxi, probas):
        for i in range(len(probas)):
            probas[i] = probas[i]*(1.0-self.learning_rate)+maxi[i]*self.learning_rate
        return probas

    def mutate_proba(self, probas):
        for i in range(len(probas)):
            if random.uniform(0, 1) < self.mut_proba:
                probas[i] = probas[i]*(1.0-self.mut_shift)+random.choice([0, 1])*self.mut_shift
        return probas

    def write_res(self, folderName, probas, y1, y2, colMax, bestScorePro, bestScore, bestInd, debut, out, yTps, yEdges):
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
                 "nombre d'arêtes: " + str(yEdges) + os.linesep + \
                 "scores:" + str(bestScorePro) + os.linesep + \
                 "meilleur score: " + str(bestScore) + os.linesep + \
                 "meilleur individu: " + str(bestInd) + os.linesep + \
                 "Arêtes:" + str(colMax) + os.linesep + \
                 "temps total: " + str(timedelta(seconds=(time.time() - debut))) + os.linesep +\
                 "mémoire: " + str(psutil.virtual_memory())
        f.write(string)
        f.close()
        a = os.path.join(os.path.join(self.path2, folderName), 'print.txt')
        f = open(a, "w")
        f.write(out)

    def natural_selection(self, part, besties, names, iters, times, names2, edges, names3):

        debut = time.time()
        print_out = ""

        for mode in part:

            np.random.seed(None)

            folderName = mode.upper()

            utility.createDirectory(path=self.path2, folderName=folderName)

            # Les axes pour le graphique
            x1, y1, y2, yTps, yEdges = [], [], [], [], []

            scoreMax, indMax, colMax, graphMax = -math.inf, -math.inf, -math.inf, -math.inf

            # Progression des meilleurs éléments
            bestScorePro, bestIndsPro, bestColsPro = [], [], []

            # Mesurer le temps d'execution
            instant = time.time()

            # Initialise le vecteur de probabilité
            probas = self.create_proba()

            # Initialise la population
            pop = self.create_population(probas=probas)

            scores, inds, cols, graphs = utility.calcScore(pop=pop, n_vertices=self.vertices)

            bestScore, bestInd, bestCols, bestGraph, bestScorePro, bestIndsPro, bestColsPro = \
                utility.conjecture_add_list(scores=scores, inds=pop, cols=cols, graphs=graphs,
                                            bestScorePro=bestScorePro, bestIndsPro=bestIndsPro, bestColsPro=bestColsPro)

            generation = 0

            mean_scores = np.mean(heapq.nlargest(int(self.n_pop / 2), scores))
            time_instant = timedelta(seconds=(time.time() - instant))
            time_debut = timedelta(seconds=(time.time() - debut))

            x1, y1, y2, yTps, yEdges = utility.add_axis_vars(bestScore=bestScore, meanScore=mean_scores,
                                                             iter=generation, vars=bestCols, time_debut=time_debut,
                                                             x1=x1, y1=y1, y2=y2, yTps=yTps, yVars=yEdges)

            print_out = utility.my_print_conjecture(print_out=print_out, mode=mode, mean=mean_scores,
                                                    bestScore=bestScore, time_exe=time_instant,
                                                    time_total=time_debut, iter=generation + 1)

            print_out = print_out + "\n"

            # Met à jour le vecteur de probabilité
            probas = self.update_proba(maxi=bestInd, probas=probas)

            # Mutation sur le vecteur de probabilité
            probas = self.mutate_proba(probas=probas)

            scoreMax = bestScore
            colMax = bestCols

            for generation in range(self.n_gen):

                instant = time.time()

                pop = self.create_population(probas=probas)

                scores, inds, cols, graphs = utility.calcScore(pop=pop, n_vertices=self.vertices)

                bestScore, bestInd, bestCols, bestGraph, bestScorePro, bestIndsPro, bestColsPro = \
                    utility.conjecture_add_list(scores=scores, inds=pop, cols=cols, graphs=graphs,
                                                bestScorePro=bestScorePro, bestIndsPro=bestIndsPro,
                                                bestColsPro=bestColsPro)

                generation = generation + 1

                mean_scores = np.mean(heapq.nlargest(int(self.n_pop / 2), scores))
                time_instant = timedelta(seconds=(time.time() - instant))
                time_debut = timedelta(seconds=(time.time() - debut))

                entropy = utility.get_entropy(pop=pop, inds=self.n_pop, size=self.n_ind)

                print_out = utility.new_my_print_conjecture(print_out=print_out, mode=mode, mean=mean_scores,
                                                            bestScore=bestScore, time_exe=time_instant,
                                                            time_total=time_debut, entropy=entropy, iter=generation,
                                                            val="/")

                print_out = print_out + "\n"

                x1, y1, y2, yTps, yEdges = utility.add_axis_vars(bestScore=bestScore, meanScore=mean_scores,
                                                                 iter=generation, vars=bestCols,
                                                                 time_debut=time_debut,
                                                                 x1=x1, y1=y1, y2=y2, yTps=yTps, yVars=yEdges)

                utility.plot_conjecture(x1=x1, y1=y1, y2=y2, yTps=yTps, yEdges=yEdges, n_pop=self.n_pop,
                                        n_gen=self.n_gen, heuristic="Apprentissage incrémental à base de population",
                                        folderName=folderName, path=self.path2, bestScore=bestScore,
                                        bestGraph=bestGraph, mean_scores=mean_scores,
                                        time_total=time_debut.total_seconds())

                probas = self.update_proba(maxi=bestInd, probas=probas)

                probas = self.mutate_proba(probas=probas)

                if bestScore > scoreMax:
                    scoreMax = bestScore
                    indMax = bestInd
                    colMax = bestCols
                    graphMax = bestGraph

                self.write_res(folderName=folderName, probas=probas, y1=y1, y2=y2, colMax=colMax,
                               bestScorePro=bestScorePro,
                               bestScore=bestScore, bestInd=indMax,
                               debut=debut, out=print_out, yTps=yTps, yEdges=yEdges)

            besties, names, iters, times, names2, edges, names3 = \
                utility.queues_put_conjecture(y2=y2, folderName=folderName, scoreMax=scoreMax, iter=generation,
                                              yTps=yTps, time=time_debut.total_seconds(), besties=besties, names=names,
                                              names2=names2, iters=iters, times=times, edges=edges, names3=names3,
                                              yEdges=yEdges, edge=bestCols)

    def init(self):

        print("################################################")
        print("#APPRENTISSAGE INCREMENTAL A BASE DE POPULATION#")
        print("################################################")
        print()

        besties, names, iters, times, names2, edges, names3 = utility.queues_init()

        mods = self.list_exp

        n = len(mods)
        mods = [mods[i::n] for i in range(n)]

        arglist = []
        for part in mods:
            arglist.append((part, besties, names, iters, times, names2, edges, names3))

        pool = multiprocessing.Pool(processes=len(mods))
        pool.starmap(self.natural_selection, arglist)
        pool.close()

        bestiesLst, namesLst, itersLst, timesLst, names2Lst, edgesLst, names3Lst =\
            utility.queues_get(n_process=len(mods), besties=besties, names=names, names2=names2, iters=iters,
                               times=times, features=edges, names3=names3)

        pool.join()

        return utility.res_conjecture(heuristic="Apprentissage incrémental à base de population", besties=bestiesLst,
                                      names=namesLst, times=timesLst, names2=names2Lst, edges=edgesLst,
                                      names3=names3Lst, path=self.path2, conjecture_name="Conjecture2.1")


if __name__ == '__main__':

    pbil = Pbil(list_exp=["EXP1", "EXP2", "EXP3", "EXP4", "EXP5",
                                  "EXP6", "EXP7", "EXP8", "EXP9", "EXP10"],
                pop=171, gen=500, learning_rate=0.1, mut_proba=0.2, mut_shift=0.05)

    pbil.init()