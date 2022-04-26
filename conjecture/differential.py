import math

import utility.de as de
import utility.utility as utility

import multiprocessing
import numpy as np
import sys
import os
import heapq
import psutil
import time
from datetime import timedelta

import warnings

warnings.filterwarnings('ignore')

np.set_printoptions(threshold=sys.maxsize)


class Differential:
    """
    Parameters
    ----------
    list_exp: [list: string] List of experiments running in parallel
    pop: [int] Number of solutions evaluated per generation
    gen: [int] Number of generations/iterations for the algorithm
    cross_proba: [float (0.0:1.0)] Crossover probability
    F: [float (0.0:2.0)] F factor
    strat: [string] Mutation strat [de_rand_1, de_best_1, de_current_to_best_1, de_rand_to_best_1,
                                    de_rand_2, de_best_2, de_current_to_best_2, de_rand_to_best_2]
    """
    def __init__(self, list_exp, pop, gen, cross_proba, F, strat):

        self.heuristic = "Differential"
        self.list_exp = list_exp
        self.n_pop = pop
        self.n_gen = gen
        self.cross_proba = cross_proba
        self.F = F
        self.strat = strat
        self.path1 = os.path.dirname(os.getcwd()) + '/in/'
        self.path2 = os.path.dirname(os.getcwd()) + '/out/' + self.heuristic + '/conjecture'
        self.vertices = 19
        self.n_ind = int(self.vertices*(self.vertices-1)/2)
        utility.cleanOut(path=self.path2)

    def write_res(self, folderName, y1, y2, colMax, bestScorePro, bestScore, bestInd, debut, out, yTps, yEdges):
        a = os.path.join(os.path.join(self.path2, folderName), 'resultat.txt')
        f = open(a, "w")
        string = "heuristique: Evolution différentielle" + os.linesep + \
                 "population: " + str(self.n_pop) + os.linesep + \
                 "générations: " + str(self.n_gen) + os.linesep + \
                 "probabilité de croisement: " + str(self.cross_proba) + os.linesep +\
                 "F: " + str(self.F) + os.linesep +\
                 "stratégie de mutation: " + str(self.strat) + os.linesep +\
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

    def differential_evolution(self, part, besties, names, iters, times, names2, edges, names3):

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

            # Initialise la population
            pop = utility.create_population_feature(inds=self.n_pop, size=self.n_ind)

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

            for generation in range(self.n_gen):

                instant = time.time()

                # Création des mutants
                for i in range(self.n_pop):

                    # mutation
                    mutant = de.mutate(F=self.F, pop=pop, bestInd=bestInd, ind_pos=i, strat=self.strat)

                    # croisement
                    trial = de.crossover(n_ind=self.n_ind, ind=pop[i], mutant=mutant,
                                         cross_proba=self.cross_proba)

                    score_m, col_m, graph_m = utility.calcScore_ind(ind=trial, n_vertices=self.vertices)

                    if scores[i] < score_m:
                        pop[i] = trial
                        scores[i] = score_m
                        cols[i] = col_m
                        graphs[i] = graph_m

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
                                        n_gen=self.n_gen, heuristic="Evolution différentielle", folderName=folderName,
                                        path=self.path2, bestScore=bestScore, bestGraph=bestGraph,
                                        mean_scores=mean_scores, time_total=time_debut.total_seconds())

                if bestScore > scoreMax:
                    scoreMax = bestScore
                    indMax = bestInd
                    colMax = bestCols
                    graphMax = bestGraph

                self.write_res(folderName=folderName, y1=y1, y2=y2, colMax=colMax, bestScorePro=bestScorePro,
                               bestScore=bestScore, bestInd=indMax,
                               debut=debut, out=print_out, yTps=yTps, yEdges=yEdges)

            besties, names, iters, times, names2, edges, names3 = \
                utility.queues_put_conjecture(y2=y2, folderName=folderName, scoreMax=scoreMax, iter=generation,
                                              yTps=yTps, time=time_debut.total_seconds(), besties=besties, names=names,
                                              names2=names2, iters=iters, times=times, edges=edges, names3=names3,
                                              yEdges=yEdges, edge=bestCols)

    def init(self):

        print("#######################################")
        print("#ALGORITHME A EVOLUTION DIFFERENTIELLE#")
        print("#######################################")
        print()

        besties, names, iters, times, names2, edges, names3 = utility.queues_init()

        mods = self.list_exp

        n = len(mods)
        mods = [mods[i::n] for i in range(n)]

        arglist = []
        for part in mods:
            arglist.append((part, besties, names, iters, times, names2, edges, names3))

        pool = multiprocessing.Pool(processes=len(mods))
        pool.starmap(self.differential_evolution, arglist)
        pool.close()

        bestiesLst, namesLst, itersLst, timesLst, names2Lst, edgesLst, names3Lst =\
            utility.queues_get(n_process=len(mods), besties=besties, names=names, names2=names2, iters=iters,
                               times=times, features=edges, names3=names3)

        pool.join()

        return utility.res_conjecture(heuristic="Evolution différentielle", besties=bestiesLst, names=namesLst,
                                      times=timesLst, names2=names2Lst, edges=edgesLst, names3=names3Lst,
                                      path=self.path2, conjecture_name="Conjecture2.1")


if __name__ == '__main__':

    diff = Differential(list_exp=["EXP1", "EXP2", "EXP3", "EXP4", "EXP5",
                                  "EXP6", "EXP7", "EXP8", "EXP9", "EXP10"], pop=171, gen=500, cross_proba=0.5, F=1,
                        strat='de_best_1')

    diff.init()