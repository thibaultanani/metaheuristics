import utility.de as de
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
from scipy import stats

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
    learning_rate: [float (0.0:1.0)] Speed at which the crossover probability is going to converge toward 0.0 or 1.0
    alpha: [float] Speed at which the number of p decrease for selecting one of p_best indivuals
    """
    def __init__(self, list_exp, pop, gen, learning_rate, alpha):

        self.heuristic = "BPLDE"
        self.list_exp = list_exp
        self.n_pop = pop
        self.n_gen = gen
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.path1 = os.path.dirname(os.getcwd()) + '/in/'
        self.path2 = os.path.dirname(os.getcwd()) + '/out/' + self.heuristic + '/conjecture'
        self.vertices = 19
        self.n_ind = int(self.vertices*(self.vertices-1)/2)
        utility.cleanOut(path=self.path2)

    def update_param(self, muCR, SCR):
        try:
            muCR = (1 - self.learning_rate) * muCR + self.learning_rate * (sum(SCR)/len(SCR))
        except ZeroDivisionError:
            cross_proba = -1
            while cross_proba > 1 or cross_proba < 0:
                cross_proba = stats.norm.rvs(loc=muCR, scale=0.1)
            muCR = (1 - self.learning_rate) * muCR + self.learning_rate * cross_proba
        return muCR

    def write_res(self, folderName, probas, y1, y2, colMax, bestScorePro, bestScore, bestInd, debut, out, yTps, yEdges):
        a = os.path.join(os.path.join(self.path2, folderName), 'resultat.txt')
        f = open(a, "w")
        string = "heuristique: Evolution différentielle" + os.linesep + \
                 "population: " + str(self.n_pop) + os.linesep + \
                 "générations: " + str(self.n_gen) + os.linesep + \
                 "taux d'apprentissage: " + str(self.learning_rate) + os.linesep + \
                 "paramètre alpha: " + str(self.alpha) + os.linesep + \
                 "probabilité de croisements: " + str(probas) + os.linesep + \
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

            # Initalise les paramètres
            muCR = 0.5
            muCRs = []

            # Initialise la population
            pop = utility.create_population_feature(inds=self.n_pop, size=self.n_ind)

            # Initialiser l'archive
            archive = []

            scores, inds, cols, graphs = utility.calcScore(pop=pop, n_vertices=self.vertices)

            pop_bar = utility.create_population_feature_bar(pop=pop)

            scores_bar, inds_bar, cols_bar, graphs_bar = utility.calcScore(pop=pop_bar, n_vertices=self.vertices)

            for i in range(self.n_pop):
                if scores[i] < scores_bar[i]:
                    pop[i] = pop_bar[i]
                    scores[i] = scores_bar[i]
                    cols[i] = cols_bar[i]
                    graphs[i] = graphs_bar[i]

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

            pbest = int(self.n_pop)

            archive.append(bestInd)

            for generation in range(self.n_gen):

                instant = time.time()

                # Liste des bons croisements
                cross_probas = []

                val = max(1, round(self.n_pop * (1 - (math.sqrt((generation / self.n_gen)*self.alpha)))))

                mylist = list(range(self.n_pop))
                random.shuffle(mylist)

                half = int(len(mylist)//2)
                list1 = mylist[:half]

                # Création des mutants
                for i in range(self.n_pop):

                    indices = (-np.array(scores)).argsort()[:val]

                    cross_proba = -1
                    while cross_proba > 1 or cross_proba < 0:
                        cross_proba = stats.norm.rvs(loc=muCR, scale=0.1)

                    pop_archive = np.vstack((pop, archive))

                    if i in list1:
                        pindex = indices[random.randint(0, len(indices) - 1)]
                    else:
                        pindex = indices[0]

                    pInd = pop[pindex]

                    while True:
                        archive_index = random.randint(0, len(pop_archive) - 1)
                        if (pindex != archive_index) and (i != archive_index):
                            break

                    idxs = [idx for idx in range(len(pop)) if idx != i and idx != pindex and idx != archive_index]
                    selected = np.random.choice(idxs, 2, replace=False)
                    xr1, xr2 = pop[selected]

                    xra = pop_archive[archive_index]

                    mutant = []

                    for j in range(self.n_ind):
                        mutant.append(((xr1[j] ^ xra[j]) and xr1[j]) or (not (xr1[j] ^ xra[j]) and pInd[j]))

                    trial = de.crossover(n_ind=self.n_ind, ind=pop[i], mutant=mutant, cross_proba=cross_proba)

                    score_m, col_m, graph_m = utility.calcScore_ind(ind=trial, n_vertices=self.vertices)

                    if scores[i] < score_m:
                        archive.append((pop[i]))

                        pop[i] = trial
                        scores[i] = score_m
                        cols[i] = col_m
                        graphs[i] = graph_m

                        cross_probas.append(cross_proba)

                        bestScore, bestInd, bestCols, bestGraph, bestScorePro, bestIndsPro, bestColsPro = \
                            utility.conjecture_add_list(scores=scores, inds=pop, cols=cols, graphs=graphs,
                                                        bestScorePro=bestScorePro, bestIndsPro=bestIndsPro,
                                                        bestColsPro=bestColsPro)

                        uniques = []
                        for arr in archive:
                            if not any(np.array_equal(arr, unique_arr) for unique_arr in uniques):
                                uniques.append(arr)

                        archive = uniques

                        while len(archive) > self.n_pop:
                            archive.pop(random.randint(0, len(archive) - 1))

                # Enlever les doublons de nos listes
                cross_probas = utility.f7(cross_probas)

                muCR = self.update_param(muCR=muCR, SCR=cross_probas)

                muCRs.append(muCR)

                generation = generation + 1

                mean_scores = np.mean(heapq.nlargest(int(self.n_pop / 2), scores))
                time_instant = timedelta(seconds=(time.time() - instant))
                time_debut = timedelta(seconds=(time.time() - debut))

                entropy = utility.get_entropy(pop=pop, inds=self.n_pop, size=self.n_ind)

                print_out = utility.new_my_print_conjecture(print_out=print_out, mode=mode, mean=mean_scores,
                                                            bestScore=bestScore, time_exe=time_instant,
                                                            time_total=time_debut, entropy=entropy, iter=generation,
                                                            val=val)

                print_out = print_out + "\n"

                x1, y1, y2, yTps, yEdges = utility.add_axis_vars(bestScore=bestScore, meanScore=mean_scores,
                                                                 iter=generation, vars=bestCols,
                                                                 time_debut=time_debut,
                                                                 x1=x1, y1=y1, y2=y2, yTps=yTps, yVars=yEdges)

                utility.plot_conjecture(x1=x1, y1=y1, y2=y2, yTps=yTps, yEdges=yEdges, n_pop=self.n_pop,
                                        n_gen=self.n_gen, heuristic="Evolution différentielle progressive binaire",
                                        folderName=folderName, path=self.path2, bestScore=bestScore,
                                        bestGraph=bestGraph, mean_scores=mean_scores,
                                        time_total=time_debut.total_seconds())

                if bestScore > scoreMax:
                    scoreMax = bestScore
                    indMax = bestInd
                    colMax = bestCols
                    graphMax = bestGraph

                self.write_res(folderName=folderName, probas=muCRs, y1=y1, y2=y2, colMax=colMax,
                               bestScorePro=bestScorePro,
                               bestScore=bestScore, bestInd=indMax,
                               debut=debut, out=print_out, yTps=yTps, yEdges=yEdges)

            besties, names, iters, times, names2, edges, names3 = \
                utility.queues_put_conjecture(y2=y2, folderName=folderName, scoreMax=scoreMax, iter=generation,
                                              yTps=yTps, time=time_debut.total_seconds(), besties=besties, names=names,
                                              names2=names2, iters=iters, times=times, edges=edges, names3=names3,
                                              yEdges=yEdges, edge=bestCols)

    def init(self):

        print("##############################################")
        print("#EVOLUTION DIFFERENTIELLE PROGRESSIVE BINAIRE#")
        print("##############################################")
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

        return utility.res_conjecture(heuristic="Evolution différentielle progressive binaire", besties=bestiesLst,
                                      names=namesLst, times=timesLst, names2=names2Lst, edges=edgesLst,
                                      names3=names3Lst, path=self.path2, conjecture_name="Conjecture2.1")


if __name__ == '__main__':

    diff = Differential(list_exp=["EXP1", "EXP2", "EXP3", "EXP4", "EXP5", "EXP6", "EXP7", "EXP8", "EXP9", "EXP10"],
                        pop=300, gen=5000, learning_rate=0.005,
                        alpha=0.5)

    diff.init()