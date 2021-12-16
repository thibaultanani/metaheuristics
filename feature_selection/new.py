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

    def __init__(self, dataset, target, metric, list_exp, pop, gen, learning_rate, p):
        self.dataset = dataset
        self.target = target
        self.metric = metric
        self.list_exp = list_exp
        self.n_pop = pop
        self.n_gen = gen
        self.learning_rate = learning_rate
        self.p = p
        self.path1 = os.path.dirname(os.getcwd()) + '/in/'
        self.path2 = os.path.dirname(os.getcwd()) + '/out'
        self.data = utility.read(filename=(self.path1 + dataset))
        self.cols = self.data.drop([self.target], axis=1).columns
        self.unique, self.count = np.unique(self.data[self.target], return_counts=True)
        self.n_class = len(self.unique)
        self.n_ind = len(self.cols)
        utility.cleanOut()

    @staticmethod
    def update_param(muCR, learning_rate, SCR):
        try:
            muCR = (1 - learning_rate) * muCR + learning_rate * (sum([CR ** 2 for CR in SCR]) / sum(SCR))
        except ZeroDivisionError:
            pass
        return muCR

    def write_res(self, folderName, y1, y2, colMax, bestScorePro,
                  bestAPro, bestPPro, bestRPro, bestFPro, bestModelPro,
                  bestScore, bestScoreA, bestScoreP, bestScoreR,
                  bestScoreF, bestModel, bestInd, muCR, debut, out, yTps):
        a = os.path.join(os.path.join(self.path2, folderName), 'resultat.txt')
        f = open(a, "w")
        string = "heuristique: Mon heuristique" + os.linesep + \
                 "population: " + str(self.n_pop) + os.linesep + \
                 "générations: " + str(self.n_gen) + os.linesep + \
                 "learning_rate: " + str(self.learning_rate) + os.linesep + \
                 "p: " + str(self.p) + os.linesep + \
                 "muCR: " + str(muCR) + os.linesep + \
                 "moyenne: " + str(y1) + os.linesep + "meilleur: " + str(y2) + os.linesep + \
                 "temps: " + str(yTps) + os.linesep + \
                 "scores: " + str(bestScorePro) + os.linesep + "exactitude: " + str(bestAPro) + os.linesep + \
                 "precision: " + str(bestPPro) + os.linesep + "rappel: " + str(bestRPro) + os.linesep + \
                 "fscore: " + str(bestFPro) + os.linesep + "model: " + str(bestModelPro) + os.linesep + \
                 "meilleur score: " + str(bestScore) + os.linesep + "meilleure exactitude: " + str(bestScoreA) + \
                 os.linesep + "meilleure precision: " + str(bestScoreP) + os.linesep + "meilleur rappel: " + \
                 str(bestScoreR) + os.linesep + "meilleur fscore: " + str(bestScoreF) + os.linesep + \
                 "meilleur model: " + str(bestModel) + os.linesep + "meilleur individu: " + str(bestInd) + \
                 os.linesep + "colonnes:" + str(colMax) + os.linesep + \
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

            method = utility.getMethod(mode)

            utility.createDirectory(path=self.path2, folderName=folderName)

            # Les axes pour le graphique
            x1 = []
            y1 = []
            y2 = []

            yTps = []

            scoreMax = 0
            modelMax = 0
            indMax = 0
            colMax = 0
            scoreAMax = 0
            scorePMax = 0
            scoreRMax = 0
            scoreFMax = 0

            # Progression des meilleurs éléments
            bestScorePro = []
            bestModelPro = []
            bestColsPro = []
            bestIndsPro = []
            bestAPro = []
            bestPPro = []
            bestRPro = []
            bestFPro = []

            # Mesurer le temps d'execution
            instant = time.time()

            # Initialise la population
            pop = utility.create_population_feature(inds=self.n_pop, size=self.n_ind)

            # Initalise les paramètres
            muCR = 0.5

            # Initialiser l'archive
            archive = []

            scores, scoresA, scoresP, scoresR, scoresF, models, cols = \
                utility.fitness_feature(n_class=self.n_class, d=self.data, pop=pop, target_name=self.target,
                                        metric=self.metric, method=method)

            bestScore, bestModel, bestInd, bestCols, bestScoreA, bestScoreP, bestScoreR, bestScoreF, bestScorePro, \
            bestModelPro, bestIndsPro, bestColsPro, bestAPro, bestPPro, bestRPro, bestFPro = \
                utility.feature_add_list(scores=scores, models=models, inds=pop, cols=cols, scoresA=scoresA,
                                         scoresP=scoresP, scoresR=scoresR, scoresF=scoresF, bestScorePro=bestScorePro,
                                         bestModelPro=bestModelPro, bestIndsPro=bestIndsPro, bestColsPro=bestColsPro,
                                         bestAPro=bestAPro, bestPPro=bestPPro, bestRPro=bestRPro, bestFPro=bestFPro)

            generation = 0

            mean_scores = np.mean(heapq.nlargest(int(self.n_pop / 2), scores))
            time_instant = timedelta(seconds=(time.time() - instant))
            time_debut = timedelta(seconds=(time.time() - debut))

            x1, y1, y2, yTps = utility.add_axis(bestScore=bestScore, meanScore=mean_scores, iter=generation,
                                                time_debut=time_debut, x1=x1, y1=y1, y2=y2, yTps=yTps)

            print_out = utility.my_print_feature(print_out=print_out, mode=mode, method=method, mean=mean_scores,
                                                 bestScore=bestScore, numCols=len(bestCols), time_exe=time_instant,
                                                 time_total=time_debut, iter=generation)

            print_out = print_out + "\n"

            pbest = self.n_pop

            div = int(self.n_gen / self.p)
            div_pbest = int(self.n_pop / self.p)

            archive.append(bestInd)

            for generation in range(self.n_gen):

                if (generation % div) == 0 and generation != 0:
                    pbest = pbest - div_pbest
                    print("nouvelle valeur de pbest: ", pbest)
                    if pbest < 1:
                        pbest = 1

                instant = time.time()

                # Liste des bons croisements et du facteur F
                cross_probas = []

                # Création des mutants
                for i in range(self.n_pop):

                    # Les indices des individus avec les scores les plus élevés
                    indices = (-np.array(scores)).argsort()[:pbest]

                    cross_proba = -1
                    while cross_proba > 1 or cross_proba < 0:
                        cross_proba = stats.norm.rvs(loc=muCR, scale=0.2)

                    pindex = indices[random.randint(0, len(indices) - 1)]
                    pInd = pop[pindex]

                    mutant = de.mutate_new_jade(n_ind=self.n_ind, pop=pop, pInd=pInd, ind_pos=i, ind_pbest=pindex)

                    if (pop[i] == mutant).all():
                        mutant = archive[random.randint(0, self.n_pop - 1)]

                    # croisement
                    trial = de.crossover(n_ind=self.n_ind, ind=pop[i], mutant=mutant, cross_proba=cross_proba)

                    score_m, accuracy_m, precision_m, recall_m, fscore_m, model_m, col_m = \
                        utility.fitness_ind_feature(n_class=self.n_class, d=self.data, ind=trial,
                                                    target_name=self.target, metric=self.metric, method=method)

                    if scores[i] < score_m:
                        archive.append((pop[i]))

                        pop[i] = trial
                        scores[i] = score_m
                        scoresA[i] = accuracy_m
                        scoresP[i] = precision_m
                        scoresR[i] = recall_m
                        scoresF[i] = fscore_m
                        models[i] = model_m
                        cols[i] = col_m

                        cross_probas.append(cross_proba)

                        bestScore, bestModel, bestInd, bestCols, bestScoreA, bestScoreP, bestScoreR, bestScoreF, \
                        bestScorePro, bestModelPro, bestIndsPro, bestColsPro, bestAPro, bestPPro, bestRPro, bestFPro = \
                            utility.feature_add_list(scores=scores, models=models, inds=pop, cols=cols, scoresA=scoresA,
                                                     scoresP=scoresP, scoresR=scoresR, scoresF=scoresF,
                                                     bestScorePro=bestScorePro,
                                                     bestModelPro=bestModelPro, bestIndsPro=bestIndsPro,
                                                     bestColsPro=bestColsPro,
                                                     bestAPro=bestAPro, bestPPro=bestPPro, bestRPro=bestRPro,
                                                     bestFPro=bestFPro)

                # Enlever les doublons de nos listes
                cross_probas = utility.f7(cross_probas)

                uniques = []
                for arr in archive:
                    if not any(np.array_equal(arr, unique_arr) for unique_arr in uniques):
                        uniques.append(arr)

                archive = uniques

                while len(archive) > self.n_pop:
                    archive.pop(random.randint(0, self.n_pop - 1))

                muCR = self.update_param(muCR=muCR, learning_rate=self.learning_rate, SCR=cross_probas)

                generation = generation + 1

                mean_scores = np.mean(heapq.nlargest(int(self.n_pop / 2), scores))
                time_instant = timedelta(seconds=(time.time() - instant))
                time_debut = timedelta(seconds=(time.time() - debut))

                print_out = utility.my_print_feature(print_out=print_out, mode=mode, method=method, mean=mean_scores,
                                                     bestScore=bestScore, numCols=len(bestCols), time_exe=time_instant,
                                                     time_total=time_debut, iter=generation)

                print_out = print_out + "\n"

                x1, y1, y2, yTps = utility.add_axis(bestScore=bestScore, meanScore=mean_scores, iter=generation,
                                                    time_debut=time_debut, x1=x1, y1=y1, y2=y2, yTps=yTps)

                utility.plot_feature(x1=x1, y1=y1, y2=y2, yTps=yTps, n_pop=self.n_pop, n_gen=self.n_gen,
                                     heuristic="Mon heuristique", folderName=folderName,
                                     path=self.path2, bestScore=bestScore, mean_scores=mean_scores,
                                     time_total=time_debut.total_seconds(), metric=self.metric)

                if bestScore > scoreMax:
                    scoreMax = bestScore
                    modelMax = bestModel
                    indMax = bestInd
                    colMax = bestCols
                    scoreAMax = bestScoreA
                    scorePMax = bestScoreP
                    scoreRMax = bestScoreR
                    scoreFMax = bestScoreF

                self.write_res(folderName=folderName, y1=y1, y2=y2, colMax=colMax,
                               bestScorePro=bestScorePro, bestAPro=bestAPro, bestPPro=bestPPro, bestRPro=bestRPro,
                               bestFPro=bestFPro, bestModelPro=bestModelPro, bestScore=bestScore, bestScoreA=scoreAMax,
                               bestScoreP=scorePMax, bestScoreR=scoreRMax, bestScoreF=scoreFMax, bestModel=modelMax,
                               bestInd=indMax, muCR=muCR, debut=debut, out=print_out, yTps=yTps)

            besties, names, iters, times, names2 = \
                utility.queues_put_feature(y2=y2, folderName=folderName, scoreMax=scoreMax, iter=generation, yTps=yTps,
                                           time=time_debut.total_seconds(), besties=besties, names=names, names2=names2,
                                           iters=iters, times=times)

    def init(self):

        print("#################")
        print("#MON HEURISTIQUE#")
        print("#################")
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

        bestiesLst, namesLst, itersLst, timesLst, names2Lst = \
            utility.queues_get(n_process=len(processes), besties=besties, names=names, names2=names2, iters=iters,
                               times=times)

        for process in processes:
            process.join()

        return utility.res(heuristic="Mon heuristique",
                           besties=bestiesLst, names=namesLst, iters=itersLst,
                           times=timesLst, names2=names2Lst, path=self.path2,
                           dataset=self.dataset)


if __name__ == '__main__':
    diff = Differential(dataset="als", target="survived",
                        metric="recall",
                        list_exp=["EXP1", "EXP2", "EXP3", "EXP4", "EXP5", "EXP6", "EXP7", "EXP8", "EXP9", "EXP10"],
                        pop=20, gen=10, learning_rate=0.1, p=5)

    diff.init()
