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

    def __init__(self, dataset, target, metric, list_exp, pop, gen):

        self.dataset = dataset
        self.target = target
        self.metric = metric
        self.list_exp = list_exp
        self.n_pop = pop
        self.n_gen = gen
        self.path1 = os.path.dirname(os.getcwd()) + '/in/'
        self.path2 = os.path.dirname(os.getcwd()) + '/out'
        self.data = utility.read(filename=(self.path1 + dataset))
        self.cols = self.data.drop([self.target], axis=1).columns
        self.unique, self.count = np.unique(self.data[self.target], return_counts=True)
        self.n_class = len(self.unique)
        self.n_ind = len(self.cols)
        utility.cleanOut()

    def write_res(self, folderName, y1, y2, colMax, bestScorePro,
                  bestAPro, bestPPro, bestRPro, bestFPro, bestModelPro,
                  bestScore, bestScoreA, bestScoreP, bestScoreR,
                  bestScoreF, bestModel, bestInd, debut, out, yTps):
        a = os.path.join(os.path.join(self.path2, folderName), 'resultat.txt')
        f = open(a, "w")
        string = "heuristique: Evolution différentielle (CODE)" + os.linesep + \
                 "population: " + str(self.n_pop) + os.linesep + \
                 "générations: " + str(self.n_gen) + os.linesep + \
                 "moyenne: " + str(y1) + os.linesep + "meilleur: " + str(y2) + os.linesep + \
                 "temps: " + str(yTps) + os.linesep + \
                 "scores: " + str(bestScorePro) + os.linesep + "exactitude: " + str(bestAPro) + os.linesep + \
                 "precision: " + str(bestPPro) + os.linesep + "rappel: " + str(bestRPro) + os.linesep + \
                 "fscore: " + str(bestFPro) + os.linesep + "model: " + str(bestModelPro) + os.linesep + \
                 "meilleur score: " + str(bestScore) + os.linesep + "meilleure exactitude: " + str(bestScoreA) + \
                 os.linesep + "meilleure precision: " + str(bestScoreP) + os.linesep + "meilleur rappel: " + \
                 str(bestScoreR) + os.linesep + "meilleur fscore: " + str(bestScoreF) + os.linesep + \
                 "meilleur model: " + str(bestModel) + "meilleur individu: " + str(bestInd) + \
                 "colonnes:" + str(colMax) + os.linesep + \
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

            # Choix de paramétrage
            params_pool = [[1.0, 0.1], [1.0, 0.9], [0.8, 0.2], [1.0, 0.5]]
            strats_pool = ['de_rand_1', 'de_rand_2', 'de_current_to_rand_1', 'de_best_1']
            n_pool = len(params_pool)

            scores, scoresA, scoresP, scoresR, scoresF, models, cols = \
                utility.fitness_feature(n_class=self.n_class, d=self.data, pop=pop, target_name=self.target,
                                        metric=self.metric, method=mode)

            bestScore, bestModel, bestInd, bestCols, bestScoreA, bestScoreP, bestScoreR, bestScoreF, bestScorePro,\
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

            print_out = utility.my_print_feature(print_out=print_out, mode=mode, mean=mean_scores,
                                                 bestScore=bestScore, numCols=len(bestCols), time_exe=time_instant,
                                                 time_total=time_debut, iter=generation)

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
                            trial = de.crossover(n_ind=self.n_ind, ind=pop[i], mutant=mutant, cross_proba=params[1])
                        else:
                            trial = mutant
                        trials.append(trial)

                    scores_t, scoresA_t, scoresP_t, scoresR_t, scoresF_t, models_t, cols_t = \
                        utility.fitness_feature(n_class=self.n_class, d=self.data, pop=trials,
                                                target_name=self.target, metric=self.metric, method=mode)
                    argmax_t = np.argmax(scores_t)

                    if scores[i] < scores_t[argmax_t]:
                        pop[i] = trials[argmax_t]
                        scores[i] = scores_t[argmax_t]
                        scoresA[i] = scoresA_t[argmax_t]
                        scoresP[i] = scoresP_t[argmax_t]
                        scoresR[i] = scoresR_t[argmax_t]
                        scoresF[i] = scoresF_t[argmax_t]
                        models[i] = models_t[argmax_t]
                        cols[i] = cols_t[argmax_t]

                        bestScore, bestModel, bestInd, bestCols, bestScoreA, bestScoreP, bestScoreR, bestScoreF,\
                        bestScorePro, bestModelPro, bestIndsPro, bestColsPro, bestAPro, bestPPro, bestRPro, bestFPro = \
                            utility.feature_add_list(scores=scores, models=models, inds=pop, cols=cols, scoresA=scoresA,
                                                     scoresP=scoresP, scoresR=scoresR, scoresF=scoresF,
                                                     bestScorePro=bestScorePro,
                                                     bestModelPro=bestModelPro, bestIndsPro=bestIndsPro,
                                                     bestColsPro=bestColsPro,
                                                     bestAPro=bestAPro, bestPPro=bestPPro, bestRPro=bestRPro,
                                                     bestFPro=bestFPro)

                generation = generation + 1

                mean_scores = np.mean(heapq.nlargest(int(self.n_pop / 2), scores))
                time_instant = timedelta(seconds=(time.time() - instant))
                time_debut = timedelta(seconds=(time.time() - debut))

                print_out = utility.my_print_feature(print_out=print_out, mode=mode, mean=mean_scores,
                                                     bestScore=bestScore, numCols=len(bestCols), time_exe=time_instant,
                                                     time_total=time_debut, iter=generation)

                print_out = print_out + "\n"

                x1, y1, y2, yTps = utility.add_axis(bestScore=bestScore, meanScore=mean_scores, iter=generation,
                                                    time_debut=time_debut, x1=x1, y1=y1, y2=y2, yTps=yTps)

                utility.plot_feature(x1=x1, y1=y1, y2=y2, yTps=yTps, n_pop=self.n_pop, n_gen=self.n_gen,
                                     heuristic="Evolution différentielle (CODE)", folderName=folderName,
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
                               bestInd=indMax, debut=debut, out=print_out, yTps=yTps)

            besties, names, iters, times, names2 = \
                utility.queues_put_feature(y2=y2, folderName=folderName, scoreMax=scoreMax, iter=generation, yTps=yTps,
                                           time=time_debut.total_seconds(), besties=besties, names=names, names2=names2,
                                           iters=iters, times=times)

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

    diff = Differential(dataset="als", target="survived", metric="recall", list_exp=["EXP1", "EXP2"], pop=30, gen=10)

    diff.init()
