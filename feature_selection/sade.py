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
        string = "heuristique: Evolution différentielle (SADE)" + os.linesep + \
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

    def merge(self, pop, mutants, strats, CRs, cross_probas):
        pop_list = []
        for ind, score, model, col, scoreA, scoreP, scoreR, scoreF in pop:
            pop_list.append(list([list(ind), score, model, col, scoreA, scoreP, scoreR, scoreF]))
        mut_list = []
        for ind, score, model, col, scoreA, scoreP, scoreR, scoreF in mutants:
            mut_list.append(list([list(ind), score, model, col, scoreA, scoreP, scoreR, scoreF]))
        newpop = []
        scores = []
        models = []
        cols = []
        scoresA = []
        scoresP = []
        scoresR = []
        scoresF = []
        ns1 = 0
        ns2 = 0
        nf1 = 0
        nf2 = 0
        for i in range(len(pop_list)):
            if pop_list[i][1] > mut_list[i][1]:
                newpop.append((pop_list[i][0]))
                scores.append((pop_list[i][1]))
                models.append((pop_list[i][2]))
                cols.append((pop_list[i][3]))
                scoresA.append((pop_list[i][4]))
                scoresP.append((pop_list[i][5]))
                scoresR.append((pop_list[i][6]))
                scoresF.append((pop_list[i][7]))
                if strats[i] == 0:
                    nf1 = nf1 + 1
                else:
                    nf2 = nf2 + 1
            else:
                newpop.append((mut_list[i][0]))
                scores.append((mut_list[i][1]))
                models.append((mut_list[i][2]))
                cols.append((mut_list[i][3]))
                scoresA.append((mut_list[i][4]))
                scoresP.append((mut_list[i][5]))
                scoresR.append((mut_list[i][6]))
                scoresF.append((mut_list[i][7]))
                if strats[i] == 0:
                    ns1 = ns1 + 1
                else:
                    ns2 = ns2 + 1
                CRs.append(cross_probas[i])

        return np.array(newpop), scores, models, cols, scoresA, scoresP, scoresR, scoresF, ns1, ns2, nf1, nf2, CRs

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

            # Initalise le vecteur de proba pour les stratégies
            strats_pool = ['de_rand_1', 'de_current_to_best_2']
            p1 = 0.5

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

            CRm = 0.5
            CRs = []
            cross_proba_inds = []

            for generation in range(self.n_gen):

                instant = time.time()

                # Liste des mutants
                mutants = []
                strats = []

                if generation % 5 == 0:
                    cross_proba_inds = []
                    for i in range(self.n_pop):
                        cross_proba = -1
                        while cross_proba > 1 or cross_proba < 0:
                            cross_proba = stats.norm.rvs(loc=CRm, scale=0.1)
                        cross_proba_inds.append(cross_proba)

                # Création des mutants
                for i in range(self.n_pop):

                    F = -1
                    while F < 0:
                        F = stats.norm.rvs(loc=0.5, scale=0.3)
                        if F > 2:
                            F = 2

                    # mutation
                    rand = random.uniform(0, 1)
                    if rand <= p1:
                        mutant = de.mutate(F=F, pop=pop, bestInd=bestInd, ind_pos=i, strat=strats_pool[0])
                        strats.append(0)
                    else:
                        mutant = de.mutate(F=F, pop=pop, bestInd=bestInd, ind_pos=i, strat=strats_pool[1])
                        strats.append(1)

                    # croisement
                    trial = de.crossover(n_ind=self.n_ind, ind=pop[i], mutant=mutant,
                                         cross_proba=cross_proba_inds[i])

                    mutants.append(trial)

                # Calcul du score pour l'ensemble des mutants
                scores_m, scoresA_m, scoresP_m, scoresR_m, scoresF_m, models_m, cols_m = \
                    utility.fitness_feature(n_class=self.n_class, d=self.data, pop=mutants, target_name=self.target,
                                            metric=self.metric, method=mode)

                pop_score = zip(pop, scores, models, cols, scoresA, scoresP, scoresR, scoresF)
                mut_score = zip(mutants, scores_m, models_m, cols_m, scoresA_m, scoresP_m, scoresR_m, scoresF_m)

                pop, scores, models, cols, scoresA, scoresP, scoresR, scoresF, ns1, ns2, nf1, nf2, CRs =\
                    self.merge(pop=pop_score, mutants=mut_score, strats=strats, CRs=CRs, cross_probas=cross_proba_inds)

                bestScore, bestModel, bestInd, bestCols, bestScoreA, bestScoreP, bestScoreR, bestScoreF, bestScorePro, \
                bestModelPro, bestIndsPro, bestColsPro, bestAPro, bestPPro, bestRPro, bestFPro = \
                    utility.feature_add_list(scores=scores, models=models, inds=pop, cols=cols, scoresA=scoresA,
                                             scoresP=scoresP, scoresR=scoresR, scoresF=scoresF,
                                             bestScorePro=bestScorePro,
                                             bestModelPro=bestModelPro, bestIndsPro=bestIndsPro,
                                             bestColsPro=bestColsPro,
                                             bestAPro=bestAPro, bestPPro=bestPPro, bestRPro=bestRPro, bestFPro=bestFPro)

                if generation <= 50:
                    try:
                        p1 = (ns1*(ns2+nf2))/((ns2*(ns1+nf1))+(ns1*(ns2+nf2)))
                    except ZeroDivisionError:
                        p1 = 0

                if generation % 25 == 0:
                    CRm = sum(CRs)/len(CRs)
                    CRs = []

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
                                      heuristic="Evolution différentielle (SADE)", folderName=folderName,
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
        print("#ALGORITHME A EVOLUTION DIFFERENTIELLE (SADE)#")
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

        return utility.res(heuristic="Evolution différentielle (SADE)", besties=bestiesLst, names=namesLst,
                           iters=itersLst, times=timesLst, names2=names2Lst, path=self.path2, dataset=self.dataset)


if __name__ == '__main__':

    diff = Differential(dataset="als", target="survived", metric="recall",
                        list_exp=["EXP1", "EXP2", "EXP3", "EXP4", "EXP5", "EXP6", "EXP7", "EXP8", "EXP9", "EXP10"],
                        pop=30, gen=1000)

    diff.init()
