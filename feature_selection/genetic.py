import utility.utility as utility

import multiprocessing
import numpy as np
import sys
import random
import os
import heapq
import psutil
from operator import itemgetter
import time
from datetime import timedelta

import warnings

warnings.filterwarnings('ignore')

np.set_printoptions(threshold=sys.maxsize)


class Genetic:

    def __init__(self, dataset, target, metric, list_exp, pop, gen, mut):

        self.dataset = dataset
        self.target = target
        self.metric = metric
        self.list_exp = list_exp
        self.n_pop = pop
        self.n_gen = gen
        self.n_mut = mut
        self.path1 = os.path.dirname(os.getcwd()) + '/in/'
        self.path2 = os.path.dirname(os.getcwd()) + '/out'
        self.data = utility.read(filename=(self.path1 + dataset))
        self.cols = self.data.drop([self.target], axis=1).columns
        self.unique, self.count = np.unique(self.data[self.target], return_counts=True)
        self.n_class = len(self.unique)
        self.n_ind = len(self.cols)
        utility.cleanOut()

    def crossover(self, p1, p2):
        decomp = utility.getNumberdecomposition(int(self.n_ind / 2))
        part1 = p1[0:decomp[0]]
        part2 = p2[decomp[0]:-decomp[1]]
        part3 = p1[-decomp[1]:]
        b = bool(random.randint(0, 3))
        if b == 1:
            c1 = list(part1) + list(part2) + list(part3)
            c2 = list(part3) + list(part2) + list(part1)
        elif b == 2:
            c1 = list(part2) + list(part1) + list(part3)
            c2 = list(part2) + list(part3) + list(part1)
        else:
            c1 = list(part1) + list(part3) + list(part2)
            c2 = list(part3) + list(part1) + list(part2)
        c1 = np.array(c1)
        c2 = np.array(c2)
        if np.all(c1 is False):
            c1[random.randint(0, len(c1) - 1)] = True
        if np.all(c2 is False):
            c2[random.randint(0, len(c2) - 1)] = True
        return c1, c2

    def mutate(self, pop):
        mutate_index = random.sample(range(0, len(pop[0])), random.randint(1, self.n_mut))
        for ind in pop:
            for x in mutate_index:
                ind[x] = not ind[x]
        return pop

    def write_res(self, folderName, y1, y2, colMax, bestScorePro,
                  bestAPro, bestPPro, bestRPro, bestFPro, bestModelPro,
                  bestScore, bestScoreA, bestScoreP, bestScoreR,
                  bestScoreF, bestModel, bestInd, debut, out, yTps):
        a = os.path.join(os.path.join(self.path2, folderName), 'resultat.txt')
        f = open(a, "w")
        string = "heuristique: Algorithme génétique" + os.linesep + \
                 "population: " + str(self.n_pop) + os.linesep + \
                 "générations: " + str(self.n_gen) + os.linesep + "mutations: " + str(self.n_mut) + os.linesep + \
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

    def selection(self, pop, i):
        try:
            return pop[i], pop[i + 1]
        except IndexError:
            pass

    def merge(self, pop, new):
        pop_list = []
        for ind, score, model, col, scoreA, scoreP, scoreR, scoreF in pop:
            pop_list.append(list([list(ind), score, model, col, scoreA, scoreP, scoreR, scoreF]))
        new_list = []
        for ind, score, model, col, scoreA, scoreP, scoreR, scoreF in new:
            new_list.append(list([list(ind), score, model, col, scoreA, scoreP, scoreR, scoreF]))
        newpop = []
        scores = []
        models = []
        cols = []
        scoresA = []
        scoresP = []
        scoresR = []
        scoresF = []
        for i in range(len(pop_list)):
            newpop.append((pop_list[i][0]))
            scores.append((pop_list[i][1]))
            models.append((pop_list[i][2]))
            cols.append((pop_list[i][3]))
            scoresA.append((pop_list[i][4]))
            scoresP.append((pop_list[i][5]))
            scoresR.append((pop_list[i][6]))
            scoresF.append((pop_list[i][7]))
        for j in range(len(new_list)):
            newpop.append((new_list[j][0]))
            scores.append((new_list[j][1]))
            models.append((new_list[j][2]))
            cols.append((new_list[j][3]))
            scoresA.append((new_list[j][4]))
            scoresP.append((new_list[j][5]))
            scoresR.append((new_list[j][6]))
            scoresF.append((new_list[j][7]))
        return np.array(newpop), scores, models, cols, scoresA, scoresP, scoresR, scoresF

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

                # Sélectionne les n_pop/2 meilleurs de la population
                i_val = heapq.nlargest(int(self.n_pop / 2), enumerate(scores), key=itemgetter(1))
                scores = [val for (i, val) in sorted(i_val)]
                indexes = [i for (i, val) in sorted(i_val)]
                pop = [pop[x] for x in indexes]
                cols = [cols[x] for x in indexes]
                scoresA = [scoresA[x] for x in indexes]
                scoresP = [scoresP[x] for x in indexes]
                scoresR = [scoresR[x] for x in indexes]
                scoresF = [scoresF[x] for x in indexes]
                models = [models[x] for x in indexes]

                # Création des enfants
                children = []
                for i in range(0, int(self.n_pop / 2), 2):
                    try:
                        p1, p2 = self.selection(pop, i)
                        c1, c2 = self.crossover(p1, p2)
                        children.append(c1)
                        children.append(c2)
                    except:
                        pass

                # Calcul du score pour l'ensemble des enfants
                scores_c, scoresA_c, scoresP_c, scoresR_c, scoresF_c, models_c, cols_c = \
                    utility.fitness_feature(n_class=self.n_class, d=self.data, pop=children, target_name=self.target,
                                            metric=self.metric, method=mode)

                pop_score = zip(pop, scores, models, cols, scoresA, scoresP, scoresR, scoresF)
                children_score = zip(children, scores_c, models_c, cols_c, scoresA_c, scoresP_c, scoresR_c, scoresF_c)

                pop, scores, models, cols, scoresA, scoresP, scoresR, scoresF =\
                    self.merge(pop=pop_score, new=children_score)

                # Création des mutants
                tmp = np.copy(pop)
                mutants = self.mutate(tmp)

                # Calcul du score pour l'ensemble des mutants
                scores_m, scoresA_m, scoresP_m, scoresR_m, scoresF_m, models_m, cols_m = \
                    utility.fitness_feature(n_class=self.n_class, d=self.data, pop=pop, target_name=self.target,
                                            metric=self.metric, method=mode)

                pop_score = zip(pop, scores, models, cols, scoresA, scoresP, scoresR, scoresF)
                mutants_score = zip(mutants, scores_m, models_m, cols_m, scoresA_m, scoresP_m, scoresR_m, scoresF_m)

                pop, scores, models, cols, scoresA, scoresP, scoresR, scoresF =\
                    self.merge(pop=pop_score, new=mutants_score)

                bestScore, bestModel, bestInd, bestCols, bestScoreA, bestScoreP, bestScoreR, bestScoreF, bestScorePro, \
                bestModelPro, bestIndsPro, bestColsPro, bestAPro, bestPPro, bestRPro, bestFPro = \
                    utility.feature_add_list(scores=scores, models=models, inds=pop, cols=cols, scoresA=scoresA,
                                             scoresP=scoresP, scoresR=scoresR, scoresF=scoresF,
                                             bestScorePro=bestScorePro,
                                             bestModelPro=bestModelPro, bestIndsPro=bestIndsPro,
                                             bestColsPro=bestColsPro,
                                             bestAPro=bestAPro, bestPPro=bestPPro, bestRPro=bestRPro, bestFPro=bestFPro)

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
                                     heuristic="Algorithme génétique", folderName=folderName, path=self.path2,
                                     bestScore=bestScore, mean_scores=mean_scores,
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
                                           time=time_debut.total_seconds(), besties=besties, names=names,
                                           names2=names2, iters=iters, times=times)

    def init(self):

        print("######################")
        print("#ALGORITHME GENETIQUE#")
        print("######################")
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

        return utility.res(heuristic="Algorithme génétique", besties=bestiesLst, names=namesLst, iters=itersLst,
                           times=timesLst, names2=names2Lst, path=self.path2, dataset=self.dataset)


if __name__ == '__main__':

    gen = Genetic(dataset="als", target="survived", metric="recall", list_exp=["EXP1", "EXP2"],
                  pop=30, gen=10, mut=5)

    gen.init()
