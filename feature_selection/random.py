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


class Random:
    """
    Parameters
    ----------
    dataset: [string] The name of the file containing the data (.csv or .xlsx)
    target: [string] Target feature for classification
    metric: [string] Choice of metric for optimization [accuracy (default), precision, recall or f1-score]
    list_exp: [list: string] List of learning methods for experiments running in parallel
              - "LR": logistic regression (default)
              - "SVM": support vector machines
              - "KNN": K-nearest neighbors
              - "RDC": random forest
              - "GNB": gaussian naive bayes
    pop: [int] Number of solutions evaluated per generation
    gen: [int] Number of generations/iterations for the algorithm
    proba: [float (0.0:1.0)] Probability to select each features
    """
    def __init__(self, dataset, target, metric, list_exp, pop, gen, proba):

        self.heuristic = "Random"
        self.dataset = dataset
        self.target = target
        self.metric = metric
        self.list_exp = list_exp
        self.n_pop = pop
        self.n_gen = gen
        self.proba = proba
        self.path1 = os.path.dirname(os.getcwd()) + '/in/'
        self.path2 = os.path.dirname(os.getcwd()) + '/out/' + self.heuristic + '/' + dataset
        self.data = utility.read(filename=(self.path1 + dataset))
        self.cols = self.data.drop([self.target], axis=1).columns
        self.unique, self.count = np.unique(self.data[self.target], return_counts=True)
        self.n_class = len(self.unique)
        self.n_ind = len(self.cols)
        utility.cleanOut(path=self.path2)

    def generate_neighbors(self, n_neighbors):
        neighbors = [np.random.choice(a=[False, True], size=self.n_ind, p=[1-self.proba, self.proba])
                     for _ in range(n_neighbors)]
        return list(neighbors)

    def write_res(self, folderName, y1, y2, colMax, bestScorePro,
                  bestAPro, bestPPro, bestRPro, bestFPro, bestModelPro,
                  bestScore, bestScoreA, bestScoreP, bestScoreR,
                  bestScoreF, bestModel, bestInd, debut, out, yTps, yVars, method):
        a = os.path.join(os.path.join(self.path2, folderName), 'resultat.txt')
        f = open(a, "w")
        string = "heuristique: Evolution différentielle" + os.linesep + \
                 "méthode: " + str(method) + os.linesep + \
                 "population: " + str(self.n_pop) + os.linesep + \
                 "générations: " + str(self.n_gen) + os.linesep + \
                 "moyenne: " + str(y1) + os.linesep + "meilleur: " + str(y2) + os.linesep + \
                 "temps: " + str(yTps) + os.linesep + \
                 "nombre de variables: " + str(yVars) + os.linesep + \
                 "scores: " + str(bestScorePro) + os.linesep + "exactitude: " + str(bestAPro) + os.linesep + \
                 "precision: " + str(bestPPro) + os.linesep + "rappel: " + str(bestRPro) + os.linesep + \
                 "fscore: " + str(bestFPro) + os.linesep + "model: " + str(bestModelPro) + os.linesep + \
                 "meilleur score: " + str(bestScore) + os.linesep + "meilleure exactitude: " + str(bestScoreA) + \
                 os.linesep + "meilleure precision: " + str(bestScoreP) + os.linesep + "meilleur rappel: " + \
                 str(bestScoreR) + os.linesep + "meilleur fscore: " + str(bestScoreF) + os.linesep + \
                 "meilleur model: " + str(bestModel) + os.linesep + "meilleur individu: " + str(bestInd) + \
                 os.linesep + "colonnes:" + str(colMax) + os.linesep + \
                 "temps total: " + str(timedelta(seconds=(time.time() - debut))) + os.linesep +\
                 "mémoire: " + str(psutil.virtual_memory())
        f.write(string)
        f.close()
        a = os.path.join(os.path.join(self.path2, folderName), 'print.txt')
        f = open(a, "w")
        f.write(out)

    def search(self, part, besties, names, iters, times, names2, features, names3):

        debut = time.time()
        print_out = ""

        for mode in part:

            np.random.seed(None)

            folderName = mode.upper()

            method = utility.getMethod(mode)

            utility.createDirectory(path=self.path2, folderName=folderName)

            # Les axes pour le graphique
            x1, y1, y2, yTps, yVars = [], [], [], [], []

            scoreMax, modelMax, indMax, colMax, scoreAMax, scorePMax, scoreRMax, scoreFMax = 0, 0, 0, 0, 0, 0, 0, 0

            # Progression des meilleurs éléments
            bestScorePro, bestModelPro, bestColsPro, bestIndsPro, bestAPro, bestPPro, bestRPro, bestFPro =\
                [], [], [], [], [], [], [], []

            # Mesurer le temps d'execution
            instant = time.time()

            initial_solution = np.random.choice(a=[False, True], size=self.n_ind)
            solution = initial_solution

            generation = 0
            time_debut = timedelta(seconds=(time.time() - debut))

            for generation in range(self.n_gen):

                instant = time.time()

                neighbors_solutions = self.generate_neighbors(n_neighbors=self.n_pop)

                neighbors_solutions.append(solution)

                scores, scoresA, scoresP, scoresR, scoresF, models, cols = \
                    utility.fitness_feature(n_class=self.n_class, d=self.data, pop=neighbors_solutions,
                                            target_name=self.target, metric=self.metric, method=method)

                bestScore, bestModel, bestInd, bestCols, bestScoreA, bestScoreP, bestScoreR, bestScoreF, bestScorePro, \
                bestModelPro, bestIndsPro, bestColsPro, bestAPro, bestPPro, bestRPro, bestFPro = \
                    utility.feature_add_list(scores=scores, models=models, inds=neighbors_solutions, cols=cols,
                                             scoresA=scoresA, scoresP=scoresP, scoresR=scoresR, scoresF=scoresF,
                                             bestScorePro=bestScorePro,
                                             bestModelPro=bestModelPro, bestIndsPro=bestIndsPro,
                                             bestColsPro=bestColsPro,
                                             bestAPro=bestAPro, bestPPro=bestPPro, bestRPro=bestRPro, bestFPro=bestFPro)

                generation = generation + 1

                mean_scores = np.mean(heapq.nlargest(int(self.n_pop / 2), scores))
                time_instant = timedelta(seconds=(time.time() - instant))
                time_debut = timedelta(seconds=(time.time() - debut))

                entropy = utility.get_entropy(pop=neighbors_solutions, inds=self.n_pop, size=self.n_ind)

                print_out = utility.new_my_print_feature(print_out=print_out, mode=mode, method=method,
                                                         mean=mean_scores, bestScore=bestScore, numCols=len(bestCols),
                                                         time_exe=time_instant, time_total=time_debut, entropy=entropy,
                                                         iter=generation, val="/")

                print_out = print_out + "\n"

                x1, y1, y2, yTps, yVars = utility.add_axis_vars(bestScore=bestScore, meanScore=mean_scores,
                                                                iter=generation,
                                                                vars=len(bestCols), time_debut=time_debut, x1=x1, y1=y1,
                                                                y2=y2, yTps=yTps, yVars=yVars)

                utility.plot_feature(x1=x1, y1=y1, y2=y2, yTps=yTps, yVars=yVars, n_pop=self.n_pop, n_gen=self.n_gen,
                                     heuristic="Recherche aléatoire", folderName=folderName, path=self.path2,
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
                               bestInd=indMax, debut=debut, out=print_out, yTps=yTps, yVars=yVars, method=method)

                generation = generation + 1

                solution = bestInd

            besties, names, iters, times, names2, features, names3 = \
                utility.queues_put_feature(y2=y2, folderName=folderName, scoreMax=scoreMax, iter=generation, yTps=yTps,
                                           time=time_debut.total_seconds(), besties=besties, names=names, names2=names2,
                                           iters=iters, times=times, features=features, names3=names3, yVars=yVars,
                                           feature=len(bestCols))

    def init(self):

        print("#####################")
        print("#RECHERCHE ALEATOIRE#")
        print("#####################")
        print()

        besties, names, iters, times, names2, features, names3 = utility.queues_init()

        mods = self.list_exp

        n = len(mods)
        mods = [mods[i::n] for i in range(n)]

        arglist = []
        for part in mods:
            arglist.append((part, besties, names, iters, times, names2, features, names3))

        pool = multiprocessing.Pool(processes=len(mods))
        pool.starmap(self.search, arglist)
        pool.close()

        bestiesLst, namesLst, itersLst, timesLst, names2Lst, featuresLst, names3Lst =\
            utility.queues_get(n_process=len(mods), besties=besties, names=names, names2=names2, iters=iters,
                               times=times, features=features, names3=names3)

        pool.join()

        return utility.res(heuristic="Recherche aleatoire", besties=bestiesLst, names=namesLst, iters=itersLst,
                           times=timesLst, names2=names2Lst, features=featuresLst, names3=names3Lst,
                           path=self.path2, dataset=self.dataset)


if __name__ == '__main__':

    alea = Random(dataset="scene", target="Urban",
                  metric="accuracy",
                  list_exp=["lr", "svm", "gnb"],
                  pop=30, gen=20, proba=1/2)

    alea.init()
