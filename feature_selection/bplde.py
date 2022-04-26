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
    learning_rate: [float (0.0:1.0)] Speed at which the crossover probability is going to converge toward 0.0 or 1.0
    alpha: [float] Speed at which the number of p decrease for selecting one of p_best indivuals
    """
    def __init__(self, dataset, target, metric, list_exp, pop, gen, learning_rate, alpha):

        self.heuristic = "BPLDE"
        self.dataset = dataset
        self.target = target
        self.metric = metric
        self.list_exp = list_exp
        self.n_pop = pop
        self.n_gen = gen
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.path1 = os.path.dirname(os.getcwd()) + '/in/'
        self.path2 = os.path.dirname(os.getcwd()) + '/out/' + self.heuristic + '/' + dataset
        self.data = utility.read(filename=(self.path1 + dataset))
        self.cols = self.data.drop([self.target], axis=1).columns
        self.unique, self.count = np.unique(self.data[self.target], return_counts=True)
        self.n_class = len(self.unique)
        self.n_ind = len(self.cols)
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

    def write_res(self, folderName, probas, y1, y2, colMax, bestScorePro,
                  bestAPro, bestPPro, bestRPro, bestFPro, bestModelPro,
                  bestScore, bestScoreA, bestScoreP, bestScoreR,
                  bestScoreF, bestModel, bestInd, debut, out, yTps, yVars, method):
        a = os.path.join(os.path.join(self.path2, folderName), 'resultat.txt')
        f = open(a, "w")
        string = "heuristique: Evolution différentielle progressive binaire" + os.linesep + \
                 "méthode: " + str(method) + os.linesep + \
                 "population: " + str(self.n_pop) + os.linesep + \
                 "générations: " + str(self.n_gen) + os.linesep + \
                 "taux d'apprentissage: " + str(self.learning_rate) + os.linesep + \
                 "paramètre alpha: " + str(self.alpha) + os.linesep + \
                 "probabilité de croisements: " + str(probas) + os.linesep + \
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

    def differential_evolution(self, part, besties, names, iters, times, names2, features, names3):

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

            # Initalise les paramètres
            muCR = 0.5
            muCRs = []

            # Initialise la population
            pop = utility.create_population_feature(inds=self.n_pop, size=self.n_ind)

            # Initialiser l'archive
            archive = []

            scores, scoresA, scoresP, scoresR, scoresF, models, cols = \
                utility.fitness_feature(n_class=self.n_class, d=self.data, pop=pop, target_name=self.target,
                                        metric=self.metric, method=method)

            pop_bar = utility.create_population_feature_bar(pop=pop)

            scores_bar, scoresA_bar, scoresP_bar, scoresR_bar, scoresF_bar, models_bar, cols_bar = \
                utility.fitness_feature(n_class=self.n_class, d=self.data, pop=pop_bar, target_name=self.target,
                                        metric=self.metric, method=method)

            for i in range(self.n_pop):
                if scores[i] < scores_bar[i]:
                    pop[i] = pop_bar[i]
                    scores[i] = scores_bar[i]
                    scoresA[i] = scoresA_bar[i]
                    scoresP[i] = scoresP_bar[i]
                    scoresR[i] = scoresR_bar[i]
                    scoresF[i] = scoresF_bar[i]
                    models[i] = models_bar[i]
                    cols[i] = cols_bar[i]

            bestScore, worstScore, bestModel, bestInd, bestCols, bestScoreA, bestScoreP, bestScoreR, \
            bestScoreF, bestScorePro, bestModelPro, bestIndsPro, bestColsPro, bestAPro, bestPPro, bestRPro, \
            bestFPro = \
                utility.feature_add_list_plus(scores=scores, models=models, inds=pop, cols=cols,
                                              scoresA=scoresA, scoresP=scoresP, scoresR=scoresR,
                                              scoresF=scoresF, bestScorePro=bestScorePro,
                                              bestModelPro=bestModelPro, bestIndsPro=bestIndsPro,
                                              bestColsPro=bestColsPro, bestAPro=bestAPro, bestPPro=bestPPro,
                                              bestRPro=bestRPro, bestFPro=bestFPro)

            generation = 0

            mean_scores = np.mean(heapq.nlargest(int(self.n_pop / 2), scores))
            time_instant = timedelta(seconds=(time.time() - instant))
            time_debut = timedelta(seconds=(time.time() - debut))

            x1, y1, y2, yTps, yVars = utility.add_axis_vars(bestScore=bestScore, meanScore=mean_scores, iter=generation,
                                                            vars=len(bestCols), time_debut=time_debut, x1=x1, y1=y1,
                                                            y2=y2, yTps=yTps, yVars=yVars)

            print_out = utility.my_print_feature(print_out=print_out, mode=mode, method=method, mean=mean_scores,
                                                 bestScore=bestScore, numCols=len(bestCols), time_exe=time_instant,
                                                 time_total=time_debut, iter=generation)

            print_out = print_out + "\n"

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

                    score_m, accuracy_m, precision_m, recall_m, fscore_m, model_m, col_m = \
                        utility.fitness_ind_feature(n_class=self.n_class, d=self.data, ind=trial,
                                                    target_name=self.target, metric=self.metric, method=method)

                    if scores[i] < score_m or ((scores[i] == score_m) and (len(cols[i]) > len(col_m))):
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

                        bestScore, worstScore, bestModel, bestInd, bestCols, bestScoreA, bestScoreP, bestScoreR,\
                        bestScoreF, bestScorePro, bestModelPro, bestIndsPro, bestColsPro, bestAPro, bestPPro, bestRPro,\
                        bestFPro = \
                            utility.feature_add_list_plus(scores=scores, models=models, inds=pop, cols=cols,
                                                          scoresA=scoresA, scoresP=scoresP, scoresR=scoresR,
                                                          scoresF=scoresF, bestScorePro=bestScorePro,
                                                          bestModelPro=bestModelPro, bestIndsPro=bestIndsPro,
                                                          bestColsPro=bestColsPro, bestAPro=bestAPro, bestPPro=bestPPro,
                                                          bestRPro=bestRPro, bestFPro=bestFPro)

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

                print_out = utility.new_my_print_feature(print_out=print_out, mode=mode, method=method,
                                                         mean=mean_scores, bestScore=bestScore, numCols=len(bestCols),
                                                         time_exe=time_instant, time_total=time_debut, entropy=entropy,
                                                         iter=generation, val=val)

                print_out = print_out + "\n"

                x1, y1, y2, yTps, yVars = utility.add_axis_vars(bestScore=bestScore, meanScore=mean_scores,
                                                                iter=generation,
                                                                vars=len(bestCols), time_debut=time_debut, x1=x1, y1=y1,
                                                                y2=y2, yTps=yTps, yVars=yVars)

                utility.plot_feature(x1=x1, y1=y1, y2=y2, yTps=yTps, yVars=yVars, n_pop=self.n_pop, n_gen=self.n_gen,
                                     heuristic="Evolution différentielle progressive binaire", folderName=folderName,
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

                self.write_res(folderName=folderName, probas=muCRs, y1=y1, y2=y2, colMax=colMax,
                               bestScorePro=bestScorePro, bestAPro=bestAPro, bestPPro=bestPPro, bestRPro=bestRPro,
                               bestFPro=bestFPro, bestModelPro=bestModelPro, bestScore=bestScore, bestScoreA=scoreAMax,
                               bestScoreP=scorePMax, bestScoreR=scoreRMax, bestScoreF=scoreFMax, bestModel=modelMax,
                               bestInd=indMax, debut=debut, out=print_out, yTps=yTps, yVars=yVars, method=method)

            besties, names, iters, times, names2, features, names3 = \
                utility.queues_put_feature(y2=y2, folderName=folderName, scoreMax=scoreMax, iter=generation, yTps=yTps,
                                       time=time_debut.total_seconds(), besties=besties, names=names, names2=names2,
                                       iters=iters, times=times, features=features, names3=names3, yVars=yVars,
                                       feature=len(bestCols))

    def init(self):

        print("##############################################")
        print("#EVOLUTION DIFFERENTIELLE PROGRESSIVE BINAIRE#")
        print("##############################################")
        print()

        besties, names, iters, times, names2, features, names3 = utility.queues_init()

        mods = self.list_exp

        n = len(mods)
        mods = [mods[i::n] for i in range(n)]

        # processes = []
        arglist = []
        for part in mods:
            arglist.append((part, besties, names, iters, times, names2, features, names3))

        pool = multiprocessing.Pool(processes=len(mods))
        pool.starmap(self.differential_evolution, arglist)
        pool.close()

        bestiesLst, namesLst, itersLst, timesLst, names2Lst, featuresLst, names3Lst =\
            utility.queues_get(n_process=len(mods), besties=besties, names=names, names2=names2, iters=iters,
                               times=times, features=features, names3=names3)

        pool.join()

        return utility.res(heuristic="Evolution différentielle progressive binaire", besties=bestiesLst, names=namesLst,
                           iters=itersLst, times=timesLst, names2=names2Lst, features=featuresLst, names3=names3Lst,
                           path=self.path2, dataset=self.dataset)


if __name__ == '__main__':

    diff = Differential(dataset="scene", target="Urban",
                        metric="accuracy",
                        list_exp=["lr", "svm", "gnb"],
                        pop=30, gen=20, learning_rate=0.1, alpha=2.0)
    diff.init()


