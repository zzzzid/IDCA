import time, math
import numpy as np
import random
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import norm
from .executor import LOG_FILE, write_log
import itertools

opt_seq = [0, 1]

class get_exchange(object):
    def __init__(self, incumbent):
        self.incumbent = incumbent  # fix values of impactful opts

    def to_next(self, opt_ids, l):
        """
        Flip selected less-impactful opt, then fix impactful optimization
        """
        ans = [0] * l
        for f in opt_ids:
            ans[f] = 1
        for f in self.incumbent:
            ans[f[0]] = f[1]
        return ans

class IDCA:
    def __init__(self, dim, get_objective_score, random, related_flags):
        """
        :param dim: number of compiler flags
        :param get_objective_score: obtain true speedup
        :param random: random parameter
        :param related_flags: program related flags for the target program
        """
        self.dim = dim
        self.get_objective_score = get_objective_score
        self.random = random
        self.related = related_flags
        self.critical = []

    def generate_random_conf(self, x):
        """
        Generation 0-1 mapping for disable-enable options
        """
        comb = bin(x).replace('0b', '')
        comb = '0' * (self.dim - len(comb)) + comb
        conf = []
        for k, s in enumerate(comb):
            if s == '1':
                conf.append(1)
            else:
                conf.append(0)
        return conf

    def get_ei(self, preds, eta):
        """
        :param preds:Sequences' speedup for EI
        :param eta:global best speedup
        :return:the EI for a sequence
        """
        preds = np.array(preds).transpose(1, 0)
        m = np.mean(preds, axis=1)
        s = np.std(preds, axis=1)

        def calculate_f(eta, m, s):
            z = (eta - m) / s
            return (eta - m) * norm.cdf(z) + s * norm.pdf(z)

        if np.any(s == 0.0):
            s_copy = np.copy(s)
            s[s_copy == 0.0] = 1.0
            f = calculate_f(eta, m, s)
            f[s_copy == 0.0] = 0.0
        else:
            f = calculate_f(eta, m, s)
        return f

    def get_ei_predict(self, model, now_best, wait_for_train):
        """
        :param model:RandomForest Model
        :param now_best:global best speedup
        :param wait_for_train:Sequences Set
        :return:the Sequences' EI
        """
        preds = []
        estimators = model.estimators_
        for e in estimators:
            preds.append(e.predict(np.array(wait_for_train)))
        acq_val_incumbent = self.get_ei(preds, now_best)
        return [[i, a] for a, i in zip(acq_val_incumbent, wait_for_train)]

    def runtime_predict(self, model, wait_for_train):
        """
        :param model:model:RandomForest Model
        :param wait_for_train:Sequences Set
        :return: the speedup of Sequences Set
        """
        preds_result = []
        estimators = model.estimators_
        t = 1
        for e in estimators:
            tmp = e.predict(np.array(wait_for_train))
            if t == 1:
                for i in range(len(tmp)):
                    preds_result.append(tmp)
                t = t + 1
            else:
                for i in range(len(tmp)):
                    preds_result[i] = preds_result[i] + tmp
                t = t + 1
            print(preds_result)
        for i in range(len(preds_result)):
            preds_result[i] = preds_result[i] / (t - 1)
        a = []

        for i in range(len(wait_for_train)):
            x = [wait_for_train[i], preds_result[0][i]]
            a.append(x)
        return a

    def getPrecision(self, model, seq):
        """
        :param model:
        :param seq:
        :return: The precision of a sequence and true speedup
        """
        true_running = self.get_objective_score(seq, k_iter=100086)
        estimators = model.estimators_
        res = []
        for e in estimators:
            tmp = e.predict(np.array(seq).reshape(1, -1))
            res.append(tmp)
        acc_predict = np.mean(res)
        return abs(true_running - acc_predict) / true_running, true_running

    def build_RF_by_CompTuner(self):
        """
        :return: model, initial_indep, initial_dep
        """
        inital_indep = []
        # randomly sample initial training instances
        while len(inital_indep) < 2:
            x = random.randint(0, 2 ** self.dim)
            initial_training_instance = self.generate_random_conf(x)
            if initial_training_instance not in inital_indep:
                inital_indep.append(initial_training_instance)
        inital_dep = [self.get_objective_score(indep, k_iter=0) for indep in inital_indep]
        all_acc = []
        model = RandomForestRegressor(random_state=self.random)
        model.fit(np.array(inital_indep), np.array(inital_dep))
        rec_size = 0
        while rec_size < 11:
            model = RandomForestRegressor(random_state=self.random)
            model.fit(np.array(inital_indep), np.array(inital_dep))
            global_best = max(inital_dep)
            estimators = model.estimators_
            if all_acc:
                all_acc = sorted(all_acc)
            neighbors = []
            for i in range(80000):
                x = random.randint(0, 2 ** self.dim)
                x = self.generate_random_conf(x)
                if x not in neighbors:
                    neighbors.append(x)
            pred = []
            for e in estimators:
                pred.append(e.predict(np.array(neighbors)))
            acq_val_incumbent = self.get_ei(pred, global_best)
            ei_for_current = [[i, a] for a, i in zip(acq_val_incumbent, neighbors)]
            merged_predicted_objectives = sorted(ei_for_current, key=lambda x: x[1], reverse=True)
            acc = 0
            flag = False
            for x in merged_predicted_objectives:
                if flag:
                    break
                if x[0] not in inital_indep:
                    inital_indep.append(x[0])
                    acc, lable = self.getPrecision(model, x[0])
                    inital_dep.append(lable)
                    all_acc.append(acc)
                    flag = True
            rec_size += 1

            if acc > 0.05:
                indx = self.selectByDistribution(merged_predicted_objectives)
                while merged_predicted_objectives[int(indx)][0] in inital_indep:
                    indx = self.selectByDistribution(merged_predicted_objectives)
                inital_indep.append(merged_predicted_objectives[int(indx)][0])
                acc, label = self.getPrecision(model, merged_predicted_objectives[int(indx)][0])
                inital_dep.append(label)
                all_acc.append(acc)
                rec_size += 1

            # if rec_size > 50 and np.mean(all_acc) < 0.04:
            #     break
        return model, inital_indep, inital_dep

    def selectByDistribution(self, merged_predicted_objectives):
        """
        :param merged_predicted_objectives: sorts sequences by EI value
        :return: selected sequence index
        """
        fitness = np.zeros(len(merged_predicted_objectives),)
        probabilityTotal = np.zeros(len(fitness))
        rec = 0.0000125
        proTmp = 0.0
        for i in range(len(fitness)):
            fitness[i] = random.uniform(0, (i+1) * rec)
            proTmp += fitness[i]
            probabilityTotal[i] = proTmp
        randomNumber = np.random.rand()
        result = 0
        for i in range(1, len(fitness)):
            if randomNumber < fitness[0]:
                result = 0
                break
            elif probabilityTotal[i - 1] < randomNumber <= probabilityTotal[i]:
                result = i
        return result

    def get_critical_flags(self, model):
        candidate_seq = []
        candidate_per = []
        while len(candidate_seq) < 30000:
            x = random.randint(0, 2 ** self.dim)
            initial_training_instance = self.generate_random_conf(x)
            if initial_training_instance not in candidate_seq:
                candidate_seq.append(initial_training_instance)
        all_per = self.runtime_predict(model,candidate_seq)
        candidate_per = [all[1] for all in all_per]
        pos_seq = [0] * len(self.related)    
        for idx in range(len(self.related)):
            new_candidate = []
            for j in candidate_seq:
                seq = candidate_seq[j][self.related[idx]] ^ 0
                new_candidate.append(seq)
            new_per = [all[1] for all in self.runtime_predict(model,new_candidate)]
            for l in len(range(new_candidate)):
                if (candidate_per[l] > new_per[l] and new_candidate[l][self.related[idx]] == 1) or (candidate_per[l] < new_per[l] and new_candidate[l][self.related[idx]] == 0):
                    pos_seq[idx] -= 1
                else:
                    pos_seq[idx] += 1
        sort_pos = sorted(enumerate(pos_seq), key=lambda x: x[1], reverse=True)
        critical_flag_idx = []
        for i in range(10):
            critical_flag_idx.append(self.related[sort_pos[i][0]])
        return critical_flag_idx
    
    def searchBycritical(self, model):
        # model, inital_indep, inital_dep = self.build_RF_by_CompTuner()
        critical_flag = self.get_critical_flags(model)
        permutations = list(itertools.product([0, 1], repeat=10))
        seqs = []
        while len(seqs) < 1024:
            x = random.randint(0, 2 ** self.dim)
            initial_training_instance = self.generate_random_conf(x)
            if initial_training_instance not in seqs:
                seqs.append(initial_training_instance)
        for i in range(len(permutations)):
            for idx in range(len(critical_flag)):
                seqs[i][critical_flag[idx]] = permutations[i][idx]
        return seqs

    def run(self):
        begin = time.time()
        """
        build model and get data set
        """
        ts = []
        model, inital_indep, inital_dep = self.build_RF_by_CompTuner()
        time_set_up = 6000
        best_per = max(inital_dep)
        best_indep = inital_indep[inital_dep.index(best_per)]
        ts.append(time.time() - begin)
        # model, inital_indep, inital_dep = self.build_RF_by_BOCA() # BOCA build method
        while (ts[-1] < time_set_up):
            seq = self.searchBycritical(model)
            result = self.runtime_predict(model,seq)
            sorted_result = sorted(result, key=lambda x: x[1])
            if sorted_result[0][1] > best_per:
                best_per = sorted_result[0][1]
                best_indep = sorted_result[0][0]
        best_result = self.get_objective_score(best_indep, k_iter=0)
        ts.append(time.time() - begin)
        ss = '{}: cur-best {}, cur-best-seq {}'.format(str(round(ts[-1])), str(best_result), str(best_indep))
        write_log(ss, LOG_FILE)
        # if (time.time() - begin) > time_set_up:
        #     break
