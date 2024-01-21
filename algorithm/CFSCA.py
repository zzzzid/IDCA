import os, sys, random, time, copy, subprocess, itertools, math
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import norm

all_flags = ['-falign-functions', '-falign-jumps', '-falign-labels', '-falign-loops', '-fasynchronous-unwind-tables', '-fbit-tests', 
             '-fbranch-count-reg', '-fcaller-saves', '-fcode-hoisting', '-fcombine-stack-adjustments', '-fcompare-elim', '-fcprop-registers',
             '-fcrossjumping', '-fcse-follow-jumps', '-fdefer-pop', '-fdevirtualize', '-fdevirtualize-speculatively', '-fdse',
             '-fexpensive-optimizations', '-fforward-propagate', '-ffunction-cse', '-fgcse', '-fgcse-after-reload', 
             '-fguess-branch-probability', '-fhoist-adjacent-loads', '-fif-conversion', '-fif-conversion2', '-findirect-inlining', '-finline', 
             '-finline-functions', '-finline-functions-called-once', '-finline-small-functions', '-fipa-bit-cp', 
             '-fipa-cp', '-fipa-cp-clone', '-fipa-icf', '-fipa-icf-functions', '-fipa-icf-variables', '-fipa-modref',
             '-fipa-profile', '-fipa-pure-const', '-fipa-ra', '-fipa-reference', '-fipa-reference-addressable', 
             '-fipa-sra', '-fipa-vrp', '-fira-share-save-slots', '-fisolate-erroneous-paths-dereference', '-fjump-tables', 
             '-floop-interchange', '-floop-unroll-and-jam', '-flra-remat', '-fmove-loop-invariants', '-fomit-frame-pointer', 
             '-foptimize-sibling-calls', '-foptimize-strlen', '-fpartial-inlining', '-fpeel-loops', '-fpeephole2', 
             '-fpredictive-commoning', '-fprintf-return-value', '-free', '-frename-registers', '-freorder-blocks',
            '-freorder-blocks-and-partition', '-freorder-functions', '-frerun-cse-after-loop', '-fsched-dep-count-heuristic', 
            '-fsched-interblock', '-fsched-rank-heuristic', '-fsched-spec-insn-heuristic', '-fschedule-fusion',
            '-fschedule-insns2', '-fshrink-wrap', '-fsigned-zeros', '-fsplit-loops', '-fsplit-paths', 
            '-fsplit-wide-types', '-fssa-phiopt', '-fstore-merging', '-fstrict-aliasing', '-fthread-jumps', 
            '-ftoplevel-reorder', '-ftree-bit-ccp', '-ftree-builtin-call-dce', '-ftree-ccp', '-ftree-ch', 
            '-ftree-coalesce-vars', '-ftree-copy-prop', '-ftree-dce', '-ftree-dominator-opts', '-ftree-dse', 
            '-ftree-fre', '-ftree-loop-distribute-patterns', '-ftree-loop-distribution', 
            '-ftree-loop-im', '-ftree-loop-optimize', '-ftree-loop-vectorize', '-ftree-partial-pre', 
            '-ftree-pre', '-ftree-pta', '-ftree-scev-cprop', '-ftree-sink', '-ftree-slp-vectorize', '-ftree-slsr', 
            '-ftree-sra', '-ftree-switch-conversion', '-ftree-tail-merge', 
            '-ftree-ter', '-ftree-vrp', '-funroll-completely-grow-size', '-funswitch-loops', '-fvar-tracking', '-fversion-loops-for-strides']


LOG_DIR = 'log' + os.sep
LOG_FILE = LOG_DIR + 'cfasc_recordc1.log'
ERROR_FILE = LOG_DIR + 'err.log'

def write_log(ss, file):
    log = open(file, 'a')
    log.write(ss + '\n')
    log.flush()
    log.close()
    
def execute_terminal_command(command):
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("命令执行成功！")
            if result.stdout:
                print("命令输出：")
                print(result.stdout)
        else:
            print("命令执行失败。")
            if result.stderr:
                print("错误输出：")
                print(result.stderr)
    except Exception as e:
        print("执行命令时出现错误：", str(e))

def get_objective_score(independent, time_o3_c, k_iter):
    opt = ''
    for i in range(len(independent)):
        if independent[i]:
            opt = opt + all_flags[i] + ' '
        else:
            negated_flag_name = all_flags[i].replace("-f", "-fno-", 1)
            opt = opt + negated_flag_name + ' '
    time_start = time.time()
    command = "gcc -O2 " + opt + " -c /home/zmx/CFSCA/cBench_V1.1/automotive_bitcount/src/*.c"
    execute_terminal_command(command)
    command2 = "gcc -o a.out -O2 " + opt + " -lm *.o"
    execute_terminal_command(command2)
    command3 = "time ./a.out 1125000"
    execute_terminal_command(command3)
    cmd4 = 'rm -rf *.o *.I *.s a.out'
    execute_terminal_command(cmd4)
    time_end = time.time()  
    time_c = time_end - time_start   #运行所花时间
    print(time_o3_c /time_c)
    op_str = "iteration:{} speedup:{}".format(str(k_iter), str(time_o3_c /time_c))
    write_log(op_str, LOG_FILE)
    return (time_o3_c /time_c)

        
time_tem = []
class CFSCA:
    def __init__(self, dim, get_objective_score, seed, related_flags, std):
        """
        :param dim: number of compiler flags
        :param get_objective_score: obtain true speedup
        :param random: random parameter
        :param related_flags: program related flags for the target program
        """
        self.dim = dim
        self.get_objective_score = get_objective_score
        self.seed = seed
        self.related = related_flags
        self.stdtime = std
        self.critical = []
        self.global_best_per = 0.0
        self.global_best_seq = []

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
        estimators = model.estimators_
        sum_of_predictions = np.zeros(len(wait_for_train))
        for tree in estimators:
            predictions = tree.predict(wait_for_train)
            sum_of_predictions += predictions
        a = []
        average_prediction = sum_of_predictions / len(estimators)
        for i in range(len(wait_for_train)):
            x = [wait_for_train[i], average_prediction[i]]
            a.append(x)
        return a
    
    def getPrecision(self, model, seq):
        """
        :param model:
        :param seq:
        :return: The precision of a sequence and true speedup
        """
        true_running = self.get_objective_score(seq, self.stdtime,k_iter=100086)
        estimators = model.estimators_
        res = []
        for e in estimators:
            tmp = e.predict(np.array(seq).reshape(1, -1))
            res.append(tmp)
        acc_predict = np.mean(res)
        return abs(true_running - acc_predict) / true_running, true_running
    
    def selectByDistribution(self, merged_predicted_objectives):
        """
        Assign probabilities for different flag combinations
        """
        # sequences = [seq for seq, per in merged_predicted_objectives]
        diffs = [abs(perf - merged_predicted_objectives[0][1]) for seq, perf in merged_predicted_objectives]
        diffs_sum = sum(diffs)
        probabilities = [diff / diffs_sum for diff in diffs]
        index = list(range(len(diffs)))
        idx = np.random.choice(index, p=probabilities)
        return idx
    
    def build_RF_by_CompTuner(self):
        """
        :return: model, initial_indep, initial_dep
        """
        inital_indep = []
        time_begin = time.time()
        # randomly sample initial training instances
        while len(inital_indep) < 2:
            x = random.randint(0, 2 ** self.dim)
            initial_training_instance = self.generate_random_conf(x)
            if initial_training_instance not in inital_indep:
                inital_indep.append(initial_training_instance)
        inital_dep = [self.get_objective_score(indep, self.stdtime, k_iter=0) for indep in inital_indep]
        
        all_acc = []
        time_tem.append(time.time() - time_begin)
        model = RandomForestRegressor(random_state=self.seed)
        model.fit(np.array(inital_indep), np.array(inital_dep))
        rec_size = 2
        
        while rec_size < 11:
            model = RandomForestRegressor(random_state=self.seed)
            model.fit(np.array(inital_indep), np.array(inital_dep))
            global_best = max(inital_dep)
            estimators = model.estimators_
            if all_acc:
                all_acc = sorted(all_acc)
            neighbors = []
            for i in range(30000):
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
            time_tem.append(time.time() - time_begin)
            ss = '{}: best_seq {}, best_per {}'.format(str(round(time_tem[-1])), str(max(inital_dep)), str(inital_indep[inital_dep.index(max(inital_dep))]))
            write_log(ss, LOG_FILE)
        self.global_best_per = max(inital_dep)
        self.global_best_seq = inital_indep[inital_dep.index(max(inital_dep))]
        return model
    
    def get_critical_flags(self, model):
        candidate_seq = []
        candidate_per = []
        
        while len(candidate_seq) < 30000:
            x = random.randint(0, 2 ** self.dim)
            initial_training_instance = self.generate_random_conf(x)
            if initial_training_instance not in candidate_seq:
                candidate_seq.append(initial_training_instance)
        begin = time.time()
        all_per = self.runtime_predict(model,candidate_seq)
        candidate_per = [all[1] for all in all_per]
        pos_seq = [0] * len(self.related)    
        now_best = max(candidate_per)
        now_best_seq = candidate_seq[candidate_per.index(now_best)]
        before_time = time_tem[-1]
        time_tem.append(time.time() - begin + before_time)
        if self.global_best_per < now_best:
            self.global_best_per = now_best
            self.global_best_seq = now_best_seq
        ss = '{}: best_seq {}, best_per {}'.format(str(round(time_tem[-1])), str(self.global_best_per), str(self.global_best_seq))
        write_log(ss, LOG_FILE)

        for idx in range(len(self.related)):
            new_candidate = []
            for j in range(len(candidate_seq)):
                seq = copy.deepcopy(candidate_seq[j])
                seq[self.related[idx]] = 1 - seq[self.related[idx]]
                new_candidate.append(seq)
            new_per = [all[1] for all in self.runtime_predict(model,new_candidate)]
            if max(new_per) > self.global_best_per:
                self.global_best_per = max(new_per)
                self.global_best_seq = new_candidate[new_per.index(now_best)]

            for l in range(len(new_candidate)):
                if (candidate_per[l] > new_per[l] and new_candidate[l][self.related[idx]] == 1) or (candidate_per[l] < new_per[l] and new_candidate[l][self.related[idx]] == 0):
                    pos_seq[idx] -= 1
                else:
                    pos_seq[idx] += 1
            time_tem.append(time.time() - begin + before_time)
            ss = '{}: best_seq {}, best_per {}'.format(str(round(time_tem[-1])), str(self.global_best_per), str(self.global_best_seq))
            write_log(ss, LOG_FILE)

        sort_pos = sorted(enumerate(pos_seq), key=lambda x: x[1], reverse=True)
        critical_flag_idx = []
        for i in range(10):
            critical_flag_idx.append(self.related[sort_pos[i][0]])
        return critical_flag_idx
    
    def searchBycritical(self, critical_flag):
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
        begin_all = time.time()
        """
        build model and get data set
        """
        model = self.build_RF_by_CompTuner()
        time_set_up = 6000
        all_before = time_tem[-1]
        critical_flag = self.get_critical_flags(model)
        begin_all = time.time()
        while (time_tem[-1] < time_set_up):
            seq = self.searchBycritical(critical_flag)
            result = self.runtime_predict(model,seq)
            sorted_result = sorted(result, key=lambda x: x[1])
            if sorted_result[0][1] > self.global_best_per:
                self.global_best_per = sorted_result[0][1]
                self.global_best_seq = sorted_result[0][0]
            time_tem.append(time.time() - begin_all + all_before)
            ss = '{}: cur-best {}, cur-best-seq {}'.format(str(round(time_tem[-1])), str(self.global_best_per), str(self.global_best_seq))
            write_log(ss, LOG_FILE)
        best_result = self.get_objective_score(self.global_best_seq, self.stdtime, k_iter=0)
        ss = '{}: cur-best {}, cur-best-seq {}'.format(str(round(time_tem[-1])), str(best_result), str(self.global_best_seq))
    
 