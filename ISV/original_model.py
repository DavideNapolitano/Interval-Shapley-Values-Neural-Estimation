import numpy as np
from scipy.stats import t

class OriginalModel:
    def __init__(self, arch, cl=0.95):
        self.model=arch #TRAINED MODEL
        self.confidence_level=cl
        self.runs=arch.n_estimators
        self.estimators=arch.estimators_

    def evaluate_ci(self, x):
        t_sh = t.ppf((self.confidence_level + 1) / 2, df=self.runs - 1)  # threshold for t_student
        ave = x.mean()  # average
        stddev = x.std(ddof=1)  # std dev
        ci = t_sh * stddev / np.sqrt(self.runs)  # confidence interval half width
        return ave, ci

    def get_pred_ci(self, x):
        res_p=[]
        res_ci=[]
        
        idx=0
        for el in x:
            prob_p=[]
            prob_n=[]
            for inner_model in self.estimators:
                p=inner_model.predict_proba([el])[0]
                prob_n.append(p[0])
                prob_p.append(p[1])
                
            p_n,ci_n= self.evaluate_ci(np.array(prob_n))
            p_p,ci_p= self.evaluate_ci(np.array(prob_p))
            p_tot=[p_n,p_p]
            ci_tot=[round(ci_n,6),round(ci_p,6)]
            #print(idx, p_tot, ci_tot)
            idx+=1
            res_p.append(p_tot)
            res_ci.append(ci_tot)
            
        return res_p, res_ci
    
class OriginalModelVV:

    def __init__(self, arch, cl=0.95):
        self.model=arch #TRAINED MODEL
        self.confidence_level=cl
        self.runs=arch.n_estimators
        self.estimators=arch.estimators_

    def evaluate_ci(self, x):
        t_sh = t.ppf((self.confidence_level + 1) / 2, df=self.runs - 1)  # threshold for t_student
        ave = x.mean()  # average
        stddev = x.std(ddof=1)  # std dev
        ci = t_sh * stddev / np.sqrt(self.runs)  # confidence interval half width
        return ave, ci

    def get_pred_VV(self, x):
        res_v1=[]
        res_v2=[]
        
        idx=0
        for el in x:
            prob_p=[]
            prob_n=[]
            for inner_model in self.estimators:
                p=inner_model.predict_proba([el])[0]
                prob_n.append(p[0])
                prob_p.append(p[1])
                
            p_n,ci_n= self.evaluate_ci(np.array(prob_n))
            p_p,ci_p= self.evaluate_ci(np.array(prob_p))
            l=[np.max([0,p_n-ci_n]), np.max([0,p_p-ci_p])]
            u=[np.min([1,p_n+ci_n]), np.min([p_p+ci_p])]
            
            v1=[l[0],u[1]]
            v2=[l[1],u[0]]
            idx+=1
            res_v1.append(v1)
            res_v2.append(v2)
            
        return res_v1, res_v2