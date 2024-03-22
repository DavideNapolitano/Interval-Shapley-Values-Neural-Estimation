import numpy as np
import shap
import pandas as pd

class Monks():
    def __init__(self):
        self.dataset = "Monks"
        data = np.loadtxt('../data/monks-1.test', dtype=object, delimiter=' ') #SWAP
        self.X = data[:,2:-1]
        self.Y =data[:,1]

        data_test = np.loadtxt('../data/monks-1.train', dtype=object, delimiter=' ') #SWAP
        self.X_test = data_test[:,2:-1]
        self.Y_test = data_test[:,1]
        self.feature_names = ["a1","a2","a3","a4","a5","a6"]


    def get_data(self):
        return self.X, self.Y, self.X_test, self.Y_test, self.feature_names, self.dataset
    

class Census():
    def __init__(self):
        self.dataset="Census"
        self.X=shap.datasets.adult()[0].to_numpy()
        self.X=self.X.astype("object")
        self.Y=shap.datasets.adult()[1]
        self.feature_names = shap.datasets.adult()[0].columns.tolist()
        
    def get_data(self):
        return self.X, self.Y,None, None, self.feature_names, self.dataset
    
class Magic():
    def __init__(self):
        self.dataset="Magic"
        self.data = np.loadtxt('../data/magic04.data', dtype=object, delimiter=',')
        self.X=self.data[:,:-1]
        self.Y=self.data[:,-1]
        self.mapper={"h": 0, "g": 1}
        self.Y=[self.mapper[el] for el in self.Y]
        self.feature_names = ["fLength","fWidth","fSize","fConc","fConc1","fAsym","fM3Long", "fM3Trans", "fAlpha", "fDist"]
        
    def get_data(self):
        return self.X, self.Y,None, None, self.feature_names, self.dataset
    

class Wbcd():
    def __init__(self):
        self.dataset="WBCD"
        self.data = pd.read_csv('../data/breast-cancer-wisconsin.data', dtype=object, delimiter=',',header=None,na_values="?")
        self.data.columns=["ID","Clump Thickness","Uniformity of Cell Size","Uniformity of Cell Shape","Marginal Adhesion","Single Epithelial Cell Size","Bare Nuclei","Bland Chromatin","Normal Nucleoli","Mitoses","Class"]
        self.data=self.data.dropna(axis=0)
        self.Y=self.data["Class"]
        self.Y=self.Y.astype(np.int64)
        tmp=[]
        for el in self.Y:
            if el==2:
                tmp.append(1)
            else:
                tmp.append(0)
        self.Y=np.array(tmp)
        self.X=self.data.drop(columns=["ID","Class"]).to_numpy()
        self.feature_names = ["Clump Thickness","Uniformity of Cell Size","Uniformity of Cell Shape","Marginal Adhesion","Single Epithelial Cell Size","Bare Nuclei","Bland Chromatin","Normal Nucleoli","Mitoses"]
        
    def get_data(self):
        return self.X, self.Y,None, None, self.feature_names, self.dataset
    
class Diabetes():
    def __init__(self):
        self.dataset="Diabetes"
        self.data=np.loadtxt('../data/diabetes_data.txt', dtype=object, delimiter=',')
        self.X = self.data[:,:-1]
        self.Y = self.data[:,-1]
        self.mapper={"tested_negative": 0, "tested_positive": 1}
        self.Y =[self.mapper[el] for el in self.Y]
        self.feature_names=["preg","plas","pres","skin","insu","mass","pedi", "age"]

    def get_data(self):
        return self.X, self.Y,None, None, self.feature_names, self.dataset

class Heart():
    def __init__(self):
        self.dataset="Heart"
        self.data=pd.read_csv('../data/heart.csv', sep=",")
        self.data=self.data.drop("id",axis=1)
        self.mapper=mapper={"present":1,"absent":0}
        self.Y=np.array([self.mapper[el] for el in self.data["label"].values])
        self.X=self.data.drop("label",axis=1)
        self.feature_names=self.X.columns
        self.X=self.X.values

    def get_data(self):
        return self.X, self.Y,None, None, self.feature_names, self.dataset

class Credit():
    def __init__(self):
        self.dataset="Credit"
        self.data=pd.read_csv('../data/credit_card.csv', sep=",")
        self.data=self.data.drop("ID", axis=1)
        self.Y=self.data["DEFAULT_PAYMENT"].values
        self.X=self.data.drop("DEFAULT_PAYMENT",axis=1)
        self.feature_names=self.X.columns
        self.X=self.X.values

    def get_data(self):
        return self.X, self.Y,None, None, self.feature_names, self.dataset

class Mozilla():
    def __init__(self):
        self.dataset="Mozilla"
        self.data=np.loadtxt('../data/mozzilla_data.txt', dtype=object, delimiter=',')
        self.X=self.data[:,1:-1]
        self.Y=self.data[:,-1]
        self.feature_names=["start","end","event","size"]

    def get_data(self):
        return self.X, self.Y,None, None, self.feature_names, self.dataset

class Phoneme():
    def __init__(self):
        self.dataset="Phoneme"
        self.data=np.loadtxt('../data/phoneme.txt', dtype=object, delimiter=',')
        self.X=self.data[:,0:-1]
        self.Y=self.data[:,-1]
        self.mapper={'1': 0, '2': 1}
        self.Y =[self.mapper[el] for el in self.Y]
        self.feature_names=["v1","v2","v3","v4","v5"]

    def get_data(self):
        return self.X, self.Y,None, None, self.feature_names, self.dataset

class Bank():
    def __init__(self):
        self.dataset="Bank"
        self.data=np.loadtxt('../data/data_banknote_authentication.txt', dtype=object, delimiter=',')
        self.X=self.data[:,0:-1]
        self.Y=self.data[:,-1]
        self.feature_names=["variance","skewness","curtosis","entropy"]

    def get_data(self):
        return self.X, self.Y,None, None, self.feature_names, self.dataset