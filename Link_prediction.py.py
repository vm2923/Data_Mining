import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA,TruncatedSVD
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn import tree
from sklearn import model_selection
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier

"""BASIC GRAPH"""

class Graph(object):

    def __init__(self):
        '''Create Graph Object '''
        self.graph_dict = {}
    def nodes(self):
        """ return name of vertices"""
        return list(self.graph_dict.keys())

    def edges(self):
        """ return connected edges """
        return self.find_edges()

    def add_node(self, node):
        """Add node to graph"""
        if node not in self.graph_dict:
            self.graph_dict[node] = []

    def add_edge(self,tail,head):
        """ add edge between nodes"""
        self.add_node(tail)
        self.add_node(head)
        if tail!=head:
            self.graph_dict[tail].append(head)
            self.graph_dict[head].append(tail)

    def find_edges(self):
        """find the edges"""
        edges = {}
        for nodes in self.graph_dict:
            edges[nodes]=[]
            for link in self.graph_dict[nodes]:
                if {nodes,link} not in edges[nodes] and nodes != link:
                    edges[nodes].append({nodes,link})
        return edges
    def nodeN(self, node):
        """return neighbour nodes"""
        return set(self.graph_dict[node])
    def neigh(self, node):
        """return neighbour nodes"""
        return self.graph_dict[node]
    def CN(self,nodes,link):
        '''common neighbour'''
        if nodes != link:
            score=len(set(graph.nodeN(nodes)).intersection(set(graph.nodeN(link))))
        return score
        
    def JC(self,nodes,link):
        '''Jacard Coefficient'''
        common=0
        if nodes != link:
            common=len(set(graph.nodeN(nodes)).intersection(set(graph.nodeN(link))))
            total=len(set(graph.nodeN(nodes)).union(set(graph.nodeN(link))))
            if total!=0:
                common=common/total
            else:
                common=0
        return common     
    
    def TN(self,nodes,link):
        '''Total neighbour'''
        common=0
        if nodes != link:
            common=len(set(graph.nodeN(nodes)).union(set(graph.nodeN(link))))
        return common 
    def ND(self,nodes,link):
        '''Neighbour Distance'''
        import math
        if nodes != link:
            common=len(set(graph.nodeN(nodes)).intersection(set(graph.nodeN(link))))
            total=len(set(graph.nodeN(nodes)))*len((set(graph.nodeN(link))))
            total=math.sqrt(total)
            if total!=0:
                common=common/total
            else:
                common=0
        return common 
    def AA(self,nodes,link):
        common=0
        if nodes != link:
            v=set(graph.nodeN(nodes)).intersection(set(graph.nodeN(link)))
            for i in v:
                deg=len(self.nodeN(i))
                common=1.0/math.log(deg)+common
        return common 

    def PA(self,nodes,link):
        '''Preferential Attachment (PA)'''
        common=0
        if nodes != link:
            common=len(set(graph.nodeN(nodes)))*len(set(graph.nodeN(link)))
        return common
    
    def RA(self,nodes,link):
        '''Resource Allocation(RA)'''
        common=0
        if nodes != link:
            v=set(graph.nodeN(nodes)).intersection(set(graph.nodeN(link)))
            for i in v:
                deg=len(self.nodeN(i))
                common=1.0/deg+common
        return common

    def katz(self,nodes,link):
        '''KATZ similarity'''
        maxiter=2
        beta=0.1
        common=0
        if nodes != link:
            length = 1
            neighbors = self.neigh(nodes)
            while length <= maxiter:
                path = neighbors.count(link)
                if path > 0:
                    common += (beta**length)*path
                neighbornext = []
                length += 1
                if(length <= maxiter):
                    for m in neighbors:
                        neighbornext += self.neigh(m)
                        neighbors = neighbornext
        return common
    def PR(self,nodes,link):
        '''Page Rank(PR)'''
        import networkx as nx
        common=0
        per={}
        for i in G.nodes():
            if(i==nodes):
                per[i]=1
            else:
                per[i]=0
        if nodes != link:
            rp=nx.pagerank(G,personalization=per)
        return rp
    
if __name__ == "__main__":
    import math
    import networkx as nx
    import csv
    graph = Graph()
    G = nx.Graph()
    with open('edges.csv','r')as f:
        data = csv.reader(f)
        for row in data:
            i=row[0].split("\t")
            graph.add_edge(i[0],i[1])
            G.add_edge(i[0],i[1])
    f.close()
    #graph.AA(i,j)
    #graph.CN(i,j)
    #graph.JC(i,j)
    #graph.PA(i,j)
    #graph.RA(i,j)
    #graph.PR(i,j)
    #graph.katz(i,j)


    
"""THEME A"""
    """calculating Features""" 
    with open('testdata.csv','r')as f:
        edges=[]
        p=None
        with open('katz.csv','w')as fp:
            data = csv.reader(f)
            for i in data:
                if(p!=i[0]):
                    if(i[0]!=i[1]):
                        p=i[0]
                        k=graph.katz(i[0],i[1]) # change this for different Feature
                        fp.write("%s,%s,%s,%s\n"%(i[0],i[1],i[2],k[(i[1])]))
                else:
                    fp.write("%s,%s,%s,%s\n"%(i[0],i[1],i[2],k[i[1]]))
    fp.close()
    f.close()

def roc(score,y,name):
    import matplotlib.pyplot as plt
    import numpy as np
    fpr = []
    tpr = []
    pr=[]
    re=[]
    avg=sum(score) / len(score)
    low=min(score)
    high=max(score)
    thresholds = np.arange(-.01, high+avg/10,avg/300)
    positive = sum(y)
    negative = len(y) - positive
    for thresh in thresholds:
        FP=0
        TP=0
        FN=0
        for i in range(len(score)):
            if (score[i] > thresh):
                if y[i] == 1:
                    TP = TP + 1
                if y[i] == 0:
                    FP = FP + 1
            else:
                if y[i]==1:
                    FN=FN+1
        if(TP+FP>0):            
            pr.append(TP/(TP+FP))
        if(TP+FN>0): 
            re.append(TP/(TP+FN))
        fpr.append(FP/float(negative))
        tpr.append(TP/float(positive))
    from sklearn.metrics import auc
    sr=auc(fpr,tpr)
    a=int(len(pr)/2)
    plot_roc_curve(fpr, tpr,name,sr,pr[a],re[-1])
    return re,pr


def plot_roc_curve(score,y,name):
    auc = roc_auc_score(y, score)
    fpr, tpr, _ = roc_curve(y, score)
    plt.figure(figsize=(10,10))
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Foursquare Restaurant '+ name+' AUC Score:'+str(auc))
    plt.legend()
    plt.savefig(name)
    plt.show()

    score=[]
    y=[]
    import csv
    with open('ND.csv','r')as f:
        data = csv.reader(f)
        for i in data:
            if(i[0]!=i[1]):
                score.append(float(i[3]))
                if(i[2]=='Yes'):
                    y.append(1)
                else:
                    y.append(0)
    name='Neighbour Distance'                
    name1=name+" PR"
    plot_roc_curve(score,y,name)   
    plot_pr(score,y,name1)
    f.close()
    


def plot_pr(score,y,name):
    lr_precision, lr_recall, _ = precision_recall_curve(y, score)
    auc = roc_auc_score(y, score)
    pr=sum(lr_precision)/len(lr_precision)
    re=sum(lr_recall)/len(lr_recall)
    plt.figure(figsize=(10,10))
    plt.plot(lr_recall,lr_precision, color='orange', label='ROC')
    plt.plot([0, 1], [1, 0], color='darkblue', linestyle='--')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Foursquare Restaurant '+ name+' AUC Score:'+str(auc))
    plt.text(.2,.2,str('Precision:'+str(pr)+'\n Recall:'+str(re)))
    plt.legend()
    plt.savefig(name)
    plt.show()














"""theme B"""
"""theme B .1"""
edges={}
    import csv
    zero=[0,0,0,0,0,0,0,0,0]
    with open('PA.csv','r')as f:
        data = csv.reader(f)
        for i in data:
            if(i[0]!=i[1]):
                edges[(i[0],i[1])]=[]
                if(i[2]=='Yes'):   
                    edges[(i[0],i[1])].append(1)
                    edges[(i[0],i[1])]=edges[(i[0],i[1])]+zero
                       
                else:
                    edges[(i[0],i[1])].append(0)
                    edges[(i[0],i[1])]=edges[(i[0],i[1])]+zero
                
    f.close()
    import csv
    with open('TN.csv','r')as f:
        data = csv.reader(f)
        for i in data:
            if(i[0]!=i[1]):
                edges[(i[0],i[1])][9]=float(i[3])
                
    f.close()
    

with open('features.csv','w')as fp:
    fp.write("%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n"%('source','dest','link','AA','CN','JC','katz','ND','PA','PR','RA','TN'))
    for i in edges:
        fp.write("%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n"%(i[0],i[1],edges[i][0],edges[i][1],edges[i][2],edges[i][3],edges[i][4],edges[i][5],edges[i][6],edges[i][7],edges[i][8],edges[i][9]))
fp.close()


"""Loading dataframe with features csv file""""
df = pd.read_csv("features.csv")
df.head()
aa=df['AA']
cn=df['CN']
jc=df['JC']
katz=df['katz']
nd=df['ND']
pa=df['PA']
pr=df['PR']
ra=df['RA']
tn=df['TN']
target = df['link']
data=[]
for i in range(len(aa)):
    data.append([0,0,0,0,0,0,0,0,0])
    data[i][0]=aa[i]
    data[i][1]=cn[i]
    data[i][2]=jc[i]
    data[i][3]=katz[i]
    data[i][4]=nd[i]
    data[i][5]=pa[i]
    data[i][6]=pr[i]
    data[i][7]=ra[i]
    data[i][8]=tn[i]

"""Splitting data into trainning anf test dataset"""
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=4)    



"""Gaussian Naive Bayes"""
import random
import math
 
def sD(D, sR):
    trainSize = int(len(D) * sR)
    trainSet = []
    copy = list(D)
    while len(trainSet) < trainSize:
           index = random.randrange(len(copy))
        trainSet.append(copy.pop(index))
    return [trainSet, copy]
 
def separateByClass(D):
    separated = {}
    for i in range(len(D)):
        vector = D[i]
        if (vector[-1] not in separated):
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)
    return separated
 
def mean(numbers):
    return sum(numbers)/float(len(numbers))
 
def stdev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
    return math.sqrt(variance)
 
def smrz(D):
    srs = [(mean(att), stdev(att)) for att in zip(*D)]
    del srs[-1]
    return srs
 
def SBC(D):
    separated = separateByClass(D)
    srs = {}
    for classValue, instances in separated.items():
        srs[classValue] = smrz(instances)
    return srs

def calculateprbty(x, mean, stdev):
    exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
    return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent
 
def calculateClassprb(srs, inputVector):
    prb = {}
    for classValue, classsrs in srs.items():
        prb[classValue] = 1
        for i in range(len(classsrs)):
            mean, stdev = classsrs[i]
            x = inputVector[i]
            prb[classValue] *= calculateprbty(x, mean, stdev)
    return prb
 
def predict(srs, inputVector):
    prb = calculateClassprb(srs, inputVector)
    bestLabel, bestProb = None, -1
    for classValue, prbty in prb.items():
        if bestLabel is None or prbty > bestProb:
            bestProb = prbty
            bestLabel = classValue
    return bestLabel
 
def getpredtns(srs, testSet):
    predtns = []
    for i in range(len(testSet)):
        result = predict(srs, testSet[i])
        predtns.append(result)
    return predtns
 
def getacc(testSet, predtns):
    correct = 0
    for i in range(len(testSet)):
        if testSet[i][-1] == predtns[i]:
            correct += 1
    return (correct/float(len(testSet))) * 100.0

def main():
    import pandas as pd
    df = pd.read_csv("features.csv")
    df.head()
    aa=df['AA']
    cn=df['CN']
    jc=df['JC']
    katz=df['katz']
    nd=df['ND']
    pa=df['PA']
    pr=df['PR']
    ra=df['RA']
    tn=df['TN']
    ln = df['link']
    data=[]
    for i in range(len(aa)):
        data.append([0,0,0,0,0,0,0,0,0,0])
        data[i][0]=aa[i]
        data[i][1]=cn[i]
        data[i][2]=jc[i]
        data[i][3]=katz[i]
        data[i][4]=nd[i]
        data[i][5]=pa[i]
        data[i][6]=pr[i]
        data[i][7]=ra[i]
        data[i][8]=tn[i]
        data[i][9]=ln[i]
    sR = .5
    D = data
    trainingSet, testSet = sD(D, sR)
    # prepare model
    srs = SBC(trainingSet)
    # test model
    predtns = getpredtns(srs, testSet)
    acc = getacc(testSet, predtns)
    print(acc/100)
 
main()
"""END OF nAIVE bAYES IMPLEMENTATION"""



"""SCM Classifier"""
# import support vector classifier 
from sklearn.svm import SVC # "Support Vector Classifier" 
model2 = SVC(C=10, cache_size=200, class_weight='balanced', coef0=0.0,
  decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
  max_iter=-1, probability=True, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
model2.fit(x_train,y_train)
pred2=model2.predict(x_test)
model2.score(x_test,y_test)



"""Plotting bar graph for fold wise accuracy"""
model = GaussianNB()
model =tree.DecisionTreeClassifier(criterion = "entropy")
# AdaBoost Classification
model = AdaBoostClassifier(n_estimators=30, random_state=2)
model=BaggingClassifier(n_estimators=10, random_state=0)
model=RandomForestClassifier(n_estimators=3, random_state=0)
results = model_selection.cross_val_score(model2,data, target, cv=5)
print(results)
print(results.mean())
width = 0.35       # the width of the bars: can also be len(x) sequence
fold = np.arange(1, 6, 1)
p1 = plt.bar(fold, results, width)
plt.title('Fold wise Accuracy Score')
plt.ylabel('Accuracy score')
plt.xlabel('Fold number')
plt.savefig('FoldScore')
plt.show()


"""ROC Curve for all the models""""

def plot_roc_curve(fpr, tpr,name,auc):
    from sklearn.metrics import roc_curve
    from matplotlib import pyplot as plt
    plt.figure(figsize=(10,10))
    plt.plot(fpr, tpr, color='orange', label=name)
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--',label='No Skill')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('FourSquare Restaurant '+ name+' AUC Score:'+str(auc))
    plt.legend()
    plt.savefig(name)
    plt.show()
    
# roc curve and auc

trainX, testX, trainy, testy = train_test_split(data, target, test_size=0.2,random_state=2000)
# generate a no skill prediction (majority class)
ns_probs = [0 for _ in range(len(testy))]
model = tree.DecisionTreeClassifier(criterion = "entropy")
# AdaBoost Classification
model = AdaBoostClassifier(n_estimators=30, random_state=2)
model = GaussianNB()
model=BaggingClassifier(n_estimators=9, random_state=0)
model=RandomForestClassifier()
model = SVC(C=10, cache_size=200, class_weight='balanced', coef0=0.0,
  decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
  max_iter=-1, probability=True, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
model.fit(trainX, trainy)
# predict probabilities
lr_probs = model.predict_proba(testX)
# keep probabilities for the positive outcome only
lr_probs = lr_probs[:, 1]
# calculate scores
ns_auc = roc_auc_score(testy, ns_probs)
lr_auc = roc_auc_score(testy, lr_probs)
# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Classifier: ROC AUC=%.3f' % (lr_auc))
# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(testy, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(testy, lr_probs)
plot_roc_curve(lr_fpr, lr_tpr,'Classifier',lr_auc)



"""Precision Recall Curve and AUC"""
    
def plot_roc_curve_PR(a,b,fpr, tpr,name,auc,f1):
    from sklearn.metrics import roc_curve
    from matplotlib import pyplot as plt
    plt.figure(figsize=(10,10))
    plt.plot(fpr, tpr, color='orange', label=name)
    plt.plot(a,b, color='darkblue', linestyle='--',label='No Skill')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('FourSquare Restaurantc '+ name+' AUC Score:'+str(auc)+'\nF Score:'+str(f1))
    plt.legend()
    plt.savefig(name)
    plt.show()

# generate 2 class dataset
trainX, testX, trainy, testy = train_test_split(data,target, test_size=0.2, random_state=2)
# generate a no skill prediction (majority class)
no_skill_probs = [0 for _ in range(len(testy))]
# fit a model
model = tree.DecisionTreeClassifier(criterion = "entropy")
model = GaussianNB()
AdaBoost Classification
model = AdaBoostClassifier(n_estimators=30, random_state=2)
model=BaggingClassifier(n_estimators=9, random_state=0)
model = SVC(C=10, cache_size=200, class_weight='balanced', coef0=0.0,
  decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
  max_iter=-1, probability=True, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
model=RandomForestClassifier()
model.fit(trainX, trainy)
# predict probabilities
lr_probs = model.predict_proba(testX)
# keep probabilities for the positive outcome only
lr_probs = lr_probs[:, 1]
# predict class values
yhat = model.predict(testX)
# calculate precision and recall for each threshold
ns_precision, ns_recall, _ = precision_recall_curve(testy, no_skill_probs)
lr_precision, lr_recall, _ = precision_recall_curve(testy, lr_probs)
# calculate scores
ns_f1, ns_auc = f1_score(testy, no_skill_probs), auc(ns_recall, ns_precision)
lr_f1, lr_auc = f1_score(testy, yhat), auc(lr_recall, lr_precision)
print(sum(lr_recall)/len(lr_recall), sum(lr_precision)/len(lr_precision))
# summarize scores
print('No Skill: f1=%.3f auc=%.3f' % (ns_f1, ns_auc))
print('Classifier: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))
plot_roc_curve_PR(ns_recall, ns_precision,lr_recall, lr_precision,'Classifier_PR',lr_auc,lr_f1)


"""Gradient Boosting technique to improve descision tree Classifier"""
#GradientBoosting
model = tree.DecisionTreeClassifier(criterion = "entropy")
model.fit(x_train, y_train)
#pred=model.predict(x_test)
model.score(x_test,y_test)

lr_list = [0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1]
score_list = []

for learning_rate in lr_list:
    gb_clf = GradientBoostingClassifier(n_estimators=30, learning_rate=learning_rate, max_features='auto', random_state=0)
    gb_clf.fit(x_train, y_train)

    print("Learning rate: ", learning_rate)
    print("Accuracy score (training): {0:.3f}".format(gb_clf.score(x_train, y_train)))
    print("Accuracy score (validation): {0:.3f}".format(gb_clf.score(x_test, y_test)))
    score_list.append(format(gb_clf.score(x_test, y_test)))
    
gb_clf2 = GradientBoostingClassifier(n_estimators=30, learning_rate=0.75, max_features='auto', random_state=0)
gb_clf2.fit(x_train, y_train)
predictions = gb_clf2.predict(x_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))

print("Classification Report")
print(classification_report(y_test, predictions))

plt.title('Gradient Boosting Classifier')
plt.plot(lr_list, score_list)
plt.legend(loc='lower right')
#plt.plot([0,1],[0,1])
#plt.xlim([0,1])
#plt.ylim([0.96,1.0])
plt.ylabel('Accuracy score')
plt.xlabel('Learning rate')
plt.savefig('GradientBoosting_ID3')
plt.show()










"""THEME B.2"""
#defining models
reg = GaussianNB()			# naive bayes
dtc = DecisionTreeClassifier()		# ID3
#loading dataset  features_checkin.csv for foursquare dataset
#feature_blog.csv for blog catalog data set.
dataset = pd.read_csv('features_checkin.csv',
	names=['source','dest','link','AA','CN','JC','katz','ND','PA','PR','RA','TN'],skiprows= [0])
#converting dataset into float datatype
dataset = dataset.astype(np.float64)
features1 = ['AA','CN','JC','katz','ND','PA','PR','RA','TN']
#dividing dataset into training and testing set
x = dataset.loc[:, features1].values
y = dataset.loc[:,['link']].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)
reg.fit(x_train,y_train)
print(reg.score(x_test,y_test))
dtc.fit(x_train,y_train)
print(dtc.score(x_test,y_test))
#applying PCA to reduce features

#for 5 principal components

pca2 = PCA(n_components=5)		# change n_components for no of componente
principalComponents2 = pca2.fit_transform(x)
principalDf3 = pd.DataFrame(data = principalComponents2
             , columns = ['principal component 1', 'principal component 2','principal component 3','p4','p5'])

x4 = principalDf3.loc[:, ['principal component 1', 'principal component 2','principal component 3','p4','p5']].values

x_train4, x_test4, y_train4, y_test4 = train_test_split(x4, y, test_size=0.2, random_state=4)

reg.fit(x_train4,y_train4)
print(reg.score(x_test4,y_test4))

dtc.fit(x_train4,y_train4)
print(dtc.score(x_test4,y_test4))

# SVD 

from sklearn.decomposition import TruncatedSVD

# 5 SVD components

svd = TruncatedSVD(n_components = 5)	# change n_components for no of componente
x5 = svd.fit(x).transform(x)
reg = GaussianNB()
x_train5, x_test5, y_train5, y_test5 = train_test_split(x5,y,test_size=0.2, random_state=4)
reg.fit(x_train5,y_train5)
print(reg.score(x_test5,y_test5))

dtc.fit(x_train5,y_train5)
print(dtc.score(x_test5,y_test5))






"""THEME C"""
import pandas as pd
import numpy as np 
df = pd.read_csv("features_blog.csv")
df.head()
aa=df['AA']
cn=df['CN']
jc=df['JC']
katz=df['katz']
nd=df['ND']
pa=df['PA']
pr=df['PR']
ra=df['RA']
tn=df['TN']
df_y = df['link']
data=[]
for i in range(len(aa)):
    data.append([0,0,0,0,0,0,0,0,0])
    data[i][0]=aa[i]
    data[i][1]=cn[i]
    data[i][2]=jc[i]
    data[i][3]=katz[i]
    data[i][4]=nd[i]
    data[i][5]=pa[i]
    data[i][6]=pr[i]
    data[i][7]=ra[i]
    data[i][8]=tn[i]
source=df['source']
dest=df['dest']
link=df['link']
import networkx as nx
G=nx.Graph()
for i in range(len(link)):
    if(link[i]==1):
        G.add_edge(source[i],dest[i])
AA={}
for i in range(len(link)):
    AA[(source[i],dest[i])]=tn[i]
    AA[(dest[i],source[i])]=tn[i] #Change this for every fearture
edge=G.edges()
A={}
for i in edge:
    A[i]=AA[i]
from operator import itemgetter
SortA=sorted(A.items(), key=itemgetter(1),reverse=True)
nx.info(G)
cc=[]
sp=[]
count=0
b=0
for i in range(0,len(SortA),100):
    for j in range(i,i+100):
        G.remove_edge(SortA[j][0][0],SortA[j][0][1])
    a=len(max(nx.connected_components(G), key=len))
    if b!=a:
        sp.append(count)
        print(count)
    b=a    
    count+=1
    cc.append(a)
