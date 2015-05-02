import pandas
import numpy
import sklearn.cross_validation
import sklearn.multiclass
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder


trainSet = pandas.read_csv("train.csv")
trainTargets = LabelEncoder().fit_transform(trainingSet['target'])
trainFeatures = trainSet.drop('target', axis=1)



'''Returns the average score of the given classifier'''
def score(estimator):
	numFolds = 5
	scores = sklearn.cross_validation.cross_val_score(estimator, trainFeatures, y=trainTargets, scoring=None, cv=numFolds, n_jobs=-1, verbose=0, fit_params=None, pre_dispatch='2*n_jobs')
	avg = 0
	for i in range(len(scores)):
		avg = avg + scores[i]/len(scores)


def onevsall(estimator):
	return sklearn.multiclass.OneVsRestClassifier(estimator, n_jobs=-1)

def onevsone(estimator):
	return sklearn.multiclass.OneVsOneClassifier(estimator, n_jobs=-1)

rbf_SVC = SVC(C=1.0, kernel='rbf', gamma='0.0')


