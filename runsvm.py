import pandas
import numpy
from sklearn import cross_validation
import sklearn.multiclass
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder





'''Returns the average score of the given classifier'''
def score(estimator, trainFeatures, trainTargets):
	kfold = 5
	scores = cross_validation.cross_val_score(estimator, trainFeatures, trainTargets, cv=kfold, n_jobs=-1)
	avg = 0
	for i in range(len(scores)):
		avg = avg + scores[i]/len(scores)


def onevsall(estimator):
	return sklearn.multiclass.OneVsRestClassifier(estimator, n_jobs=-1)

def onevsone(estimator):
	return sklearn.multiclass.OneVsOneClassifier(estimator, n_jobs=-1)


if __name__ == '__main__':
	trainSet = pandas.read_csv("train.csv")
	trainTargets = LabelEncoder().fit_transform(trainSet['target'])
	trainFeatures = trainSet.drop('target', axis=1)
	trainFeatures = trainFeatures.T.to_dict().values()
 	rbf_SVC = SVC(C=1.0, kernel='rbf', gamma='2.0')
	vsone_rbf_score = score(onevsone(rbf_SVC), trainFeatures, trainTargets)
	print vsone_rbf_score



