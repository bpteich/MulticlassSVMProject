import pandas
import numpy
from sklearn import cross_validation
import sklearn.multiclass
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder





'''Returns the average score of the given classifier'''
def score(estimator, trainFeatures, trainTargets):
	kfold = 5
	scores = cross_validation.cross_val_score(estimator, trainFeatures, trainTargets, cv=kfold, n_jobs=1)
	avg = 0
	for i in range(len(scores)):
		avg = avg + scores[i]/len(scores)
	return avg

def onevsall(estimator):
	return sklearn.multiclass.OneVsRestClassifier(estimator, n_jobs=-1)

def onevsone(estimator):
	return sklearn.multiclass.OneVsOneClassifier(estimator, n_jobs=-1)


def quickScore(estimator, trainFeatures, trainTargets):
	feat_train, feat_test, tar_train, tar_test = cross_validation.train_test_split(trainFeatures, trainTargets, train_size=1000, test_size=10)
	return score(estimator, feat_train, tar_train)

'''
def score(estimator, test):
	return estimator.decision_function(test)
	'''

if __name__ == '__main__':
	trainSet = pandas.read_csv("train.csv")
	trainTargets = LabelEncoder().fit_transform(trainSet['target'])
	trainFeatures = trainSet.drop('target', axis=1)
	#trainFeatures[trainFeatures > 4] = 4
	test = pandas.read_csv("test.csv") 
	#trainFeatures = trainFeatures.T.to_dict().values()
 	rbf_SVC = SVC(C=1.0, kernel='rbf', gamma=2.0)
	#rbf_1v1 = onevsone(rbf_SVC).fit(trainFeatures,trainTargets)
	#poly_SVC = SVC(C=1.0, kernel='poly',degree=3)

	result = quickScore(rbf_SVC, trainFeatures, trainTargets)
	print result


