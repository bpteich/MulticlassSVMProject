import pandas
import numpy
from sklearn import cross_validation
import sklearn.multiclass
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
import math





'''Returns the average score of the given classifier'''
def getscore(estimator, trainFeatures, trainTargets):
	kfold = 3
	scores = cross_validation.cross_val_score(estimator, trainFeatures, trainTargets, cv=kfold, n_jobs=1, scoring='accuracy')
	avg = 0
	for i in range(len(scores)):
		avg = avg + scores[i]/len(scores)
	minimum = min(scores)
	return avg

def onevsall(estimator):
	return sklearn.multiclass.OneVsRestClassifier(estimator, n_jobs=-1)

def onevsone(estimator):
	return sklearn.multiclass.OneVsOneClassifier(estimator, n_jobs=-1)


def quickScore(estimator, trainFeatures, trainTargets):
	feat_train, feat_test, tar_train, tar_test = cross_validation.train_test_split(trainFeatures, trainTargets, train_size=0.25)
	return getscore(estimator, feat_train, tar_train)

'''
def score(estimator, test):
	return estimator.decision_function(test)
	'''

if __name__ == '__main__':
	trainSet = pandas.read_csv("train.csv")
	trainTargets = LabelEncoder().fit_transform(trainSet['target'])
	trainFeatures = trainSet.drop('target', axis=1)
 	rbf_SVC = SVC(C=1.0, kernel='rbf', gamma=2.0)
	poly_SVC = SVC(C=1.0, kernel='poly',degree=3)
	linear_SVC = LinearSVC(C=1.0)

	#p1v1 = onevsone(poly_SVC)
	#r1v1 = onevsone(rbf_SVC)

	#result2 = quickScore(poly_SVC, trainFeatures, trainTargets)
	#print "Poly Kernel 1v1 Results: " + str(result2)

	#result1 = quickScore(rbf_SVC, trainFeatures, trainTargets)
	#print "RBF Kernel 1v1 Results: " + str(result1)

	print "Trying various C values for Polynomial Kernel"
	bestC = 0.0
	bestAccuracy = -1.0
	C =[math.pow(10,i) for i in range(-5,5)]
	for i in range(len(C)):
		cVal = C[i]
		poly_SVC = SVC(C=cVal, kernel='poly',degree=3)
		score = quickScore(poly_SVC, trainFeatures, trainTargets)
		if (score > bestAccuracy):
			bestC = cVal
			bestAccuracy = score
		print "C: " + str(cVal) + ", Score: " + str(score)

	print "Best C was: " + str(bestC) +", with accuracy: " + str(bestAccuracy)

	print "\n"
	print "Trying various C values for Linear SVM"
	bestC = 0.0
	bestAccuracy = -1.0
	C =[math.pow(10,i) for i in range(-5,5)]
	for i in range(len(C)):
		cVal = C[i]
		linear_SVC = LinearSVC(C=cVal)
		score = quickScore(poly_SVC, trainFeatures, trainTargets)
		if (score > bestAccuracy):
			bestC = cVal
			bestAccuracy = score
		print "C: " + str(cVal) + ", Score: " + str(score)

	print "Best C was: " + str(bestC) +", with accuracy: " + str(bestAccuracy)
	#model = LinearSVC(C=bestC).fit(trainFeatures, trainTargets)
	model = LinearSVC(C=bestC).fit(trainFeatures, trainTargets)
	print "Trained..."
	test = pandas.read_csv("test.csv") 
	results = model.decision_function(test)
	predictions = 1.0 / (1.0 + numpy.exp(-results))
	row_sums = predictions.sum(axis=1)
	predictions_normalised = predictions / row_sums[:, numpy.newaxis]

	# create submission file for Kaggle
	prediction_DF = pandas.DataFrame(predictions_normalised, index=sample_submission.id.values, columns=sample_submission.columns[1:])
	prediction_DF.to_csv('svc_submission.csv', index_label='id')


