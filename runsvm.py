import pandas
import numpy
from sklearn import cross_validation
import sklearn.multiclass
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from sklearn import grid_search
import math
import time





'''Returns the average score of the given classifier'''
def getScore(estimator, trainFeatures, trainTargets):
	kfold = 5
	scores = cross_validation.cross_val_score(estimator, trainFeatures, trainTargets, cv=kfold, n_jobs=-1, verbose=0)
	#print scores
	return scores.mean(), scores.std()

def onevsall(estimator):
	return sklearn.multiclass.OneVsRestClassifier(estimator, n_jobs=-1)

def onevsone(estimator):
	return sklearn.multiclass.OneVsOneClassifier(estimator, n_jobs=-1)


def quickScore(estimator, trainFeatures, trainTargets, val):
	feat_train, feat_test, tar_train, tar_test = cross_validation.train_test_split(trainFeatures, trainTargets, train_size=val)
	return getScore(estimator, feat_train, tar_train)

'''
def score(estimator, test):
	return estimator.decision_function(test)
	'''

if __name__ == '__main__':
	trainSet = pandas.read_csv("train.csv")
	trainTargets = LabelEncoder().fit_transform(trainSet['target'])
	trainFeatures = trainSet.drop('target', axis=1)
 	#rbf_SVC = SVC(C=1.0, kernel='rbf', gamma=2.0)
	#poly_SVC = SVC(C=1.0, kernel='poly',degree=3)
	#linear_SVC = LinearSVC(C=1.0)

	feat_train, feat_test, tar_train, tar_test = cross_validation.train_test_split(trainFeatures, trainTargets, train_size=0.1)

	'''
	parameters = {'kernel':('linear', 'poly', 'rbf'), 'C':(0.0001, 0.1, 1, 5, 10)}
	svr = SVC()
	clf = grid_search.GridSearchCV(svr, parameters, n_jobs=4, verbose=1)
	print "Fitting..."
	clf.fit(feat_train, tar_train)

	print "Best score: " + str(clf.best_score_) + ", with parameters:"
	print clf.best_params_
	model = SVC(kernel=clf.best_params_['kernel'], C=clf.best_params_['C'])
	'''

	'''
	model = SVC(kernel='rbf', verbose=1)
	model.fit(trainFeatures, trainTargets)
	test = pandas.read_csv("test.csv") 
	results = model.decision_function(test)
	predictions = 1.0 / (1.0 + numpy.exp(-results))
	row_sums = predictions.sum(axis=1)
	predictions_normalised = predictions / row_sums[:, numpy.newaxis]

	# create submission file for Kaggle
	sample_submission = pandas.read_csv("sampleSubmission.csv")
	print len(predictions_normalised)
	prediction_DF = pandas.DataFrame(predictions_normalised, index=sample_submission.id.values, columns=sample_submission.columns[1:])
	prediction_DF.to_csv('svc_submission.csv', index_label='id')
	'''
	#p1v1 = onevsone(poly_SVC)
	#r1v1 = onevsone(rbf_SVC)

	#result2 = quickScore(poly_SVC, trainFeatures, trainTargets)
	#print "Poly Kernel 1v1 Results: " + str(result2)

	#result1 = quickScore(rbf_SVC, trainFeatures, trainTargets)
	#print "RBF Kernel 1v1 Results: " + str(result1)




	print "Trying various Degrees for Poly SVM"
	bestD = 0
	bestAccuracy = -1.0
	D = [1,2, 3, 4, 5]
	bstd = 0
	bestTime = 0
	for i in range(len(D)):
		startTime = time.time()
		dVal = D[i]
		score, std = getScore(SVC(C=3, kernel='poly',degree=dVal), trainFeatures, trainTargets)
		elapsedTime = time.time() - startTime
		if (score > bestAccuracy):
			bestD = dVal
			bestAccuracy = score
			bstd = std
			bestTime = elapsedTime
		print "D: " + str(dVal) + ", Score: " + str(score) +", Standard Deviation = " + str(std) + " Running Time: " + str(elapsedTime)
	
	print "Best D was: " + str(bestD) +", with accuracy: " + str(bestAccuracy) +", Standard Deviation: "+str(bstd) + " Running Time: " + str(bestTime)


	print "\n"
	print "Trying various C values for Polynomial Kernel of degree: " + str(bestD)
	bestC = 0.0
	bestAccuracy = -1.0
	C =[math.pow(10,i) for i in range(-5,5)]
	bstd = 0
	for i in range(len(C)):
		startTime = time.time()
		cVal = C[i]
		score, std = getScore(SVC(C=cVal, kernel='poly',degree=bestD), trainFeatures, trainTargets)
		elapsedTime = time.time() - startTime
		if (score > bestAccuracy):
			bestC = cVal
			bestAccuracy = score
			bstd = std
			bestTime = elapsedTime
		print "C: " + str(cVal) + ", Score: " + str(score)+", Standard Deviation = " + str(std)+ " Running Time: " + str(elapsedTime)
	
	print "Best C was: " + str(bestC) +", with accuracy: " + str(bestAccuracy) +", Standard Deviation: "+str(bstd) + " Running Time: " + str(bestTime)


	print "\n"
	print "Trying various Gamma values for RBF Kernel"
	bestG = 0.0
	bestAccuracy = -1.0
	G =[math.pow(10,i) for i in range(-5,5)]
	bstd = 0
	for i in range(len(G)):
		startTime = time.time()
		gVal = G[i]
		#score = quickScore(poly_SVC, trainFeatures, trainTargets)
		score, std = getScore(SVC(C=3, kernel='rbf',gamma=gVal), trainFeatures, trainTargets)
		elapsedTime = time.time() - startTime
		if (score > bestAccuracy):
			bestG = gVal
			bestAccuracy = score
			bstd = std
			bestTime = elapsedTime
		print "Gamma: " + str(gVal) + ", Score: " + str(score) +", Standard Deviation = " + str(std)+ " Running Time: " + str(elapsedTime)
	elapsedTime = time.time() - startTime
	print "Best Gamma was: " + str(bestG) +", with accuracy: " + str(bestAccuracy) +", Standard Deviation: "+str(bstd) + " Running Time: " + str(bestTime)


	print "\n"
	print "Trying various C values for RBF Kernel with Gamma = " + str(bestG)
	bestC = 0.0
	bestAccuracy = -1.0
	bstd = 0
	for i in range(len(C)):
		startTime = time.time()
		cVal = C[i]
		score, std = getScore(SVC(C=cVal, kernel='rbf',gamma=bestG), trainFeatures, trainTargets)
		elapsedTime = time.time() - startTime
		if (score > bestAccuracy):
			bestC = cVal
			bestAccuracy = score
			bstd = std
			bestTime = elapsedTime
		print "C: " + str(cVal) + ", Score: " + str(score)+", Standard Deviation = " + str(std)+ " Running Time: " + str(elapsedTime)
	elapsedTime = time.time() - startTime
	print "Best C was: " + str(bestC) +", with accuracy: " + str(bestAccuracy) +", Standard Deviation: "+str(bstd) + " Running Time: " + str(bestTime)



	print "\nTrying various C values for Linear Kernel"
	bestC = 0.0
	bestAccuracy = -1.0
	bstd = 0
	for i in range(len(C)):
		startTime = time.time()
		cVal = C[i]
		score, std = getScore(LinearSVC(C=cVal), trainFeatures, trainTargets)
		elapsedTime = time.time() - startTime
		if (score > bestAccuracy):
			bestC = cVal
			bestAccuracy = score
			bstd = std
			bestTime = elapsedTime
		print "C: " + str(cVal) + ", Score: " + str(score)+", Standard Deviation = " + str(std)+ " Running Time: " + str(elapsedTime)
	
	print "Best C was: " + str(bestC) +", with accuracy: " + str(bestAccuracy) +", Standard Deviation: "+str(bstd) + " Running Time: " + str(bestTime)

	print "Best C was: " + str(bestC) +", with accuracy: " + str(bestAccuracy)
	


