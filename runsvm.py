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
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA





'''Returns the average score of the given classifier'''
def getScore(estimator, trainFeatures, trainTargets, kfold=10):
	scores = cross_validation.cross_val_score(estimator, trainFeatures, trainTargets, cv=kfold, n_jobs=-1, verbose=0)
	#print scores
	return scores.mean(), scores.std()



def onevsall(estimator):
	return sklearn.multiclass.OneVsRestClassifier(estimator, n_jobs=1)

def onevsone(estimator):
	return sklearn.multiclass.OneVsOneClassifier(estimator, n_jobs=1)


def quickScore(estimator, trainFeatures, trainTargets, val):
	feat_train, feat_test, tar_train, tar_test = cross_validation.train_test_split(trainFeatures, trainTargets, train_size=val)
	return getScore(estimator, feat_train, tar_train, 3)

'''
def score(estimator, test):
	return estimator.decision_function(test)
	'''

def compareMultiClass(trainFeatures, trainTargets):
	rbfsvm = SVC(C=1, kernel='rbf',gamma=2)
	polysvm = SVC(C=1, kernel='poly',degree=3)

	p1v1 = onevsone(polysvm )
	r1v1 = onevsone(rbfsvm)

	r1va = onevsall(rbfsvm)
	p1va = onevsall(polysvm)

	print "Polynomial SVM 1v1:"
	startTime = time.time()
	score, std = getScore(p1v1, trainFeatures, trainTargets)
	elapsedTime = time.time() - startTime
	print "Running Time: " + str(elapsedTime) +", Accuracy: "+str(score) + ", Standard Deviation: " + str(std)
	print "\n"
	
	print "Polynomial SVM 1vAll:"
	startTime = time.time()
	score, std = getScore(p1va, trainFeatures, trainTargets)
	elapsedTime = time.time() - startTime
	print "Running Time: " + str(elapsedTime) +", Accuracy: "+str(score) + ", Standard Deviation: " + str(std)
	print "\n"

	print "RBF SVM 1v1:"
	startTime = time.time()
	score, std = getScore(r1v1, trainFeatures, trainTargets)
	elapsedTime = time.time() - startTime
	print "Running Time: " + str(elapsedTime) +", Accuracy: "+str(score) + ", Standard Deviation: " + str(std)
	print "\n"

	print "RBF SVM 1vAll:"
	startTime = time.time()
	score, std = getScore(r1va, trainFeatures, trainTargets)
	elapsedTime = time.time() - startTime
	print "Running Time: " + str(elapsedTime) +", Accuracy: "+str(score) + ", Standard Deviation: " + str(std)
	print "\n"
		

def search_vals():
	
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
	
	bestG = 0.000004
	
	print "\n"
	print "Trying various Gamma values for RBF Kernel"
	bestG = 0.0
	bestAccuracy = -1.0
	G =[0.000003, 0.000004, 0.0000045, 0.000005, 0.0000055, 0.000006, 0.000007, 0.00001]
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
	C =[i for i in range(1,10)]
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


	bestC = 4


def predict_and_save(estimator):

	results = model.predict_proba(test)
	#print results[0]
	#predictions = 1.0 / (1.0 + numpy.exp(-results))
	#row_sums = predictions.sum(axis=1)
	#predictions_normalised = predictions / row_sums[:, numpy.newaxis]
	
	# create submission file for Kaggle
	sample_submission = pandas.read_csv("sampleSubmission.csv")
	prediction_DF = pandas.DataFrame(results, index=sample_submission.id.values, columns=sample_submission.columns[1:])
	#prediction_DF = pandas.DataFrame(predictions_normalised, index=sample_submission.id.values, columns=sample_submission.columns[1:])
	prediction_DF.to_csv('svc_submission.csv', index_label='id')

def plot():

	# title for the plots
	titles = ['RBF C=4, Gamma=0.000004', 'RBF C=4, Gamma=0.0001', 'Linear C=0.001']
	rbfsvm = SVC(C=4, kernel='rbf',gamma=0.000004, verbose=0).fit(trainFeatures, trainTargets)
	polysvm = SVC(C=4, kernel='rbf',gamma=0.0001, verbose=0).fit(trainFeatures, trainTargets)
	linsvm = SVC(C=0.001, kernel='poly',degree=1, verbose=0).fit(trainFeatures, trainTargets)
	
	# visualize the decision surface, projected down to the first
	# two principal components of the dataset
	'''
	pca = PCA().fit(trainFeatures)

	X = pca.transform(trainFeatures)
	#X = pca.transform(feat_train)

	# Gemerate grid along first two principal components
	multiples = numpy.arange(-2, 2, 0.1)

	# steps along first component
	first = multiples[:, numpy.newaxis] * pca.components_[0, :]
	# steps along second component
	second = multiples[:, numpy.newaxis] * pca.components_[1, :]

	# combine
	grid = first[numpy.newaxis, :, :] + second[:, numpy.newaxis, :]
	grid = (numpy.absolute(grid)*10)//1
	#print grid[1][0]
	#print grid
	#print len(grid)
	#data = trainFeatures.as_matrix()
	data = trainFeatures
	#print data.shape
	#print grid.shape
	flat_grid = grid.reshape(-1, data.shape[1])
	#print flat_grid.shape
	#print flat_grid
	#Z = rbfsvm.predict(test)
	#print max(Z)

	plt.figure(figsize=(12, 5))

	# predict and plot
	for i, clf in enumerate((rbfsvm, polysvm, linsvm)):
		# Plot the decision boundary. For that, we will assign a color to each
		# point in the mesh [x_min, m_max]x[y_min, y_max].
		plt.subplot(1, 3, i + 1)
		Z = clf.predict(X)
		#print Z.shape

		# Put the result into a color plot
		Z = Z.reshape(grid.shape[0],-1)
		plt.contourf(multiples, multiples, Z, cmap=plt.cm.Paired)
		plt.axis('off')

		# Plot also the training points
		#plt.scatter(X[:, 0], X[:, 1], c=trainTargets, cmap=plt.cm.Paired)
		#plt.scatter(X[:, 0], X[:, 1], c=tar_train, cmap=plt.cm.Paired)
		plt.title(titles[i])
	plt.tight_layout()
	plt.show()
	'''
	
	coords = 10
	mults = numpy.arange(0,coords,1)

	feat_train, feat_test, tar_train, tar_test = cross_validation.train_test_split(trainFeatures, trainTargets, train_size=coords**2)
	
	plt.figure(figsize=(12, 5))

	# predict and plot
	for i, clf in enumerate((rbfsvm, polysvm, linsvm)):
		# Plot the decision boundary. For that, we will assign a color to each
		# point in the mesh [x_min, m_max]x[y_min, y_max].
		plt.subplot(1, 3, i + 1)
		Z = clf.predict(feat_train)
		#print Z.shape

		# Put the result into a color plot
		Z = Z.reshape(coords,coords)
		plt.contourf(mults,mults, Z, cmap=plt.cm.Paired)
		plt.axis('off')

		# Plot also the training points
		#plt.scatter(mults, mults, c=tar_train, cmpap=plt.cm.Paired)
		#plt.scatter(X[:, 0], X[:, 1], c=trainTargets, cmap=plt.cm.Paired)
		#plt.scatter(X[:, 0], X[:, 1], c=tar_train, cmap=plt.cm.Paired)
		plt.title(titles[i])
	plt.tight_layout()
	plt.show()
	
	'''
	#feat_train, feat_test, tar_train, tar_test = cross_validation.train_test_split(trainFeatures, trainTargets, train_size=val)
	h = 0.5  # step size in the mesh
	X = trainFeatures.as_matrix()
	Y = trainTargets
	x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
	y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
	xx, yy = numpy.meshgrid(numpy.arange(x_min, x_max, h), numpy.arange(y_min, y_max, h))

	for i, clf in enumerate((rbfsvm, polysvm, linsvm)):
		# Plot the decision boundary. For that, we will assign a color to each
		# point in the mesh [x_min, m_max]x[y_min, y_max].
		plt.subplot(1, 3, i + 1)
		Z = clf.predict(X)

		# Put the result into a color plot
		#Z = Z.reshape(grid.shape[:-1])
		plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
		plt.axis('off')

		# Plot also the training points
		plt.scatter(X[:, 0], X[:, 1], c=trainTargets, cmap=plt.cm.Paired)
		plt.title(titles[i])
	plt.tight_layout()
	plt.show()
	'''


if __name__ == '__main__':
	trainSet = pandas.read_csv("train.csv")
	trainTargets = LabelEncoder().fit_transform(trainSet['target'])
	trainFeatures = trainSet.drop('target', axis=1)
	test = pandas.read_csv("test.csv") 

	'''Due to the massive time complexity of some of these operations only uncomment the things you want to run! '''

	###Uncomment this to compare 1v1 and 1va###
	#compareMultiClass(trainFeatures,trainTargets)

	###Uncomment this to (attempt to) plot the decision Boundaries ###
	#plot() 

	###Uncomment this to do a sort of repeated linear grid search for parameters. ###
	search_vals()


 	#rbf_SVC = SVC(C=1.0, kernel='rbf', gamma=2.0)
	#poly_SVC = SVC(C=1.0, kernel='poly',degree=3)
	#linear_SVC = LinearSVC(C=1.0)

	#feat_train, feat_test, tar_train, tar_test = cross_validation.train_test_split(trainFeatures, trainTargets, train_size=0.1)

	'''#Exhaustive Grid Search
	parameters = {'kernel':('linear', 'poly', 'rbf'), 'C':(0.0001, 0.1, 1, 5, 10)}
	svr = SVC()
	clf = grid_search.GridSearchCV(svr, parameters, n_jobs=4, verbose=1)
	print "Fitting..."
	clf.fit(feat_train, tar_train)

	print "Best score: " + str(clf.best_score_) + ", with parameters:"
	print clf.best_params_
	model = SVC(kernel=clf.best_params_['kernel'], C=clf.best_params_['C'])
	'''

	''' ### This is what you uncomment to run the best SVM found so far over the data and save the result ###
	model = SVC(C=4, kernel='rbf',gamma=0.000004, verbose=0, probability=True).fit(trainFeatures, trainTargets)
	print "C = 4, Gamma = 0.000004, # Support Vectors = " + str(model.n_support_)
	predict_and_save(model) #saves predictions of the test set to svc_submission.csv
	'''
	

	#result2 = quickScore(poly_SVC, trainFeatures, trainTargets)
	#print "Poly Kernel 1v1 Results: " + str(result2)

	#result1 = quickScore(rbf_SVC, trainFeatures, trainTargets)
	#print "RBF Kernel 1v1 Results: " + str(result1)



	

	'''
	print "\nTrying various C values for Linear Kernel"
	bestC = 0.0
	bestAccuracy = -1.0
	bstd = 0
	C =[0.0005, 0.0008, 0.001, 0.003, 0.006, 0.008, 0.01]
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
	'''
	


