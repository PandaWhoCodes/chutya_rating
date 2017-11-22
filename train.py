import argparse
import numpy as np
import matplotlib.pyplot as plt

from sklearn import decomposition
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
from sklearn import gaussian_process

parser = argparse.ArgumentParser()
parser.add_argument('-model', type=str, default='linear_model')
parser.add_argument('-featuredim', type=int, default=20)
args = parser.parse_args()

features = np.loadtxt("data/features.txt", delimiter=',')
features_train = features[0:-50]
features_test = features[-50:]

pca = decomposition.PCA(n_components=args.featuredim)
pca.fit(features_train)
features_train = pca.transform(features_train)
features_test = pca.transform(features_test)

ratings = np.loadtxt("data/ratings.txt", delimiter=',')
ratings_train = ratings[0:-50]
ratings_test = ratings[-50:]

if args.model == 'linear_model':
    model = linear_model.LinearRegression()
elif args.model == 'svm':
    model = svm.SVR()
elif args.model == 'rf':
    model = RandomForestRegressor(n_estimators=50, max_depth=None, min_samples_split=2, random_state=0)
elif args.model == 'gpr':
    model = gaussian_process.GaussianProcess(theta0=1e-2, thetaL=1e-4, thetaU=1e-1)
else:
    raise NameError('Unknown machine learning model. Please us one of: rf, svm, linear_model, gpr')

model.fit(features_train, ratings_train)
ratings_predict = model.predict(features_test)
corr = np.corrcoef(ratings_predict, ratings_test)[0, 1]
print('Correlation:', corr)

residue = np.mean((ratings_predict - ratings_test) ** 2)
print('Residue:', residue)

truth, = plt.plot(ratings_test, 'r')
prediction, = plt.plot(ratings_predict, 'b')
plt.legend([truth, prediction], ["Real Value", "Prediction"])

plt.show()
