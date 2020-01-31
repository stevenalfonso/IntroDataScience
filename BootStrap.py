import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model

data = np.loadtxt("notas_andes.dat", skiprows=1)
Y = data[:,4]
X = data[:,:4]

betas = []
regresion = sklearn.linear_model.LinearRegression()
for i in range(10000):
    a = np.random.randint(0 , 69, size = 69)
    regresion.fit(X[a,:], Y[a])
    betas.append(regresion.coef_)
betas = np.array(betas)

labels_std = [np.std(betas[0]),np.std(betas[1]),np.std(betas[2]),np.std(betas[3])]
labels_ave = [np.mean(betas[0]),np.mean(betas[1]),np.mean(betas[2]),np.mean(betas[3])]
plt.figure(figsize=(8,8))
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.hist(betas[:,i])
    plt.title(r'$\beta = $ %0.4f' %labels_ave[i] + r'$\pm$ %0.4f' %labels_std[i])
    plt.xlabel(r'$\beta$')
    plt.tight_layout()
plt.savefig('BootStrap.png')
plt.show()
