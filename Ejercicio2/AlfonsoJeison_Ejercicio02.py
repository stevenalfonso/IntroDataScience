import numpy as np
import matplotlib.pyplot as plt
data = np.loadtxt('numeros_20.txt')
x_training = data[0:10,0]
y_training = data[0:10,1]
x_test = data[10:,0]
y_test = data[10:,1]
Erms_Training = []
Erms_Test = []
#Vector Beta
fig=plt.figure(figsize=(20,30))
for s in range(10):
    m=s+1
    n=10
    X = np.zeros((n, m))
    for i in range(n):
        for ii in range(m):
            X[i, ii] = x_training[i]**ii
    beta = np.matmul(np.linalg.pinv(X), y_training[0:n])
    x = np.linspace(0,1,100)
    y = np.zeros(100)
    for i in range(x.shape[0]):
        for j in range(beta.shape[0]):
            y[i] += beta[j]*x[i]**j
    plt.subplot(5,2,m)
    plt.scatter(x_training, y_training,marker='.',c='black')
    plt.plot(x, y, c='red')
    plt.ylim(0,2)
    plt.title(r'$M=$'+str(s))
    plt.xlabel(r'$X$')
    plt.ylabel(r'$Y$')
    plt.savefig('XvsY.png')
#plt.show()
    
    y_Training = np.zeros(10)
    for i in range(10):
        for j in range(m):
            y_Training[i] += beta[j]*x_training[i]**j
    E_Training = []
    for i in range(10):
        E_Training.append(1/2 * (y_Training[i] - y_training[i])**2)  
    E = sum(E_Training) 
    Erms_Training.append(np.sqrt(2 * E / 10))
    
    y_Test = np.zeros(10)
    for i in range(10):
        for j in range(m):
            y_Test[i] += beta[j]*x_test[i]**j
    E_Test = []
    for i in range(10):
        E_Test.append(1/2 * (y_Test[i] - y_test[i])**2)  
    E_ = sum(E_Test) 
    Erms_Test.append(np.sqrt(2 * E_ / 10))

M = []
for i in range(10):
    M.append(i)
plt.figure()
plt.plot(M, Erms_Training, 'b-o',label='$E_{Training}$')
plt.plot(M, Erms_Test,'r-o',label='$E_{Test}$')
plt.xlabel(r'$M$')
plt.ylabel(r'$E_{RMS}$')
plt.legend()
plt.savefig('MvsErms.png')
plt.show()
