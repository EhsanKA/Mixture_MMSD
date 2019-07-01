import numpy as np
import scipy as sp


def third_order_tensor(Y,U,V,W):
    output= np.zeros(shape=(U.shape[1], U.shape[1], U.shape[1]))
    for j1 in range(output.shape[0]):
        for j2 in range(output.shape[1]):
            for j3 in range(output.shape[2]):
                output[j1,j2,j3] = third_order_tensor_partial(Y,U,V,W,j1,j2,j3)
    return output
def third_order_tensor_partial(Y, U, V, W, j1, j2, j3):
    s = 0
    for i1 in range(U.shape[0]):
        for i2 in range(U.shape[0]):
            for i3 in range(U.shape[0]):
                s+= U[i1,j1]* V[i2,j2]* W[i3,j3] * Y[i1,i2,i3]
    return s


def three_tensor_product(a,b,c):
    a= np.kron(a,b)
    c = np.kron(a,c)
    return c


def second_part_of_third_order_tensor(sigma_hat, W_hat, mu_hat_second, e):
    # e = np.eye(W_hat.shape[0])
    s = np.zeros((W_hat.shape[1]**3))
    for i in range(W_hat.shape[0]):
        s1= three_tensor_product(np.dot(W_hat.T, mu_hat_second), np.dot(W_hat.T, e[:, i]), np.dot(W_hat.T, e[:, i]) )
        s2= three_tensor_product(np.dot(W_hat.T, e[:, i]), np.dot(W_hat.T, mu_hat_second), np.dot(W_hat.T, e[:, i]) )
        s3= three_tensor_product(np.dot(W_hat.T, e[:, i]), np.dot(W_hat.T, e[:, i]), np.dot(W_hat.T, mu_hat_second) )
        s+= (s1+ s2+ s3)
    s = s.reshape((W_hat.shape[1],W_hat.shape[1],W_hat.shape[1]))
    return sigma_hat* s

# Generating samples
d= 3
k= 2
n= 60

W = np.random.uniform(0,1,k)
W = W/np.sum(W)
Means = []

# Genetating means
for i in range(k):
    mean_j = np.random.uniform(-10, 10, d)
    Means.append(mean_j)

# select each sample comes from which Gaussian
which_gaussian = np.random.choice(k, n, p=W)
which_gaussian = which_gaussian.tolist()
# same sigmas
sigma = np.random.uniform(0,100)

X = []
for i in which_gaussian:
    z = np.random.multivariate_normal(Means[i], sigma*np.eye(d))
    X.append(z)

X= np.array((X))
mu_hat = np.mean(X[0:int(n / 2), :], axis=0)

covariance_sum = np.zeros((d,d))
for i in range(int(n/2)):
    covariance_sum += np.outer(X[i], X[i])

M2 = (2/n)* covariance_sum

tensors_sum = np.zeros((d**3))
for i in range(int(n/2), n):
    c= np.kron(X[i],X[i])
    c = np.kron(c, X[i])
    tensors_sum += c

M3_hat_second = (2/n)* tensors_sum
M3_hat_second = M3_hat_second.reshape((d,d,d))

mu_hat_second = np.mean(X[int(n / 2):, :], axis=0)

v, _ = np.linalg.eig(M2 - np.dot(mu_hat_second, mu_hat_second.T))

# predicting the variance
sigma_hat = v[k-1]

u, s, vh = np.linalg.svd(M2 - sigma_hat*np.eye(d), full_matrices=False)
M2_hat = np.dot(np.dot(u[:, :k], np.diag(s[:k])), vh[:k, :])
eigvalues, eigvectors = np.linalg.eig(M2_hat)
U_hat = u[:, :k]
W_hat = np.dot(U_hat, (sp.linalg.sqrtm(abs(np.linalg.pinv(np.dot(np.dot(U_hat.T, M2_hat), U_hat))))))
B_hat = np.dot(U_hat, (sp.linalg.sqrtm(abs(np.dot(np.dot(U_hat.T, M2_hat), U_hat)))))

# Second half of data

M_hat_www_second = third_order_tensor(M3_hat_second,W_hat,W_hat,W_hat) - second_part_of_third_order_tensor(sigma_hat, W_hat, mu_hat_second, eigvectors)

for t in range(1):
    theta = np.zeros((k))
    bound = 0
    for i in range(k):
        c = np.random.uniform(-np.sqrt(1 - bound**2),np.sqrt(1 - bound**2))
        theta[i] = c
        bound = np.linalg.norm(theta)
        if bound >=1:
            break
last_squared_matrix = np.dot(M_hat_www_second, theta)
eigvals, eigvecs = np.linalg.eig(last_squared_matrix)


print("sigma is: ", sigma_hat)
mus = np.zeros((d,k))
for i in range(k):
    mus[:, i] = (eigvals[i]/ (np.dot(theta.T, eigvecs[i])))* np.dot(B_hat, eigvecs[i])

w = np.dot(np.linalg.pinv(mus), mu_hat)

print("Mu is: ", mus)
print("W is: ", w)

