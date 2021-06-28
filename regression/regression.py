from json import load
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.labelsize'] = 14
mae_olss, mae_ldas, mae_plss = [], [], []
std_lda, std_pls = [], []
for i in range(1,5):
    train_smi, train_e, train_vec, train_lda_vec = load(open('reg_train_%i.json' % i))
    test_smi, test_e, test_vec, test_lda_vec = load(open('reg_test_%i.json' % i))
    mae_ols, mae_lda, mae_pls = [], [], []
    for j in range(5, 85, 5):
        #OLS
        train_vec_ols = [vec[:j] for vec in train_vec]
        test_vec_ols = [vec[:j] for vec in test_vec]
        lr = LinearRegression()
        lr.fit(train_vec_ols, train_e)
        y_hat_ols = lr.predict(test_vec_ols).ravel()
        y_err_ols = np.mean(np.abs(np.array(test_e)-y_hat_ols))
        mae_ols.append(y_err_ols)
        
        #LDA
        train_vec_lda = [vec[:j] for vec in train_lda_vec]
        test_vec_lda = [vec[:j] for vec in test_lda_vec]
        lda = LinearRegression()
        lda.fit(train_vec_lda, train_e)
        y_hat_lda = lda.predict(test_vec_lda).ravel()
        y_err_lda = np.mean(np.abs(np.array(test_e)-y_hat_lda))
        mae_lda.append(y_err_lda)
        #PLS
        pls = PLSRegression(n_components=j)
        pls.fit(train_vec, train_e)
        y_hat_pls = pls.predict(test_vec).ravel()
        y_err_pls = np.mean(np.abs(np.array(test_e)-y_hat_pls))
        mae_pls.append(y_err_pls)
        
    mae_olss.append(mae_ols)
    mae_ldas.append(mae_lda)
    mae_plss.append(mae_pls)


mean_lda = [mae_lda for mae_lda in np.mean(mae_ldas,axis=0)]
mean_pls = [mae_pls for mae_pls in np.mean(mae_plss,axis=0)]
mean_reg = [mae_reg for mae_reg in np.mean(mae_olss,axis=0)]

max_lda = [mae_lda for mae_lda in np.max(mae_ldas,axis=0)]
max_pls = [mae_pls for mae_pls in np.max(mae_plss,axis=0)]
max_reg = [mae_reg for mae_reg in np.max(mae_olss,axis=0)]

min_lda = [mae_lda for mae_lda in np.min(mae_ldas,axis=0)]
min_pls = [mae_pls for mae_pls in np.min(mae_plss,axis=0)]
min_reg = [mae_reg for mae_reg in np.min(mae_olss,axis=0)]

min_e_lda = np.asarray(mean_lda) - np.asarray(min_lda)
min_e_pls = np.asarray(mean_pls) - np.asarray(min_pls)
min_e_reg = np.asarray(mean_reg) - np.asarray(min_reg)

max_e_lda = np.asarray(max_lda) - np.asarray(mean_lda)
max_e_pls = np.asarray(max_pls) - np.asarray(mean_pls)
max_e_reg = np.asarray(max_reg) - np.asarray(mean_reg)

reg_error = [min_e_reg, max_e_reg]
plt.errorbar(range(5,85,5), mean_reg, yerr=reg_error,color='k',fmt='o',capsize=5,label="OLS")
lda_error = [min_e_lda, max_e_lda]
plt.errorbar(range(5,85,5), mean_lda, yerr=lda_error,color='r',fmt='o',capsize=5,label="LDA" )
pls_error = [min_e_pls, max_e_pls]
plt.errorbar(range(5,85,5), mean_pls, yerr=pls_error,color='b',fmt='o',capsize=5,label="PLS")

hb_mean = [1.4561]*16
plt.plot(range(5,85,5), hb_mean, 'c-',label='Hotbit')
#plt.legend()
plt.xlabel("Dimensions", fontsize=16)
plt.ylabel("Mean Absolute Error (eV)", fontsize=16)
plt.savefig('all_hb.png')
plt.show()
    
