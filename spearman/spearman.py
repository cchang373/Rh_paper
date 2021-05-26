from scipy.stats import spearmanr
from json import load

for i in range(1,5):
    rho_pls, rho_lda, es_pls, es_lda, es_dft, smis = load(open('rho_pls_lda_%i.json' % i))
    for e_pls, e_lda, e_dft in zip(es_pls, es_lda, es_dft):
        rho_pls_n, p_pls = spearmanr(e_dft, e_pls)
        rho_lda_n, p_lda = spearmanr(e_dft, e_lda)
        #print(rho_pls_n)
        #print(rho_lda_n)
    
