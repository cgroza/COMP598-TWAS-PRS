import os
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model, model_selection
import statsmodels.api as sm
import scipy.stats as stat
import sys
import pickle

os.environ["OMP_NUM_THREADS"] = "10" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "10" # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "10" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "10" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "10" # export NUMEXPR_NUM_THREADS=6

root = sys.argv[1]
causal_tissue = sys.argv[2]

print("Loading train data")
# load data
tissue = os.path.join(root, "eQTL", causal_tissue)
genotypes = pd.read_csv(os.path.join(root, "genotype", "TWAS_genotype.csv.gz"))
phenotypes = pd.read_csv(os.path.join(root, "phenotype", "TWAS_phenotype.csv"))

genes = {}
# load genes
for gene_csv in os.listdir(tissue):
    print("Loading " + gene_csv)
    eqtls = pd.read_csv(os.path.join(tissue, gene_csv), sep=',', header=0)
    eqtls = eqtls.set_index("Unnamed: 0").T
    genes[os.path.splitext(os.path.basename(gene_csv))[0]] = eqtls

print("Imputing train gene expression")
# will contain imputed gene expression for each gene in each sample
train_expression = {}
# impute
for gene in genes:
    gene_eqtl_effects = genes[gene]
    sample_eqtls = genotypes.loc[:, list(gene_eqtl_effects.columns)]
    train_expression[gene] = sample_eqtls.dot(gene_eqtl_effects.T['w_hat'])
# Gene data to regress on
gene_expr = pd.DataFrame(train_expression)


print("Loading test data")
test_genotypes = pd.read_csv(os.path.join(root, "genotype", "test_genotype.csv.gz"))
test_phenotypes = pd.read_csv(os.path.join(root, "phenotype", "test_phenotype.csv"))

print("Imputing test gene expression")
# will contain imputed gene expression for each gene in each sample
test_expression = {}
# impute
for gene in genes:
    gene_eqtl_effects = genes[gene]
    sample_eqtls = test_genotypes.loc[:, list(gene_eqtl_effects.columns)]
    test_expression[gene] = sample_eqtls.dot(gene_eqtl_effects.T['w_hat'])
test_expr = pd.DataFrame(test_expression)

# single gene TWAS-PRS
models = dict()
for gene in gene_expr:
        model = linear_model.LinearRegression()
        model.fit(gene_expr[[gene]], phenotypes['phenotye'])
        models[gene] = [model.coef_[0]]

coef_df = pd.DataFrame(models)
print("Single Gene R2 " + str(stat.pearsonr(coef_df.dot(test_expr.T).T[0], test_phenotypes['phenotye'])))


print("Weight,Train,Test")
for a in [1, 0.5, 0.2, 0.15, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005]:
    twas = linear_model.Lasso(alpha=a)
    twas.fit(gene_expr, phenotypes['phenotye'])
    print(str(a) + "," +
            str(twas.score(gene_expr, phenotypes['phenotye'])) + "," +
            str(twas.score(test_expr, test_phenotypes['phenotye'])))

print("Genotype lasso")
for a in [1, 0.5, 0.2, 0.15, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005]:
    twas_genotype = linear_model.Lasso(alpha=a)
    twas_genotype.fit(genotypes, phenotypes['phenotye'])
    print(str(a) + "," +
            str(twas_genotype.score(genotypes, phenotypes['phenotye'])) + "," +
            str(twas_genotype.score(test_genotypes, test_phenotypes['phenotye'])))

#print("\n".join(twas.coef_))

corr_csv = os.path.join("correlations", os.path.basename(os.path.normpath(root)))
gene_expr.corr().to_csv(corr_csv)
