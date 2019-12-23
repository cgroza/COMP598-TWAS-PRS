#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import os
import fnmatch
import shlex
import statsmodels.api as sm
import matplotlib.pyplot as plt
import scipy
import sys
import time
import uuid
from pandas_plink import read_plink
from timeit import default_timer as timer
from scipy.stats import norm
from scipy import sparse, io
from subprocess import run, PIPE
from datetime import datetime



def run_bash_cmd(command):
    cmd = shlex.split(command)
    output = run(cmd, stderr=PIPE, universal_newlines=True)
    print(output.stderr)


def updt(total, progress):
    barLength, status = 40, ""
    progress = float(progress) / float(total)
    if progress >= 1.:
        progress, status = 1, "\r\n"
    block = int(round(barLength * progress))
    text = "\r[{}] {:.0f}% {}".format(
        "#" * block + "-" * (barLength - block), round(progress * 100, 0),
        status)
    sys.stdout.write(text)
    sys.stdout.flush()

sim_num = 9
num_causal_snp_per_gene = 0.5 # float("inf")
num_causal_gene = 0.1
num_pleiotropy_snp_per_gene = 0.1
h_g_2 = 0.1
sigma_g1 = 0.3
sigma_g2 = 0.5
N_GWAS = 10000
N_TWAS = 10000
N_eQTL = 1000
N_Test = 5000
N_1KG = 489
num_block = 50
num_tissue = 1
causal_tissue = 0
is_pleiotropy = True


data_dir = os.path.join("/", "home", "lisiyi93", "GSR", "data")
gene2snps_dir = os.path.join(data_dir, "fake_genes", "fake_genes")
snp_info_dir = os.path.join(data_dir, "LD_blocks", "snpinfo.csv")
simulated_data_dir = os.path.join("/", "home", "lisiyi93", "comp598", "simulation_setting{}".format(sim_num), "simulation_{}_{}".format(datetime.now().strftime('%Y-%m-%d-%H:%M:%S'), str(uuid.uuid4()).split("-")[0]))


run_bash_cmd("mkdir {}".format(simulated_data_dir))
run_bash_cmd("mkdir {}".format(os.path.join(simulated_data_dir, "causality")))
run_bash_cmd("mkdir {}".format(os.path.join(simulated_data_dir, "genotype")))
run_bash_cmd("mkdir {}".format(os.path.join(simulated_data_dir, "phenotype")))
run_bash_cmd("mkdir {}".format(os.path.join(simulated_data_dir, "eQTL")))
run_bash_cmd("mkdir {}".format(os.path.join(simulated_data_dir, "GWAS")))


with open(os.path.join(simulated_data_dir, "settings.txt"), "w+") as f:
    print("num_causal_snp_per_gene: {}".format(num_causal_snp_per_gene), file=f)
    print("num_causal_gene: {}".format(num_causal_gene), file=f)
    print("num_pleiotropy_snp_per_gene: {}".format(num_pleiotropy_snp_per_gene), file=f)
    print("h_g_2: {}".format(h_g_2), file=f)
    print("sigma_g1: {}".format(sigma_g1), file=f)
    print("sigma_g2: {}".format(sigma_g2), file=f)
    print("pleiotropy: {}".format(is_pleiotropy))
    print("N_GWAS: {}".format(N_GWAS), file=f)
    print("N_TWAS: {}".format(N_TWAS), file=f)
    print("N_eQTL: {}".format(N_eQTL), file=f)
    print("N_Test: {}".format(N_Test), file=f)
    print("num_block: {}".format(num_block), file=f)
    print("num_tissue: {}".format(num_tissue), file=f)


snp_info_df = pd.read_csv(snp_info_dir, index_col="rsid")



# loading gene chromosomal location information into a dictionary
gene_chromo_dir = os.path.join(data_dir, "gene_annotation", "gene_to_chromosome.csv")
gene_to_chromo = {}

with open(gene_chromo_dir) as gene_chromo_file:
    gene_chromo_file.readline()
    for l in gene_chromo_file:
        l = l.split(',')
        if l[1] not in ['X', 'Y', 'MT']:
            gene_to_chromo[l[0].split('.')[0]] = l[1][:-1]




thousand_G_dir = os.path.join(data_dir, "LDREF")
one_KG_SNPs_dict = {}

for i in range(1, 23):
    chromo_dir = os.path.join(thousand_G_dir, "1000G.EUR.{}".format(i))
    (bim, fam, bed) = read_plink(chromo_dir, verbose=False)
    chromo_snp = np.array(bim['snp'])
    X = bed.compute().T   # columns as SNP and row as number of individuals
    X_df = pd.DataFrame(data=X, columns=chromo_snp)

    one_KG_SNPs_dict[str(i)] = X_df


# ## Randomly pick LD blocks


# first loading block information

snps_LD_blocks_dir = os.path.join(data_dir, "LD_blocks", "snps2LDblock.csv")
snps_LD_blocks = {}

with open(snps_LD_blocks_dir) as f:
    f.readline()
    for l in f:
        l = l.rstrip().split(",")

        if int(l[1]) not in snps_LD_blocks:
            snps_LD_blocks[int(l[1])] = []

        snps_LD_blocks[int(l[1])].append(l[0] )


genes_LD_blocks_dir = os.path.join(data_dir, "LD_blocks", "genes2LDblock.csv")
genes_LD_blocks = {}

with open(genes_LD_blocks_dir) as f:
    f.readline()
    for l in f:
        l = l.rstrip().split(",")

        if int(l[1]) not in genes_LD_blocks:
            genes_LD_blocks[int(l[1])] = []

        genes_LD_blocks[int(l[1])].append(l[0].split('.')[0])



# sampling 1KG individual's index for each block
block_eQTL_index = {}
block_GWAS_index = {}
block_TWAS_index = {}
block_Test_index = {}

# select blocks that have both gene and snps
selected_blocks = np.random.choice(np.intersect1d(list(snps_LD_blocks.keys()), list(genes_LD_blocks.keys())),
                                   size=num_block, replace=False)

# sampling 1KG index for each block
for b in selected_blocks:
    eQTL_index = np.concatenate((np.arange(N_1KG), np.random.randint(N_1KG, size=(N_eQTL - N_1KG))))
    np.random.shuffle(eQTL_index)
    block_eQTL_index[b] = eQTL_index

    GWAS_index = np.concatenate((np.arange(N_1KG), np.random.randint(N_1KG, size=(N_GWAS - N_1KG))))
    np.random.shuffle(GWAS_index)
    block_GWAS_index[b] = GWAS_index

    TWAS_index = np.concatenate((np.arange(N_1KG), np.random.randint(N_1KG, size=(N_TWAS - N_1KG))))
    np.random.shuffle(TWAS_index)
    block_TWAS_index[b] = TWAS_index

    Test_index = np.concatenate((np.arange(N_1KG), np.random.randint(N_1KG, size=(N_Test - N_1KG))))
    np.random.shuffle(Test_index)
    block_Test_index[b] = Test_index



selected_snps = set()
selected_genes = set()

snps_block_GWAS_index = {}
snps_block_TWAS_index = {}
snps_block_Test_index = {}

gene_block_eQTL_index = {}
gene_block_GWAS_index = {}
gene_block_TWAS_index = {}
gene_block_Test_index = {}


for b in selected_blocks:
    selected_snps.update(snps_LD_blocks[b])
    # keep track of the block index for snp and genes
    for snp in snps_LD_blocks[b]:
        snps_block_GWAS_index[snp] = block_GWAS_index[b]
        snps_block_TWAS_index[snp] = block_TWAS_index[b]
        snps_block_Test_index[snp] = block_Test_index[b]

    for gn in genes_LD_blocks[b]:
        gene_file = fnmatch.filter(os.listdir(gene2snps_dir), '*{}*'.format(gn))
        if len(gene_file) != 0:
            selected_genes.update([gn])
            gene_block_eQTL_index[gn] = block_eQTL_index[b]
            gene_block_GWAS_index[gn] = block_GWAS_index[b]
            gene_block_TWAS_index[gn] = block_TWAS_index[b]
            gene_block_Test_index[gn] = block_Test_index[b]


selected_snps = np.array(list(selected_snps))
selected_genes = np.array(list(selected_genes))
snp_info_df = snp_info_df.loc[selected_snps]


print("number of SNPs: {}".format(selected_snps.shape[0]))
print("number of genes: {}".format(selected_genes.shape[0]))




# identify which chromosome is located for each selected snp
snp_to_ch_dict = {}
TWAS_genotype = []
Test_genotype = []


for ch in one_KG_SNPs_dict:
    ch_SNPs = one_KG_SNPs_dict[ch].columns  # getting the SNPs in the chromosome
    overlap_snps = np.intersect1d(selected_snps, ch_SNPs)  # selecting those SNPs that are sampled
    for snp in overlap_snps:
        snp_to_ch_dict[snp] = ch

    if len(overlap_snps) != 0:
        TWAS_genotype_df = one_KG_SNPs_dict[ch].loc[snps_block_TWAS_index[overlap_snps[0]], overlap_snps]
        TWAS_genotype_df = TWAS_genotype_df.set_index(np.arange(N_TWAS))
        TWAS_genotype.append(TWAS_genotype_df)

        Test_genotype_df = one_KG_SNPs_dict[ch].loc[snps_block_Test_index[overlap_snps[0]], overlap_snps]
        Test_genotype_df = Test_genotype_df.set_index(np.arange(N_Test))
        Test_genotype.append(Test_genotype_df)



# save the TWAS genotype
TWAS_whole_genome_df = pd.concat(TWAS_genotype, axis=1, join="inner")
TWAS_whole_genome_df.to_csv(os.path.join(simulated_data_dir, "genotype", "TWAS_genotype.csv.gz"))

# save the test genotype
Test_whole_genome_df = pd.concat(Test_genotype, axis=1, join="inner")
Test_whole_genome_df.to_csv(os.path.join(simulated_data_dir, "genotype", "test_genotype.csv.gz"))




# List of real gene expression of all tissues
GWAS_real_genes_expression = {}
TWAS_real_genes_expression = {}
Test_real_genes_expression = {}
genetic_causal_snps_dict = {}
phenotypic_causal_snps_dict = {}
eQTL_result_dir = os.path.join(simulated_data_dir, "eQTL")

# run_bash_cmd(f"find {eQTL_result_dir} -name \*.csv -type f -delete")

for t in range(num_tissue):
    run_bash_cmd("mkdir {}".format(os.path.join(eQTL_result_dir, f"tissue{t}")))

    GWAS_tissue_gene_expression = np.zeros((N_GWAS, selected_genes.shape[0]))
    TWAS_tissue_gene_expression = np.zeros((N_TWAS, selected_genes.shape[0]))
    Test_tissue_gene_expression = np.zeros((N_Test, selected_genes.shape[0]))

    for g, gene in enumerate(selected_genes):
        gene_file = fnmatch.filter(os.listdir(gene2snps_dir), '*{}*'.format(gene))[0]
        gene_SNPs = pd.read_csv(os.path.join(gene2snps_dir, gene_file), index_col=0).index # getting the gene SNPs
        intersect_SNPs = np.intersect1d(gene_SNPs, selected_snps)  # select the SNPs that has sampled

        if num_causal_snp_per_gene < 1:
            causal_snps_index = np.random.choice(np.arange(len(intersect_SNPs)),
                                                 size=np.ceil(num_causal_snp_per_gene*len(intersect_SNPs)).astype(np.int32),
                                                 replace=False)
        else:
            causal_snps_index = np.random.choice(np.arange(len(intersect_SNPs)), size=num_causal_snp_per_gene,
                                                 replace=False)


        if len(causal_snps_index) == 0:
            GWAS_tissue_gene_expression[:, g] = np.nan
            TWAS_tissue_gene_expression[:, g] = np.nan
            Test_tissue_gene_expression[:, g] = np.nan
            continue


        causal_snps = intersect_SNPs[causal_snps_index]

        if t == causal_tissue:
            genetic_causal_snps_dict[gene] = causal_snps

        ch = gene_to_chromo[gene] # which chromosome the gene is located

        # sampling real W and errors for gene
        W_g = np.random.normal(loc=0, scale=np.sqrt(h_g_2), size=len(causal_snps_index))

        eQTL_X_g_snp = one_KG_SNPs_dict[ch].loc[gene_block_eQTL_index[gene], causal_snps].values
        eQTL_X_g_snp = (eQTL_X_g_snp - eQTL_X_g_snp.mean(axis=0)) / eQTL_X_g_snp.std(axis=0)
        eQTL_e_g = np.random.normal(loc=0, scale=np.sqrt(1-h_g_2), size=N_eQTL)
        eQTL_A_g = eQTL_X_g_snp.dot(W_g) + eQTL_e_g
        eQTL_A_g = (eQTL_A_g - eQTL_A_g.mean(axis=0)) / eQTL_A_g.std(axis=0)


        # doing regression here (eQTL)
        X_for_regression = one_KG_SNPs_dict[ch].loc[gene_block_eQTL_index[gene], intersect_SNPs].values
        X_for_regression = (X_for_regression - X_for_regression.mean(axis=0)) / X_for_regression.std(axis=0)

        # regression = sm.OLS(eQTL_A_g, X_for_regression).fit() # this is for fast debugging
        regression = sm.OLS(eQTL_A_g, X_for_regression).fit_regularized(method='elastic_net')
        w_hat_df = pd.DataFrame(data=regression.params, index=intersect_SNPs, columns=["w_hat"])
        w_hat_df.to_csv(os.path.join(eQTL_result_dir, f"tissue{t}", f"{gene}.csv"))


        GWAS_X_g_snp = one_KG_SNPs_dict[ch].loc[gene_block_GWAS_index[gene], causal_snps].values
        GWAS_X_g_snp = (GWAS_X_g_snp - GWAS_X_g_snp.mean()) / GWAS_X_g_snp.std()
        GWAS_e_g = np.random.normal(loc=0, scale=np.sqrt(1-h_g_2), size=N_GWAS)
        GWAS_A_g = GWAS_X_g_snp.dot(W_g) + GWAS_e_g
        GWAS_A_g = (GWAS_A_g - GWAS_A_g.mean(axis=0)) / GWAS_A_g.std(axis=0)
        GWAS_tissue_gene_expression[:, g] += GWAS_A_g

        TWAS_X_g_snp = one_KG_SNPs_dict[ch].loc[gene_block_TWAS_index[gene], causal_snps].values
        TWAS_X_g_snp = (TWAS_X_g_snp - TWAS_X_g_snp.mean()) / TWAS_X_g_snp.std()
        TWAS_e_g = np.random.normal(loc=0, scale=np.sqrt(1-h_g_2), size=N_TWAS)
        TWAS_A_g = TWAS_X_g_snp.dot(W_g) + TWAS_e_g
        TWAS_A_g = (TWAS_A_g - TWAS_A_g.mean(axis=0)) / TWAS_A_g.std(axis=0)
        TWAS_tissue_gene_expression[:, g] += TWAS_A_g

        Test_X_g_snp = one_KG_SNPs_dict[ch].loc[gene_block_Test_index[gene], causal_snps].values
        Test_X_g_snp = (Test_X_g_snp - Test_X_g_snp.mean()) / Test_X_g_snp.std()
        Test_e_g = np.random.normal(loc=0, scale=np.sqrt(1-h_g_2), size=N_Test)
        Test_A_g = Test_X_g_snp.dot(W_g) + Test_e_g
        Test_A_g = (Test_A_g - Test_A_g.mean(axis=0)) / Test_A_g.std(axis=0)
        Test_tissue_gene_expression[:, g] += Test_A_g


        time.sleep(.1)
        updt(selected_genes.shape[0], g+1)


    GWAS_tissue_gene_expression_df = pd.DataFrame(data=GWAS_tissue_gene_expression, columns=selected_genes).dropna(axis=1, how="all")
    GWAS_real_genes_expression[t] = GWAS_tissue_gene_expression_df

    TWAS_tissue_gene_expression_df = pd.DataFrame(data=TWAS_tissue_gene_expression, columns=selected_genes).dropna(axis=1, how="all")
    TWAS_real_genes_expression[t] = TWAS_tissue_gene_expression_df

    Test_tissue_gene_expression_df = pd.DataFrame(data=Test_tissue_gene_expression, columns=selected_genes).dropna(axis=1, how="all")
    Test_real_genes_expression[t] = Test_tissue_gene_expression_df




# randomly pick causal genes
causal_tissue_genes = GWAS_real_genes_expression[causal_tissue].columns
causal_genes = np.random.choice(causal_tissue_genes,
                                size=np.ceil(num_causal_gene*len(causal_tissue_genes)).astype(np.int32),
                                replace=False)

with open(os.path.join(simulated_data_dir, "causality", "causal_tissue_and_gene.txt"), "w+") as f:
    print(f"causal tissue: {causal_tissue}", file=f)
    print(f"causal genes: {causal_genes}", file=f)



if not is_pleiotropy:
    alpha = np.random.normal(loc=0, scale=np.sqrt(sigma_g1), size=len(causal_genes))

    GWAS_causal_gene_expression = GWAS_real_genes_expression[causal_tissue][causal_genes].values
    GWAS_e_exp = np.random.normal(loc=0, scale=np.sqrt(1-sigma_g1), size=N_GWAS)
    GWAS_phenotype = GWAS_causal_gene_expression.dot(alpha) + GWAS_e_exp

    TWAS_causal_gene_expression = TWAS_real_genes_expression[causal_tissue][causal_genes].values
    TWAS_e_exp = np.random.normal(loc=0, scale=np.sqrt(1-sigma_g1), size=N_TWAS)
    TWAS_phenotype = TWAS_causal_gene_expression.dot(alpha) + TWAS_e_exp
    TWAS_phenotype_df = pd.DataFrame(TWAS_phenotype, columns=["phenotye"])
    TWAS_phenotype_df.to_csv(os.path.join(simulated_data_dir, "phenotype", "TWAS_phenotype.csv"))

    Test_causal_gene_expression = Test_real_genes_expression[causal_tissue][causal_genes].values
    Test_e_exp = np.random.normal(loc=0, scale=np.sqrt(1-sigma_g1), size=N_Test)
    Test_phenotype = Test_causal_gene_expression.dot(alpha) + Test_e_exp
    Test_phenotype_df = pd.DataFrame(Test_phenotype, columns=["phenotye"])
    Test_phenotype_df.to_csv(os.path.join(simulated_data_dir, "phenotype", "test_phenotype.csv"))

else:
    print("doing pleiotropy")

    GWAS_X_pleiotropy = None
    TWAS_X_pleiotropy = None
    Test_X_pleiotropy = None


    for gene in causal_genes:
        gene_file = fnmatch.filter(os.listdir(gene2snps_dir), '*{}*'.format(gene))[0]
        gene_SNPs = pd.read_csv(os.path.join(gene2snps_dir, gene_file), index_col=0).index # getting the gene SNPs
        intersect_SNPs = np.intersect1d(gene_SNPs, selected_snps)  # select the SNPs that has sampled

        causal_snps = np.random.choice(genetic_causal_snps_dict[gene], size=np.ceil(num_pleiotropy_snp_per_gene*len(intersect_SNPs)).astype(np.int32), replace=False)

        if len(causal_snps) == 0:
            continue

        ch = gene_to_chromo[gene] # which chromosome the gene is located

        GWAS_X_g_snp = one_KG_SNPs_dict[ch].loc[gene_block_GWAS_index[gene], causal_snps].values
        GWAS_X_g_snp = (GWAS_X_g_snp - GWAS_X_g_snp.mean()) / GWAS_X_g_snp.std()
        if GWAS_X_pleiotropy is None:
            GWAS_X_pleiotropy = GWAS_X_g_snp
        else:
            GWAS_X_pleiotropy = np.concatenate((GWAS_X_pleiotropy, GWAS_X_g_snp), axis=1)


        TWAS_X_g_snp = one_KG_SNPs_dict[ch].loc[gene_block_TWAS_index[gene], causal_snps].values
        TWAS_X_g_snp = (TWAS_X_g_snp - TWAS_X_g_snp.mean()) / TWAS_X_g_snp.std()
        if TWAS_X_pleiotropy is None:
            TWAS_X_pleiotropy = TWAS_X_g_snp
        else:
            TWAS_X_pleiotropy = np.concatenate((TWAS_X_pleiotropy, TWAS_X_g_snp), axis=1)


        Test_X_g_snp = one_KG_SNPs_dict[ch].loc[gene_block_Test_index[gene], causal_snps].values
        Test_X_g_snp = (Test_X_g_snp - Test_X_g_snp.mean()) / Test_X_g_snp.std()
        if Test_X_pleiotropy is None:
            Test_X_pleiotropy = Test_X_g_snp
        else:
            Test_X_pleiotropy = np.concatenate((Test_X_pleiotropy, Test_X_g_snp), axis=1)


    beta = np.random.normal(loc=0, scale=np.sqrt(sigma_g2), size=GWAS_X_pleiotropy.shape[1])

    GWAS_e_exp = np.random.normal(loc=0, scale=np.sqrt(1-sigma_g2), size=N_GWAS)
    GWAS_phenotype = GWAS_X_pleiotropy.dot(beta) + GWAS_e_exp

    TWAS_e_exp = np.random.normal(loc=0, scale=np.sqrt(1-sigma_g2), size=N_TWAS)
    TWAS_phenotype = TWAS_X_pleiotropy.dot(beta) + TWAS_e_exp
    TWAS_phenotype_df = pd.DataFrame(TWAS_phenotype, columns=["phenotye"])
    TWAS_phenotype_df.to_csv(os.path.join(simulated_data_dir, "phenotype", "TWAS_phenotype.csv"))

    Test_e_exp = np.random.normal(loc=0, scale=np.sqrt(1-sigma_g2), size=N_Test)
    Test_phenotype = Test_X_pleiotropy.dot(beta) + Test_e_exp
    Test_phenotype_df = pd.DataFrame(Test_phenotype, columns=["phenotye"])
    Test_phenotype_df.to_csv(os.path.join(simulated_data_dir, "phenotype", "test_phenotype.csv"))



z_hats = np.zeros(len(selected_snps))
p_values = np.zeros(len(selected_snps))
beta_hats = np.zeros(len(selected_snps))
bse = np.zeros(len(selected_snps))
ref_frq = np.zeros(len(selected_snps))

start = timer()
for s, snp in enumerate(selected_snps):
    ch = snp_to_ch_dict[snp]
    GWAS_X_g = one_KG_SNPs_dict[ch].loc[snps_block_GWAS_index[snp], snp].values
    ref_frq[s] = GWAS_X_g.sum() / (N_GWAS * 2)

    GWAS_X_g = (GWAS_X_g - GWAS_X_g.mean()) / GWAS_X_g.std()
    regression = sm.OLS(GWAS_phenotype, GWAS_X_g).fit()
    z_hat = regression.params[0] / regression.bse[0]
    pv = norm.pdf(z_hat)
    z_hats[s] += z_hat
    p_values[s] += pv
    beta_hats[s] += regression.params[0]
    bse[s] += regression.bse[0]

end = timer()
print("time: {}".format(end - start))


GWAS_result_dir = os.path.join(simulated_data_dir, "GWAS")
SNP_df = pd.DataFrame(data={"rsid": selected_snps, "Z": z_hats, "PVAL": p_values,
                            "BETA" : beta_hats, "SE": bse, "N": [N_GWAS]*len(selected_snps),
                            "REF_FRQ": ref_frq})
SNP_df.index = SNP_df["rsid"]
SNP_df = pd.concat([SNP_df, snp_info_df[["start", "Chrom", "A1", "A2"]]], axis=1, join='inner')
SNP_df.rename(columns={"rsid": "SNP_ID", "start": "POS", "A1": "REF", "A2": "ALT", "Chrom": "CHR"}, inplace=True)
SNP_df.head(10)
SNP_df.to_csv(os.path.join(GWAS_result_dir, "simulated_trait.sumstats"), sep='\t', index=False)
