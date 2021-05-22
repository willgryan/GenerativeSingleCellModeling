import numpy as np
import scanpy as sc

# preprocessing method by Turecki et al. implemented in scanpy
if __name__ == '__main__':
    adata = sc.read_10x_mtx("GSE144136")

    # modified seurat recipe
    sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, min_mean=0.003, max_mean=2, min_disp=1)
    adata = adata[:, adata.var.highly_variable]
    sc.pp.regress_out(adata, ['total_counts'])
    sc.pp.scale(adata, max_value=10)

    # sanity checks
    print(adata)
    print(np.min(adata.X, axis=1))
    print(np.max(adata.X, axis=1))

    adata.write('data/GSE144136_preprocessed.h5ad', compression="gzip")
