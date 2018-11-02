# Python implementation of the uniFrac distance measure

Inspired by [Quantitative assessment of cell population diversity in single-cell landscapes. PLOS Biology 2018](https://journals.plos.org/plosbiology/article?id=10.1371%2Fjournal.pbio.2006687).

## Requirements
- [ETE](http://etetoolkit.org/): `conda install -c etetoolkit ete3 ete_toolchain`
- scipy, numpy, pandas

## Usage
```python
from sklearn.datasets import make_blobs
import pandas as pd

# create some data
X,y  = make_blobs()
df_meta = pd.DataFrame()
df_meta['label'] = y
df_meta.index = [str(_) for _ in df_meta.index]  # only working with str in the index for now

import unifrac
UF = unifrac.UniFrac(X, df_meta)
UF.build_tree(method='average', metric='correlation')
UF.cluster(n_clusters=3)

# form to groups of samples and check if they are 'uniformly' spread across the tree (small distance)
# or if they localize to certain parts of the tree (high distance)
# since our groups correspond to the cluster, they should be far apart in uniFrac distance
group1 = df_meta.query('label==0').index.values
group2 = df_meta.query('label==1').index.values
d, drand, pvalue = UF.unifrac_distance(group1, group2, randomization=100)
print(f'UniFrac distance: {d}\nRandomimzed distance {drand.mean()}+/-{drand.std()}\np={pvalue}')```