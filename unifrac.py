from ete3 import Tree, TreeStyle, TextFace
from scipy.cluster.hierarchy import linkage, dendrogram, cut_tree, to_tree
import numpy as np
from random import shuffle
from scipy import stats


def linkage_to_newick(Z, labels):
    """
    Input :  Z = scipy.cluster.hierarchy.linkage matrix, labels = leaf labels
    Output:  Newick formatted tree string
    """
    tree = to_tree(Z, False)

    def buildNewick(node, newick, parentdist, leaf_names):
        if node.is_leaf():
            return f"{leaf_names[node.id]}:{(parentdist - node.dist)/2}{newick}"
        else:
            if len(newick) > 0:
                newick = f"):{(parentdist - node.dist)/2}{newick}"
            else:
                newick = ");"
            newick = buildNewick(node.get_left(), newick, node.dist, leaf_names)
            newick = buildNewick(node.get_right(), f",{newick}", node.dist, leaf_names)
            newick = f"({newick}"
            return newick
    return buildNewick(tree, "", tree.dist, labels)


def find_cluster_roots(the_tree):
    """
    also annotates the distance to the root, we will need this in uniFRAC
    """
    n_clusters = len(set([leaf.clustering for leaf in the_tree.get_leaves()]))
    ROOT = the_tree.get_tree_root()
    cluster_roots = []
    for i in range(n_clusters):
        datapoints_in_cluster = the_tree.search_nodes(clustering=i)
        ancestor = the_tree.get_common_ancestor(datapoints_in_cluster)
        d2root = ancestor.get_distance(ROOT)
        ancestor.add_features(distance2root=d2root)
        cluster_roots.append(ancestor)
    return cluster_roots


class UniFrac(object):
    """
    note that the whole association between metadata and leave nodes works by .loc:
    the nodes are named according to the dataframe index and we look up a nodes metadata with df_metadata.loc[node.name]
    """

    def __init__(self, datamatrix, df_metadata):
        super(UniFrac, self).__init__()

        "make sure that the dataframe index is unique"
        assert len(df_metadata) == len(set(df_metadata.index)), 'row-index is not unique, but we need uniqueness to associate the metadata with the leaves in the tree'

        self.datamatrix = datamatrix
        self.df_metadata = df_metadata
        self.tree = None
        self._linkage = None  # just kept to do the cut_trees call
        self.cluster_roots = None  # just kept to do the cut_trees call
        self.nodes2leaves = None  # for caching leaf lookups, however this return a set!!

    def _update_leave_metadata(self):
        "puts the metadata in self.metadata as features into the trees leaves"
        assert self.tree

        # to speed things up, query the dataframe only once
        leaves = self.tree.get_leaves()
        leavenames = [leave.name for leave in leaves]
        meta = self.df_metadata.loc[leavenames].values  # sorts the metadata in the same order as leavenames
        featurenames = self.df_metadata.columns.values
        for i, leaf in enumerate(leaves):
            leaf.add_features(**dict(zip(featurenames, meta[i,:])))
            #TODO not sure if this overwrites previous features (thats what i want) or just adds additional features!

    def build_tree(self, method, metric):
        """
        constructs the hierarchical clustering tree, 
        but no clustering (corresponding to some tree pruning) in here
        """
        self._linkage = linkage(self.datamatrix, method=method, metric=metric)
        # turn it into a ete tree
        leave_labels = self.df_metadata.index.values
        newick_tree = linkage_to_newick(self._linkage, labels=leave_labels)
        self.tree = Tree(newick_tree)
        self.nodes2leaves = self.tree.get_cached_content()  # makes it easy to lookup leaves of a node
        # populate the leaves with metadatqa
        self._update_leave_metadata()

    def cluster(self, n_clusters):
        """
        prunes the hierarchical clustering tree to get clusters of data
        this clustering is also added to the metadata
        also adds the self.cluster_roots (caching it, we need it in unifrac calls)
        """
        assert self.tree
        clustering_prune = cut_tree(self._linkage, n_clusters)
        self.df_metadata['clustering'] = clustering_prune
        self._update_leave_metadata()
        self.cluster_roots = find_cluster_roots(self.tree)
        
        for i, cluster_root in enumerate(self.cluster_roots):
            cluster_root.add_features(**{'is_cluster_root': i, 'n_datapoints': len(self.nodes2leaves[cluster_root])})


    def unifrac_distance(self, group1, group2, randomization=None):
        """
        calculates the uniFrac distance of the two sample-groups
        group1: list of nodenames (i.e. indices of the metadata)
        group2: ---"--- 
        randomization: (int) how many times to compute the 'randomized' uniFrac distance to get a pvalue
        """
        assert 'clustering' in self.df_metadata.columns and self.cluster_roots, "run cluster() first"

        # all_leaves = self.tree.get_leaves()  # TODO this is a performance hog
        the_Root = self.tree.get_tree_root()
        all_leaves = self.nodes2leaves[the_Root]  # for performance reasons this is better then the line above

        # make sure all group elements are in hte tree
        leaf_names = [_.name for _ in all_leaves]
        assert all([_ in leaf_names for _ in group1])
        assert all([_ in leaf_names for _ in group2])

        # t.get_leaves_by_name(group1)
        group1_nodes = set([_ for _ in all_leaves if _.name in group1])  # sets for faster `in` lookup
        group2_nodes = set([_ for _ in all_leaves if _.name in group2])  # TODO replace by search_nodes?!

        the_distance = self._unifrac_dist(group1_nodes, group2_nodes)

        if randomization and randomization > 0:
            G1 = len(group1_nodes)
            G2 = len(group2_nodes)
            all_nodes = list(group1_nodes | group2_nodes)  # union, but turn into list for partioning later
            randomized_distances = []

            for i in range(randomization):
                shuffle(all_nodes)  # inplace shuffle
                group1_nodes_random = set(all_nodes[:G1])
                group2_nodes_random = set(all_nodes[G1:])
                randomized_distances.append(self._unifrac_dist(group1_nodes_random, group2_nodes_random))

            randomized_distances = np.array(randomized_distances)

            # pvalue
            p = 1 - stats.norm(loc=randomized_distances.mean(-1), scale=randomized_distances.std(-1)).cdf(the_distance)
            p2 = np.sum(randomized_distances > the_distance) / len(randomized_distances)
            # print(p, p2)
            return the_distance, randomized_distances, p2
        else:
            return the_distance

    def _unifrac_dist(self, group1_nodes, group2_nodes):
        "given two node lists, calculate the unifrac distance"
        At, Bt = len(group1_nodes), len(group2_nodes)
        nom = {}
        denom = {}
        for i, current_cluster_root in enumerate(self.cluster_roots):
            leafs = list(self.nodes2leaves[current_cluster_root])  # all the datapoitns in the cluster
            Ai = len([_ for _ in leafs if _ in group1_nodes])
            Bi = len([_ for _ in leafs if _ in group2_nodes])
            distance2root = current_cluster_root.distance2root  # cached already
            nom[i] = distance2root * np.abs(Ai / At - Bi / Bt)
            denom[i] = distance2root * np.abs(Ai / At + Bi / Bt)

        n_clusters = len(nom)
        summed_nom = sum([nom[i] for i in range(n_clusters)])
        summed_denom = sum([denom[i] for i in range(n_clusters)])
        unifrac_distance = summed_nom / summed_denom
        return unifrac_distance

    def visualize(self, group1=None, group2=None):
        import matplotlib
        import matplotlib.pyplot as plt

        # annotate the cluster roots with their fractions
        if group1 or group2:
            for i, cluster_root in enumerate(self.cluster_roots):
                # count downstream conditions in the leafs
                datapoints_in_cluster = list(self.nodes2leaves[cluster_root])
                cluster_root.add_face(TextFace(f"Group1: {len(group1)}// Group2:{len(group2)}"), column=0, position="branch-right")

        def _custom_layout(node):
            cmap_cluster = plt.cm.tab10(np.linspace(0,1,len(self.cluster_roots)))
            cmap_treated = plt.cm.viridis(np.linspace(0,1,2))

            if node.is_leaf():
                c_cluster = matplotlib.colors.rgb2hex(cmap_cluster[node.clustering,:])
                c_treat = matplotlib.colors.rgb2hex(cmap_treated[node.treated,:])
                node.img_style["fgcolor"] = c_treat
                node.img_style["bgcolor"] = c_cluster

            if 'is_cluster_root' in node.features:
                c_cluster = matplotlib.colors.rgb2hex(cmap_cluster[node.is_cluster_root,:])
                node.img_style["bgcolor"] = c_cluster
                node.img_style["draw_descendants"] = False
                node.add_face(TextFace(f"#data:{node.n_datapoints}"), column=0, position="branch-right")

        ts = TreeStyle()
        ts.mode = "r"
        ts.show_leaf_name = False
        ts.arc_start = -180 # 0 degrees = 3 o'clock
        ts.arc_span = 270
        ts.layout_fn = _custom_layout
        self.tree.show(tree_style=ts)
