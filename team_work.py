import plwn
import networkx as nx

wn = plwn.load('./data/default_model')

G = nx.read_graphml('./data/graph_synset.xml')

# simlex unique words
simlex = pd.read_csv('./data/MSimLex999_Polish.txt', sep='\t', header=None)
unique_words = set(list(simlex['word1'].unique()) + list(simlex['word2'].unique()))

# synset = wn.synset('pies', plwn.PoS.noun_pl, 1)

def get_sysnset_id(word):
    synset_id = None
    try:
        synset_id = wn.synset(word, plwn.PoS.noun, 1).to_dict()['id']
    except plwn.exceptions.SynsetNotFound:
        pass
    if not synset_id:
        try:
            synset_id = wn.synset(word, plwn.PoS.verb, 1).to_dict()['id']
        except plwn.exceptions.SynsetNotFound:
            pass
    if not synset_id:
        try:
            synset_id = wn.synset(word, plwn.PoS.adjective, 1).to_dict()['id']
        except plwn.exceptions.SynsetNotFound:
            pass
    if not synset_id:
        try:
            synset_id = wn.synset(word, plwn.PoS.adverb, 1).to_dict()['id']
        except plwn.exceptions.SynsetNotFound:
            pass

    if synset_id:
        synset_id = (word, synset_id)
    
    return synset_id

from multiprocessing import Pool

with Pool(7) as p:
    ids = p.map(get_sysnset_id, unique_words)

ids = [i if i is not None for i in ids]


from multiprocessing import Pool
import tqdm

pool = Pool(processes=8)
for _ in tqdm.tqdm(pool.imap_unordered(do_work, tasks), total=len(tasks)):
    pass


# get subgraph
ids = list(ids_dict.values())
subG = G.subgraph(ids)
nx.write_edgelist(subG, "sub_graph.edgelist")

ensure_ascii=False

filtered_edges = [edge for edge in list(G.edges) if (edge[2].endswith('hiperonimia')) or (edge[2].endswith('hiponimia'))]

subG = G.edge_subgraph(filtered_edges)

[node[0] for node in list(G.in_degree()) if node[1] == 0]

subG = nx.DiGraph()
for edge in tqdm(filtered_edges):
    subG.add_edge(edge[0], edge[1], label=edge[2])


# cykle
from networkx.algorithms.cycles import cycle_basis
cycle_basis(subG)


# wu palmer

from networkx.algorithms.lowest_common_ancestors import lowest_common_ancestor

def wu_palmer(node_1, node_2):
    
    lso = lowest_common_ancestor(subG, node_1, node2)

    a = 2 * len(nx.shortest_path(subG, 'root', lso))
    b = len(nx.shortest_path(subG.to_undirected(), node_1, lso)) + len(nx.shortest_path(subG.to_undirected(), node_2, lso)) + a

    return a/b



# l c normalized path

nodes_depth = [(node, len(nx.shortest_path(subG, 'root', node))) for node in subG.nodes()]
nodes_depth.sort(key=lambda el: el[1])

def lc_normalized_path(node_1, node_1):
    a = len(nx.shortest_path(subG.to_undirected(), node_1, node_2))
    b = 2 * max_depth

    return -math.log(a/b)

from networkx.algorithms.minors import contracted_nodes

true_node = ''
false_nodes = ['']
for n in false_nodes:
    subG = contracted_nodes(subG, true_node, n)

NEW CYCLE
podjąć
zająć się
zrobić
zacząć


NEW CYCLE
robić
podejmować
zajmować się


words_in_graph = set(ids_dict.keys())
filtered_simlex = simlex[(simlex['word1'].isin(words_in_graph)) & (simlex['word2'].isin(words_in_graph)]


def add_wp(simlex_row):
    node_1 = str(ids_dict[simlex_row['word1']])
    node_2 = str(ids_dict[simlex_row['word2']])

    try:
        simlex_row['wu_palmer'] = wu_palmer(node_1, node_2)
    except Exception as e:
        print(e)
        simlex_row['wu_palmer'] = None

    return simlex_row


def add_lcn(simlex_row):
    node_1 = str(ids_dict[simlex_row['word1']])
    node_2 = str(ids_dict[simlex_row['word2']])

    try:
        simlex_row['leacon'] = lc_normalized_path(node_1, node_2)
    except Exception as e:
        print(e)
        simlex_row['leacon'] = None

    return simlex_row



tqdm.pandas()
filtered_simlex = filtered_simlex.progress_apply(add_wp, axis=1)
filtered_simlex = filtered_simlex.progress_apply(add_lcn, axis=1)