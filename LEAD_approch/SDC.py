import json
import torch
from sklearn.cluster import KMeans
import stanza
import networkx as nx
from grakel.utils import graph_from_networkx
from grakel.kernels import WeisfeilerLehman, VertexHistogram
from tqdm import tqdm

nlp = stanza.Pipeline(
    lang='zh',
    processors='tokenize,pos,lemma,depparse',
    dir="",
    download_method=None
)
input_file = ''
output_file = ''
filtered_output_file = ''

with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

graphs = []
instructions_and_outputs = []
for item in tqdm(data, desc="Processing sentences", ncols=100):
    sentence = item['instruction'] + " " + item['output']
    instructions_and_outputs.append(sentence)

    doc = nlp(sentence)
    sent_graph = nx.Graph()

    for word in doc.sentences[0].to_dict():
        if isinstance(word['id'], tuple):
            continue

        if word['id'] not in sent_graph.nodes():
            sent_graph.add_node(word['id'])
        sent_graph.nodes[word['id']]['label'] = word['upos']

        if word['head'] not in sent_graph.nodes():
            sent_graph.add_node(word['head'])
        sent_graph.add_edge(word['id'], word['head'])

    sent_graph.nodes[0]['label'] = 'none'
    graphs.append(sent_graph)

G = list(graph_from_networkx(graphs, node_labels_tag='label'))

gk = WeisfeilerLehman(n_iter=2, normalize=True, base_graph_kernel=VertexHistogram)
K = gk.fit_transform(G)

K = torch.tensor(K).to(torch.float).to("cuda:0")
K_cpu = K.cpu().numpy()

kmeans = KMeans(n_clusters=130, random_state=42)
kmeans.fit(K_cpu)

labels = kmeans.labels_


for i, item in enumerate(tqdm(data, desc="Assigning labels", ncols=100)):
    item['id'] = int(labels[i])
with open(output_file, 'w', encoding='utf-8') as f_out:
    json.dump(data, f_out, ensure_ascii=False, indent=4)

print(f"Results written to {output_file}")

sorted_data = sorted(data, key=lambda x: (x['id'], x['LETS']), reverse=True)
filtered_data = []
last_label = -1
count = 0

for item in sorted_data:
    if item['id'] != last_label:
        last_label = item['id']
        count = 1
        filtered_data.append(item)
    elif count < 20:
        count += 1
        filtered_data.append(item)

with open(filtered_output_file, 'w', encoding='utf-8') as f_filtered:
    json.dump(filtered_data, f_filtered, ensure_ascii=False, indent=4)

print(f"Filtered results written to {filtered_output_file}")
print(f"Filtered data contains {len(filtered_data)} items.")