import spacy
import networkx as nx
import matplotlib.pyplot as plt

nlp = spacy.load("en_core_web_sm")

class ProbabilisticGraph:
    def __init__(self):
        self.graph = nx.MultiDiGraph()

    def add_edges_from_svo(self, svo_triples):
        for s, v, o in svo_triples:
            self.graph.add_edge(s, o, relation=v)

    def extract_svo_triples(self, sentence):
        doc = nlp(sentence)
        svo_triples = []
        for token in doc:
            if token.dep_ in ["nsubj", "nsubjpass"]:
                subject = token.text
                verb = token.head.text
                for child in token.head.children:
                    if child.dep_ in ["dobj", "pobj", "attr", "oprd"]:
                        object = child.text
                        svo_triples.append((subject, verb, object))
        return svo_triples

    def draw_graph(self):
        pos = nx.spring_layout(self.graph)
        
        nx.draw_networkx_nodes(self.graph, pos, node_size=7000, node_color='lightblue', alpha=0.6)
        
        edge_labels = dict([((u, v,), d['relation'])
                            for u, v, d in self.graph.edges(data=True)])
        nx.draw_networkx_edges(self.graph, pos, arrowstyle='->', arrowsize=20)
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels)
        
        nx.draw_networkx_labels(self.graph, pos, font_size=12, font_family="sans-serif")
        
        plt.axis('off')
        plt.show()

def update_global_context(global_context, sentences):
    for sentence in sentences:
        svo_triples = global_context.extract_svo_triples(sentence)
        global_context.add_edges_from_svo(svo_triples)