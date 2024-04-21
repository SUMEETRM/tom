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
                verb = token.head.lemma_ 
                for child in token.head.children:
                    if child.dep_ in ["dobj", "pobj", "attr", "oprd"]:
                        object = child.text
                        svo_triples.append((subject, verb, object))
        return svo_triples

    def draw_graph(self):
        pos = nx.spring_layout(self.graph)
        plt.figure(figsize=(8, 6))
        nx.draw(self.graph, pos, with_labels=True, node_size=3000, node_color='lightblue', arrows=True)
        edge_labels = {(u, v): d['relation'] for u, v, d in self.graph.edges(data=True)}
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels)
        plt.title("Probabilistic Graph")
        plt.axis('off')
        plt.show()

class ProbabilisticGraphsContainer:
    def __init__(self):
        self.local_contexts = {}

    def add_character(self, character):
        self.local_contexts.setdefault(character, ProbabilisticGraph())

    def propagate_knowledge(self, global_context, witnesses):
        for character in witnesses:
            if character in self.local_contexts:
                local_context = self.local_contexts[character]
                for edge in global_context.graph.edges(data=True):
                    local_context.graph.add_edge(edge[0], edge[1], relation=edge[2]['relation'])

def update_global_and_extract_svo(global_context, sentences):
    for sentence in sentences:
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
        global_context.add_edges_from_svo(svo_triples)

def extract_characters(sentence):
    doc = nlp(sentence)
    characters = []
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            characters.append(ent.text)
    return characters

def update_local_contexts(global_context, local_contexts, story_sentences):
    current_characters = set()
    for i, sentence in enumerate(story_sentences):
        characters = extract_characters(sentence)
        
        #SVO triples
        svo_triples = []
        doc = nlp(sentence)
        for token in doc:
            if token.dep_ in ["nsubj", "nsubjpass"]:
                subject = token.text
                verb = token.head.text
                for child in token.head.children:
                    if child.dep_ in ["dobj", "pobj", "attr", "oprd"]:
                        object = child.text
                        svo_triples.append((subject, verb, object))
        
        #update local context graphs of the characters in the current sentence
        for character in characters:
            local_contexts.add_character(character)
            local_context = local_contexts.local_contexts[character]
            local_context.add_edges_from_svo(svo_triples)
        
        #remove characters that exit
        if "exits" in sentence.lower():
            for character in characters:
                current_characters.discard(character)
        else:
            #add characters that enter
            current_characters.update(characters)
        
        #global to local (for characters in the current sentence)
        for character in current_characters:
            if character in local_contexts.local_contexts:
                local_context = local_contexts.local_contexts[character]
                local_context.add_edges_from_svo(svo_triples)