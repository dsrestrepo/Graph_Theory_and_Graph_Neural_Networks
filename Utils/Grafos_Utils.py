#!/usr/bin/env python
# coding: utf-8

import networkx as nx
# import graphviz
import matplotlib.pyplot as plt
from networkx.readwrite import json_graph
import json
import numpy as np

""" 
Funcion para eliminar duplicados en una lista 
Function to eliminate duplicates in a list
"""

def eliminar_duplicados(lista):
    
    lista_final = []
    
    for indice, item in enumerate(lista):
        
        # Verificamos si el item no está entre los valores previos de la lista
        # We check if the item is not among the previous values of the list
        if item not in lista[:indice]:
            # Si no es así añadimos el item
            # If item not in list we add the item
            lista_final.append(item)
    
    return lista_final


""" 
Funcion que determina el número de relaciones transitivas en un arreglo de relaciones R

Entradas:
R: Lista de vertices (Edges) en el formato (Node1, Node2)
verbose = Ture para mostrar detalles de la función

Salidas:
Lista con todas las relaciones transitivas en el grafo

----- ENG:
Function that determines the number of transitive relations in an array of rrelations R

Inputs:
R: List of vertices (Edges) in the format (Node1, Node2)
verbose = Ture to display function details

Outputs:
List with all transitive relations in the graph
"""
def es_transitiva(R, verbose=None):
    transitiva = []
    for relacion_1 in R:
        for relacion_2 in R:
            # No queremos analizar relaciones iguales
            # We don't want to analyze self-loops
            if relacion_1 == relacion_2:
                continue
            # Igualamos el segundo y la primer elemento de las relaciones
            # First and second element in a relation should be the same
            if relacion_1[1] == relacion_2[0]:
                # Si se cumple que: relacion_1 = (a, b) & relacion_2 = (b, c)
                # Creamos la relacion_3 = (a, c)
                relacion_3 = (relacion_1[0], relacion_2[1])
                # Validamos si la relación_3 (a, c) se encuentra en R:
                if relacion_3 in R:
                    if verbose:
                        print(f'Relación {relacion_3} está en R, por lo tanto {relacion_1, relacion_2} es transitiva')
                    transitiva.append([relacion_1, relacion_2, relacion_3])
                else:
                    if verbose:
                        print(f'La relación {relacion_3} no está en R, por lo tanto es no Transitiva')
                    
    # Nos quedamos con las relaciones transitivas y eliminamos duplicados:
    transitiva = eliminar_duplicados(transitiva)
    
    if verbose:
        print(f'El número de relaciones transitivas (transitive relations): {len(transitiva)}')
    
    return transitiva


""" 
Mostrar grafo
Plot Graph 
"""

def plot_graph(G, k=0.5, labels=True, color='skyblue', figsize=(12,12)):
    plt.figure(figsize=figsize)
    # k regulates the distance between nodes
    pos = nx.spring_layout(G, k=k)
    nx.draw(G, with_labels=labels, node_color=color, node_size=1500, edge_cmap=plt.cm.Blues, pos = pos)
    plt.show()



""" 
Generar Grafo 
Generate Graph
"""
#!pip install networkx
#!pip install graphviz
#%conda install pygraphviz

def generar_grafo(nodos, aristas, info=True, plot=True):
    G = nx.Graph() 
 
    #Añadir nodos
    G.add_nodes_from(nodos)
 
    #Añadir aristas
    G.add_edges_from(aristas)
    
    if info:
        print("====== Nodos (Nodes): =====")
        print(f'El número de nodos es (# of nodes): {len(G.nodes)}')
        print(f'Nodos (Nodes):')
        print(G.nodes)
        
        print("====== Aristas (Edges): =====")
        print(f'El número de aristas es (# of edges): {len(G.edges)}')
        print(f'Aristas (edges):')
        print(G.edges)

    if plot:
        plot_graph(G)
    
    return G


def relacion_bidireccional(edges):
    edges = list(edges)
    bidirectional = []
    for edge in edges:
        if ((edge[1], edge[0]) in edges) and (edge[1] != edge[0]):
            bidirectional.append(edge)
    return bidirectional

def auto_vertice(edges):
    edges = list(edges)
    auto = []
    for edge in edges:
        if edge[1] == edge[0]:
            auto.append(edge)
    return auto


def informacion_de_vertices(edges):
    print(f'El número de vertices (edges): {len(edges)}')

    bidirectional = relacion_bidireccional(edges)
    print(f'El número de vertices bidireccionales (bidirectional edges) {len(bidirectional)}')

    auto = auto_vertice(edges)
    print(f'El número de relaciones de nodos con ellos mismos es (self-loops) {len(auto)}')

    print(f""" De los {len(edges)} vertices, {len(bidirectional) + len(auto)} son realciones bidireccionales o con ellos mismos""")
    
    return bidirectional, auto


# Para leer los datos de las proteinas
class Dataset():
    """A toy Protein-Protein Interaction network dataset.

    Adapted from https://github.com/williamleif/GraphSAGE/tree/master/example_data.

    The dataset contains 24 graphs. The average number of nodes per graph
    is 2372. Each node has 50 features and 121 labels.

    We use 20 graphs for training, 2 for validation and 2 for testing.
    """
    def __init__(self, mode, PATH):
        """Initialize the dataset.

        Paramters
        ---------
        mode : str
            ('train', 'valid', 'test').
        """
        self.mode = mode

        with open(f'{PATH}/raw/{self.mode}_graph.json') as jsonfile:
            self.g_data = json.load(jsonfile)
            
        self.labels = np.load(f'{PATH}/raw/{self.mode}_labels.npy')
        self.features = np.load(f'{PATH}/raw/{self.mode}_feats.npy')
        self.graph_id = np.load(f'{PATH}/raw/{self.mode}_graph_id.npy')
        
        self.graph = nx.DiGraph(json_graph.node_link_graph(self.g_data))

    def subgrafo(self, size=0):
        if size == 0:
            index = np.unique(self.graph_id)[0]
            subset = np.sum(self.graph_id == index)
            subset = range(0, subset)
        elif size == 1:
            index = np.unique(self.graph_id)[1]
            subset = np.sum(self.graph_id == index)
            subset = range(subset, self.graph_id.shape[0])
        elif size < 0 or size > self.graph_id.shape[0]:
            print('Error, el tamaño del subgrafo debe ser mayor a 0 y menor a el grafo total')
            return 0
        else:
            subset = range(0, size)

        subgraph = self.graph.subgraph(subset)
        return subgraph

""" 
Distribución de probabilidad 
Probability Distrobution
1. Probability Density Function (PDF)
2. Complementary Cumulative Distribution Function (CCDF)
"""
def distribucion_probabilidades(valores_grado):
    pdf = []
    ccdf = []
    grado_indice = []
    acumulado = 0

    max_d = valores_grado.max()

    # Cacular la probabilidad para cada grado posible (Calculate the probability for each degree)
    for grado in range(0, max_d+1):
        
        grado_indice.append(grado)
        
        # PDF:
        # Cacular los nodos con grado igual a "grado"
        # Calculate the nodes with degree "grado"
        valor_grado = (valores_grado == grado).sum()
        probabilidad = valor_grado/len(valores_grado)
        pdf.append(probabilidad)
        
        # CCDF
        acumulado += valor_grado
        probabilidad_acumulada = acumulado/len(valores_grado)
        ccdf.append(probabilidad_acumulada)

    return pdf, ccdf, grado_indice

"""
Graficar PDF
Plot PDF
"""
def plot_pdf(pdf, grado_indice, x=None, y=None):
    grid = sns.lineplot(x=grado_indice, y=pdf)
    grid.set(xlim=0)

    if x=='log' and y=='log':
        grid.set(xscale="log", yscale="log")
        plt.xlabel('Grado [log]')
        plt.ylabel('Probabilidad [log]')
        plt.title('PDF log - log')

    elif x=='log':
        grid.set(xscale="log")
        plt.xlabel('Grado [log]')
        plt.ylabel('Probabilidad')
        plt.title('PDF linear - log')
    elif y=='log':
        grid.set(yscale="log")
        plt.xlabel('Grado')
        plt.ylabel('Probabilidad [log]')
        plt.title('PDF log - linear')
    else:
        plt.xlabel('Grado')
        plt.ylabel('Probabilidad')
        plt.title('PDF')
    grid.set(xlim=0)
    plt.show()

"""
Graficar ECDF
Plot ECDF
"""
def plot_ecdf(ecdf, grado_indice, x=None, y=None):
    grid = sns.lineplot(x=grado_indice, y=ecdf)
    grid.set(xlim=0)

    if x=='log' and y=='log':
        grid.set(xscale="log", yscale="log")
        plt.xlabel('Grado [log]')
        plt.ylabel('Probabilidad [log]')
        plt.title('ECDF log - log')

    elif x=='log':
        grid.set(xscale="log")
        plt.xlabel('Grado [log]')
        plt.ylabel('Probabilidad')
        plt.title('ECDF linear - log')
    elif y=='log':
        grid.set(yscale="log")
        plt.xlabel('Grado')
        plt.ylabel('Probabilidad [log]')
        plt.title('ECDF log - linear')
    else:
        plt.xlabel('Grado')
        plt.ylabel('Probabilidad')
        plt.title('ECDF')
    grid.set(xlim=0)
    plt.show()