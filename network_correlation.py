import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import networkx as nx

def plot_correlation_heatmap(corr_matrix: pd.DataFrame):
    """Crea heatmap di correlazione con plotly."""
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.round(2).values,
        texttemplate='%{text}',
        textfont={"size": 10},
        hoverongaps=False
    ))
    fig.update_layout(
        title="Matrice di Correlazione",
        height=600,
        width=800,
        xaxis_title="Ticker",
        yaxis_title="Ticker"
    )
    return fig

def plot_network_graph(corr_matrix: pd.DataFrame, threshold: float = 0.5):
    """Crea grafo a rete basato su correlazioni sopra soglia."""
    G = nx.Graph()
    tickers = corr_matrix.columns
    for t in tickers:
        G.add_node(t)
    for i in range(len(tickers)):
        for j in range(i+1, len(tickers)):
            corr = corr_matrix.iloc[i, j]
            if abs(corr) > threshold:
                G.add_edge(tickers[i], tickers[j], weight=abs(corr))
    if len(G.edges) == 0:
        return None
    pos = nx.spring_layout(G, seed=42)
    # Crea trace per plotly
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=1, color='gray'), hoverinfo='none')
    
    node_x, node_y = [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
    node_trace = go.Scatter(x=node_x, y=node_y, mode='markers+text', text=list(G.nodes()), textposition="top center",
                            marker=dict(size=20, color='lightblue'), hoverinfo='text')
    
    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(title="Grafo di Correlazione (soglia > {})".format(threshold), showlegend=False, hovermode='closest')
    return fig