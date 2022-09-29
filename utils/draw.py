# import packages
# general tools

import os

import matplotlib
import matplotlib.cm as cm
import numpy as np
from cairosvg import svg2png
from matplotlib import pyplot as plt
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from skimage.io import imread



def draw_last_layer(weights):
    fig = plt.figure(figsize=(16, 4), dpi=64)
    ax = fig.gca()
    ax.matshow(weights)
    fig.canvas.draw()

    plt.xlabel('Prototype')
    plt.ylabel('Output')

    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    plt.close()
    return data.reshape(fig.canvas.get_width_height()[::-1] + (3,))


def draw_similarity(weights):
    figure = plt.figure(dpi=128)
    axes = figure.add_subplot(111)
    caxes = axes.matshow(weights, vmin=0, vmax=1)
    figure.colorbar(caxes)

    figure.canvas.draw()

    data = np.frombuffer(figure.canvas.tostring_rgb(), dtype=np.uint8)
    plt.close()

    return data.reshape(figure.canvas.get_width_height()[::-1] + (3,))



def img_for_mol(mol, atom_weights=[]):
    # print(atom_weights)
    highlight_kwargs = {}
    if len(atom_weights) > 0:
        norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)
        cmap = cm.get_cmap('bwr')
        plt_colors = cm.ScalarMappable(norm=norm, cmap=cmap)
        atom_colors = {
            i: plt_colors.to_rgba(atom_weights[i]) for i in range(len(atom_weights))
        }
        highlight_kwargs = {
            'highlightAtoms': list(range(len(atom_weights))),
            'highlightBonds': [],
            'highlightAtomColors': atom_colors
        }

    rdDepictor.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DSVG(512, 512)
    drawer.SetFontSize(1)

    mol = rdMolDraw2D.PrepareMolForDrawing(mol)
    drawer.DrawMolecule(mol, **highlight_kwargs)
                        # highlightAtoms=list(range(len(atom_weights))),
                        # highlightBonds=[],
                        # highlightAtomColors=atom_colors)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    svg = svg.replace('svg:', '')
    svg2png(bytestring=svg, write_to='tmp.png', dpi=128)
    img = imread('tmp.png')
    os.remove('tmp.png')
    return img