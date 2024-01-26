import argparse
import torch
import torch.nn.functional as F
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
import tqdm
import ast
import os
import sys
import csv
import torchvision.transforms
from PIL import Image
from IPython.core.display import display, HTML
import sklearn.metrics
import scipy.cluster.hierarchy as hierarchy
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import warnings
warnings.filterwarnings("ignore")

def compute_hyperplans_and_grid(feats, ARGS, crops=False):
    """
        Given features, compute the hyperplans between each class and returns the a grid of predictions.
        - In:
            * feats : Feature tensor with lower dimension (Here dim=2).
            * ARGS : args parser object.
            * crops : Handle multiple crops per image. For now set it to False
        - Out:
            * plot_grid : A dictionnary with necessary
    """
    assert feats.shape[-1]==2, 'Error, features should be of dimension 2'
    feats = feats[:, :ARGS.nb_shots]
    if crops: feats = torch.mean(feats, dim=2)

    grid_size=100
    x_min, x_max = feats.reshape(-1, ARGS.nb_ways-1).min(dim=0)[0][0].item(), feats.reshape(-1, ARGS.nb_ways-1).max(dim=0)[0][0].item()
    y_min, y_max = feats.reshape(-1, ARGS.nb_ways-1).min(dim=0)[0][1].item(), feats.reshape(-1, ARGS.nb_ways-1).max(dim=0)[0][1].item()

    xmingrid = x_min-abs(x_min)*0.4
    ymingrid = y_min-abs(y_min)*0.4
    xmaxgrid = x_max+abs(x_max)*0.4
    ymaxgrid = y_max+abs(y_max)*0.4

    means = torch.mean(feats, dim=1)
    x = torch.linspace(xmingrid, xmaxgrid, grid_size)  # extent of the grid on the x axis
    y = torch.linspace(ymingrid, ymaxgrid, grid_size)  # extent of the grid on the y axis
    [xx, yy] = torch.meshgrid(x, y)
    grid_points = torch.stack([xx.ravel(), yy.ravel()], axis=1).to(feats.device)
    distances = torch.norm((grid_points.reshape(-1,1,2)-means.reshape(1,-1,2)), dim=2, p=2)
    Z = torch.min(distances, dim = 1)[1]
    Z = Z.reshape(grid_size, grid_size)

    hyperplans = []
    for n in range(-1, ARGS.nb_ways-1):
        x1 = means[n].cpu() # centroid class 2
        x2 = means[n+1].cpu() # centroid class 2
        middle = (x1+x2)/2
        a = (x1[1]-x2[1])/(x1[0]-x2[0])
        b = middle[1] + middle[0]/a
        if abs(a)<10e-3:
            hyperplans.append([np.inf,middle[0], middle])
        elif abs(a)>10e3:
            hyperplans.append([0,middle[1], middle])
        else:
            hyperplans.append([-1/a.item(),b.item(), middle])

    def get_intersect(line1, line2):
        a1, b1, _ = line1
        a2, b2, _ = line2
        x = (b2-b1)/(a1-a2)
        y = a1*(b2-b1)/(a1-a2) + b1
        return [x,y]

    intersection = get_intersect(hyperplans[-1], hyperplans[-2])
    plot_grid = {'xmingrid':xmingrid,
                'xmaxgrid':xmaxgrid,
                'ymingrid':ymingrid,
                'ymaxgrid':ymaxgrid,
                'xx':xx,
                'yy':yy,
                'Z':Z,
                'hyperplans':hyperplans,
                'intersection':intersection}

    return plot_grid

def visualize_few_shot_space(reduced_features, run_classes, run_indices, ARGS, query_start=0, query_end=2, plot_support=True, crops=False, figsize=(25, 15)):
    """
        Visualize the feature space. Allows to plot multiple queries at the same time.
        Plotting all queries simultaneously is not recommended as it would overload the plot.
        - In:
            * feats: Features Tensor of dimension [ARGS.nb_ways, ARGS.nb_shots+ARGS.nb_queries, 2].
            * run_classes: Tensor identifying which classes are used in the few shot problem.
            * run_indices: Tensor identifying which samples are used in the few shot problem.
            * ARGS: args parser object.
            * query_start: starting query number.
            * query_end: ending query number.
            * plot_support: boolean value, if True, plots the support points too.
            * crops: boolean value, if plot using crops or not.
            * figsize: figure size default is (25, 15).
    """
    plt.figure(figsize=figsize)
    colors = ['red', 'deepskyblue', 'orange', 'green']
    markers=['P', '>','^', 'D', 'd', '*', '1', '2', '3', '4', '|', 'D', 'd', '.', '*', '1', 'D', 'd', '.', '*', '1']

    assert query_start<ARGS.nb_queries, f'query_start should be smaller than {ARGS.nb_queries}'
    assert query_start>=0, f'query_start should be larger than 0'
    assert query_end<=ARGS.nb_queries, f'query_end should be smaller than {ARGS.nb_queries}'
    assert query_end>0, f'query_end should be larger than 0'

    # Plot grid
    plot_grid = compute_hyperplans_and_grid(reduced_features, ARGS, crops=crops)

    if len(reduced_features.shape) == 3: reduced_features = reduced_features.unsqueeze(2)

    # Support
    for c in range(reduced_features.shape[0]):
        support = reduced_features[c, :ARGS.nb_shots]
        support_indices = [i.item() for i in run_indices[c][:ARGS.nb_shots]]
        if plot_support:
            for s in range(ARGS.nb_shots):
                support_mean = support[s].mean(dim=0)
                plt.scatter(support[s][:, 0].cpu(), support[s][:, 1].cpu(), color=colors[c], edgecolor='black', s=300, linewidth=2, marker=markers[s], label=f'C={run_classes[c]} | s={support_indices[s]}[support]', zorder=1)
                plt.scatter(support_mean[0].cpu(), support_mean[1].cpu(), color=colors[c], edgecolor='black', s=300, linewidth=2, marker=markers[s], zorder=1)
    # Queries
    for c in range(reduced_features.shape[0]):
        query = reduced_features[c, ARGS.nb_shots:]
        query_indices = [i.item() for i in run_indices[c][ARGS.nb_shots:]]
        for q in range(min(query.shape[0], query_start),min(query.shape[0], query_end)):
            query_mean = query[q].mean(dim=0)
            plt.scatter(query[q][:, 0].cpu(), query[q][: , 1].cpu(), label=f'C={run_classes[c]} | s={query_indices[q]}[query]', color=colors[c], edgecolor=colors[c], s=100, marker=markers[ARGS.nb_shots+q-query_start])
            plt.scatter(query_mean[0].cpu(), query_mean[1].cpu(), color=colors[c], s=300, marker=markers[ARGS.nb_shots+q-query_start], linewidth=4)

    # Plot Grid
    xx, yy, Z = plot_grid['xx'], plot_grid['yy'], plot_grid['Z']
    plt.contourf(xx.cpu(), yy.cpu(), Z.cpu(), alpha=0.1)

    # Plot hyperplans
    hyperplans, intersection = plot_grid['hyperplans'], plot_grid['intersection']
    ymingrid, ymaxgrid, xmingrid, xmaxgrid = plot_grid['ymingrid'], plot_grid['ymaxgrid'], plot_grid['xmingrid'], plot_grid['xmaxgrid']

    for i, h in enumerate(hyperplans[:]):
        a,b, middle = h
        if a!= np.inf and a!=0:
            if intersection[0]>middle[0]: #intersect with left grid
                plt.plot([intersection[0], xmingrid], [intersection[1], a*xmingrid+b], color=colors[i])
            else:
                plt.plot([intersection[0], xmaxgrid], [intersection[1], a*xmaxgrid+b], color=colors[i])
        if a==np.inf:
            if middle[1].item()<intersection[1]:
                plt.plot([b, b], [ymingrid, intersection[1]], color=colors[i])
            else:
                plt.plot([b, b], [intersection[1], ymaxgrid], color=colors[i])

        if a==0: #b is y
            if middle[0].item()<intersection[0]:
                plt.plot([xmingrid, intersection[0]], [b, b], color=colors[i])
            else:
                plt.plot([intersection[0], xmaxgrid], [b,b], color=colors[i])

    plt.legend()
    plt.xlim(xmingrid, xmaxgrid)
    plt.ylim(ymingrid, ymaxgrid)
    plt.show()

def convert_label_to_color(labels, colors):
    """
    Convert label to color
    """
    if len(labels.shape)>1:
        labels = labels.reshape(-1)
    return [colors[l] for l in labels]

def clip(x):
    """
    Normalize images between 0 and 255
    """
    return (255*(x-x.min())/(x.max()-x.min())).int()

def opencsv(filename):
    """
    Open csv file
    """
    file = open(filename)
    csvreader = csv.reader(file)
    header = []
    header = next(csvreader)
    rowstrain = []
    rows = []
    for row in csvreader:
        rows.append(row)
    return rows

def getimg(classe, sample=None, filepath='', directory=''):
    """
    Get an image given its classe and sample
    """
    src = opencsv(filepath)
    if type(classe) == torch.Tensor : classe = classe.item()
    if type(classe)==int:
        if sample==None:
            idx=int(np.random.randint(600))
        else:
            if type(sample) == torch.Tensor : sample = sample.item()
            idx = sample
        print("idx: ", idx)
        print("classe: ", classe)
        filename=src[idx+20*classe][0]
        im = Image.open(os.path.join(directory,filename))
        return im

def return_annotation_boxes(classe, sample, n_augmentation, xybox, args, crops=False, features_path='', filepath='./test.csv', directory='./images'):
    """
    Returns annotation boxes given a classe and a sample
    """
    if crops:
        images = torch.load(os.path.join(features_path, 'run1.pt'), map_location='cpu')[classe, sample]
        list_image_hover = []
        for n in range(n_augmentation):
            img_array = images[n]
            im = OffsetImage(img_array, zoom=1)
            annotBox = AnnotationBbox(im, (0,0), xybox=xybox, xycoords='data',boxcoords="offset points",  pad=0.3,  arrowprops=dict(arrowstyle="->"))
            list_image_hover.append(annotBox)
        return list_image_hover
    else:
        img_array = np.array(getimg(classe, sample, filepath=filepath, directory=directory))
        im = OffsetImage(img_array, zoom=1)
        annotBox = AnnotationBbox(im, (0,0), xybox=xybox, xycoords='data',boxcoords="offset points",  pad=0.3,  arrowprops=dict(arrowstyle="->"))
        return [annotBox]

def hover_few_shot_space(reduced_features, run_classes, run_indices, ARGS, figsize=(25, 15), query_start=0, query_end=2, plot_support=True, crops=False, images_path='./images/'):
    """
        Visualize the feature space. Allows to plot multiple queries at the same time.
        Plotting all queries simultaneously is not recommended as it would overload the plot.
        - In:
            * feats: Features Tensor of dimension [ARGS.nb_ways, ARGS.nb_shots+ARGS.nb_queries, 2].
            * run_classes: Tensor identifying which classes are used in the few shot problem.
            * run_indices: Tensor identifying which samples are used in the few shot problem.
            * ARGS: args parser object.
            * query_start: starting query number.
            * query_end: ending query number.
            * plot_support: boolean value, if True, plots the support points too.
            * crops: boolean value, if plot using crops or not.
            * figsize: figure size default is (25, 15).
            * images_path: Path of images
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    colors = ['red', 'deepskyblue', 'orange', 'green']
    markers=['P', '>','^', 'D', 'd', '.', '*', '1', '2', '3', '4', '|', 'D', 'd', '.', '*', '1', 'D', 'd', '.', '*', '1']
    xybox=(25., 25.)

    assert query_start<ARGS.nb_queries, f'query_start should be smaller than {ARGS.nb_queries}'
    assert query_start>=0, f'query_start should be larger than 0'
    assert query_end<=ARGS.nb_queries, f'query_end should be smaller than {ARGS.nb_queries}'
    assert query_end>0, f'query_end should be larger than 0'

    # Plot Grid
    plot_grid = compute_hyperplans_and_grid(reduced_features, ARGS, crops=crops)
    xx, yy, Z = plot_grid['xx'], plot_grid['yy'], plot_grid['Z']
    plt.contourf(xx.cpu(), yy.cpu(), Z.cpu(), alpha=0.1)

    # Plot hyperplans
    hyperplans, intersection = plot_grid['hyperplans'], plot_grid['intersection']
    ymingrid, ymaxgrid, xmingrid, xmaxgrid = plot_grid['ymingrid'], plot_grid['ymaxgrid'], plot_grid['xmingrid'], plot_grid['xmaxgrid']

    for i, h in enumerate(hyperplans[:]):
        a,b, middle = h
        if a!= np.inf and a!=0:
            if intersection[0]>middle[0]: #intersect with left grid
                plt.plot([intersection[0], xmingrid], [intersection[1], a*xmingrid+b], color=colors[i])
            else:
                plt.plot([intersection[0], xmaxgrid], [intersection[1], a*xmaxgrid+b], color=colors[i])
        if a==np.inf:
            if middle[1].item()<intersection[1]:
                plt.plot([b, b], [ymingrid, intersection[1]], color=colors[i])
            else:
                plt.plot([b, b], [intersection[1], ymaxgrid], color=colors[i])

        if a==0: #b is y
            if middle[0].item()<intersection[0]:
                plt.plot([xmingrid, intersection[0]], [b, b], color=colors[i])
            else:
                plt.plot([intersection[0], xmaxgrid], [b,b], color=colors[i])

    X, Y, colors_list, edgecolor_list, list_image_hover = [], [], [], [], []

    if len(reduced_features.shape) == 3: reduced_features = reduced_features.unsqueeze(2)

    # Support
    for c in range(reduced_features.shape[0]):
        support = reduced_features[c, :ARGS.nb_shots]
        support_indices = [i.item() for i in run_indices[c][:ARGS.nb_shots]]
        if plot_support:
            for s in range(ARGS.nb_shots):
                X.append(support[s, :, 0].cpu())
                Y.append(support[s, :, 1].cpu())
                colors_list = colors_list+ [colors[c]]*len(support[s])
                edgecolor_list = edgecolor_list+ ['black']*len(support[s])
                if crops :
                    c_ind, s_ind = c, s
                else:
                    c_ind, s_ind = run_classes[c], run_indices[c][s]

                list_image_hover = list_image_hover + return_annotation_boxes(c_ind, s_ind, 50, xybox, ARGS, features_path=images_path, crops=crops)

        query = reduced_features[c, ARGS.nb_shots:]
        query_indices = [i.item() for i in run_indices[c][ARGS.nb_shots:]]

        for q in range(min(query.shape[0], query_start),min(query.shape[0], query_end)):
            X.append(query[q, :, 0].cpu())
            Y.append(query[q, :, 1].cpu())
            colors_list = colors_list+ [colors[c]]*len(query[q])
            edgecolor_list = edgecolor_list+ [colors[c]]*len(query[q])
            if crops :
                c_ind, q_ind = c, ARGS.nb_shots+q
            else:
                c_ind, q_ind = run_classes[c], run_indices[c][q+ARGS.nb_shots]

            list_image_hover = list_image_hover + return_annotation_boxes(c_ind, q_ind, 50, xybox, ARGS, features_path=images_path, crops=crops)

    X = torch.cat(X)
    Y = torch.cat(Y)

    line = plt.scatter(X, Y, color=colors_list, edgecolor=edgecolor_list)
    plt.contourf(xx.cpu(), yy.cpu(), Z.cpu(), alpha=0.1, color=colors) #cmap=plt.cm.RdBu

    for annotBox in list_image_hover:
        ax.add_artist(annotBox)
        annotBox.set_visible(False)

    def hover(event):
        # if the mouse is over the scatter points
        if line.contains(event)[0]:
            ind, = line.contains(event)[1]["ind"]
            w,h = fig.get_size_inches()*fig.dpi
            ws = (event.x > w/2.)*-1 + (event.x <= w/2.)
            hs = (event.y > h/2.)*-1 + (event.y <= h/2.)

            list_image_hover[ind].xybox = (xybox[0]*ws, xybox[1]*hs)
            list_image_hover[ind].set_visible(True)
            list_image_hover[ind].xy =(X[ind], Y[ind])
            for other_ind in range(len(list_image_hover)):
                if other_ind!=ind:
                    list_image_hover[other_ind].set_visible(False)
        else:
            #if the mouse is not over a scatter point
            for ind in range(len(list_image_hover)):
                list_image_hover[ind].set_visible(False)
        fig.canvas.draw_idle()
    # add callback for mouse moves
    fig.canvas.mpl_connect('motion_notify_event', hover)
    fig = plt.gcf()
    fig.set_size_inches(figsize[0], figsize[1])

    plt.xlim(xmingrid, xmaxgrid)
    plt.ylim(ymingrid, ymaxgrid)

    plt.show()
