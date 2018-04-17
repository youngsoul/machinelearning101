import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

# Import patch for drawing rectangles in the legend
from matplotlib.patches import Rectangle

# Create color maps
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bg = ListedColormap(['#333333', '#666666', '#999999'])

# Create a legend for the colors, using rectangles for the corresponding colormap colors
labelList = []


def _init_colors():
    if not labelList:
        for color in cmap_bold.colors:
            labelList.append(Rectangle((0, 0), 1, 1, fc=color))


def plot_clf_boundaries(clf, X_small, y, target_names, x_label='Sepal length (cm)', y_label='Sepal width (cm)', title="Model", cmaps=[cmap_light, cmap_bold], show_plot=True):
    """

    :param clf: classifier to use
    :param X_small: 2 dimensional array of the features.  2 features only for plotting
    :param y: all of the rows
    :param target_names: the set of possible target names
    :param n_neighbors: number of neighbors to consider
    :param weights: uniform or distance
    :param x_label: of the graph
    :param y_label: of the graph
    :return:
    """
    _init_colors()
    h = .02  # step size in the mesh

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    x_min, x_max = X_small[:, 0].min() - 1, X_small[:, 0].max() + 1
    y_min, y_max = X_small[:, 1].min() - 1, X_small[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]) # Make a prediction oat every point
                                                   # in the mesh in order to find the
                                                   # classification areas for each label

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(8, 6))
    plt.pcolormesh(xx, yy, Z, cmap=cmaps[0])

    # Plot the training points
    if y is not None:
        plt.scatter(X_small[:, 0], X_small[:, 1], c=y, cmap=cmaps[1])
    else:
        plt.scatter(X_small[:, 0], X_small[:, 1])
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # Plot the legend
    if target_names is not None:
        plt.legend(labelList, target_names)

    if show_plot:
        plt.show()


def plot_knn_boundaries(X_small, y, target_names, n_neighbors=1, weights='uniform', x_label='Sepal length (cm)', y_label='Sepal width (cm)'):
    """

    :param X_small: 2 dimensional array of the features.  2 features only for plotting
    :param y: all of the rows
    :param target_names: the set of possible target names
    :param n_neighbors: number of neighbors to consider
    :param weights: uniform or distance
    :param x_label: of the graph
    :param y_label: of the graph
    :return:
    """
    clf = KNeighborsClassifier(n_neighbors, weights=weights)
    clf.fit(X_small, y)
    plot_clf_boundaries(clf, X_small, y, target_names, x_label=x_label, y_label=y_label, title=f"3-Class classification (k = {n_neighbors}), weights={weights}")

def plot_logreg_boundaries(X_small, y, target_names, x_label='Sepal length (cm)', y_label='Sepal width (cm)'):
    """

    :param X_small: 2 dimensional array of the features.  2 features only for plotting
    :param y: all of the rows
    :param target_names: the set of possible target names
    :param n_neighbors: number of neighbors to consider
    :param weights: uniform or distance
    :param x_label: of the graph
    :param y_label: of the graph
    :return:
    """
    clf = LogisticRegression()
    clf.fit(X_small, y)
    plot_clf_boundaries(clf, X_small, y, target_names, x_label=x_label, y_label=y_label, title='LogisticRegression')


# Plot what the KFold looks like from the sample data
def print_test_fold(cv, X, y):
    """

    :param cv: cross validation model selection e.g. KFold, StratifiedKFold, etc
    :param X: feature data
    :param y: target values
    :return: None, print the folds of all of the test data
    """
    n_samples = X.shape[0]
    masks = []
    for train_index, test_index in cv.split(X, y):
        print(test_index)


def plot_cv(cv, X, y):
    n_samples = X.shape[0]
    masks = []
    for train_index, test_index in cv.split(X, y):
        mask = np.zeros(n_samples, dtype=bool)
        mask[test_index] = 1
        masks.append(mask)

    plt.matshow(masks)

def plot_iris_scatter(X, y, target_names, x_label='Sepal length (cm)', y_label='Sepal width (cm)', title='Sepal width vs length', cmap = cmap_bold):
    _init_colors()
    # Get the minimum and maximum values with an additional 0.5 border
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

    plt.figure(figsize=(8, 6))

    # Plot the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    # Set the plot limits
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    # Plot the legend
    if target_names:
        plt.legend(labelList, target_names)

    plt.show()

def plot_kmeans_boundaries(clf, X_small, y=None, target_names=None, x_label='Sepal length (cm)', y_label='Sepal width (cm)',title='KMeans (clusters=3)', cmaps=[cmap_bg, None], samples=None):
    _init_colors()
    plot_clf_boundaries(clf, X_small, y, target_names, x_label=x_label, y_label=y_label, title=title, cmaps=cmaps, show_plot=False)
    # Plot the centroids as a black X
    centroids = clf.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=169, linewidths=3,
                color='k', zorder=10)

    # Display the new examples as labeled text on the graph
    if samples:
        plt.text(samples[0][0], samples[0][1], 'A', fontsize=14)
        plt.text(samples[1][0], samples[1][1], 'B', fontsize=14)

    plt.show()