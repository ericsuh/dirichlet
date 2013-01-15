import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def simplex2cart(points):
    '''Converts array of 2-simplex coordinates to an array of Cartesian
    coordinates on a 2D triangle in the first quadrant, i.e.::

        (1,0,0) -> (0, 0)
        (0,1,0) -> (0, 1)
        (0,0,1) -> (0.5, np.sqrt(3.0)/2)

    :param points: Points on a 2-simplex.
    :type points: N x 3 list or ndarray.
    :returns: Cartesian coordinate points.
    :rtype: N x 2 ndarray.'''
    points = np.array(points)
    x = 0.5*(2*points[:,1] + points[:,2])/points.sum(axis=1)
    y = (np.sqrt(3.0)/2) * points[:,2]/points.sum(axis=1)
    return np.vstack([x,y]).T

def cart2simplex(points):
    '''Inverse of :func:`simplex2cart`.'''
    points = np.array(points)
    c = (2/np.sqrt(3.0))*points[:,1]
    b = (2*points[:,0] - c)/2.0
    a = 1.0 - c - b
    return np.vstack([a,b,c]).T

def plot_simplexscatter(points, vertexlabels=None, **kwargs):
    '''Scatter plot of 2-simplex points on a 2D triangle.

    :param points: Points on a 2-simplex.
    :type points: N x 3 list or ndarray.
    :param vertexlabels: Labels for corners of plot in the order
        ``(a, b, c)`` where ``a == (1,0,0)``, ``b == (0,1,0)``,
        ``c == (0,0,1)``.
    :type vertexlabels: 3-tuple of strings.
    :param **kwargs: Arguments to :func:`plt.scatter`.
    :type **kwargs: keyword arguments.
    '''
    if vertexlabels is None:
        vertexlabels = ('1','2','3')

    projected = simplex2cart(points)
    plt.scatter(projected[:,0], projected[:,1], **kwargs)

    draw_axes(vertexlabels)
    return plt.gcf()

def plot_simplexcontourf(f, vertexlabels=None, **kwargs):
    '''Filled contour plot of 2-simplex points on a 2D triangle.

    :param f: Function to evaluate on N x 3 ndarray of coordinates
    :type f: ``ufunc``
    :param vertexlabels: Labels for corners of plot in the order
        ``(a, b, c)`` where ``a == (1,0,0)``, ``b == (0,1,0)``,
        ``c == (0,0,1)``.
    :type vertexlabels: 3-tuple of strings.
    :param **kwargs: Arguments to :func:`plt.tricontourf`.
    :type **kwargs: keyword arguments.
    '''
    if vertexlabels is None:
        vertexlabels = ('1','2','3')
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, np.sqrt(3.0)/2.0, 100)
    points2d = np.transpose([np.tile(x, len(y)), np.repeat(y, len(x))])
    points3d = cart2simplex(points2d)
    valid = (points3d.sum(axis=1) == 1.0) & ((0.0 <= points3d).all(axis=1))
    points2d = points2d[np.where(valid),:][0]
    points3d = points3d[np.where(valid),:][0]
    z = f(points3d)
    plt.tricontourf(points2d[:,0], points2d[:,1], z, **kwargs)
    draw_axes(vertexlabels)
    return plt.gcf()

def draw_axes(vertexlabels):
    l1 = matplotlib.lines.Line2D([0,0.5,1.0,0],
                                 [0, np.sqrt(3)/2, 0, 0],
                                 color='k')
    axes = plt.gca()
    axes.add_line(l1)
    axes.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
    axes.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
    axes.text(-0.05, -0.05, vertexlabels[0])
    axes.text(1.05, -0.05, vertexlabels[1])
    axes.text(0.5, np.sqrt(3) / 2 + 0.05, vertexlabels[2])
    axes.set_xlim(-0.2, 1.2)
    axes.set_ylim(-0.2, 1.2)
    axes.set_aspect('equal')
    return axes
