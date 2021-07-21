import numpy
import vtk
from matplotlib import pyplot as plt, cm
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

plt.close('all')

def get_vect(fib):

    vers=numpy.zeros((fib.shape[0],3))
    for p in xrange(fib.shape[0]):
        if p == fib.shape[0]-1:
            vers[p,:]=vers[p-1,:];
            break
        x0 = fib[p, 0]
        x1 = fib[p+1,0]
        y0 = fib[p, 1]
        y1 = fib[p + 1, 1]
        z0 = fib[p, 2]
        z1 = fib[p + 1, 2]

        x_val=x1-x0
        y_val=y1-y0
        z_val=z1-z0
        vers[p,:]=[x_val,y_val,z_val];

    return vers

def norm_vect(vector):
    vector_min = vector.astype(float) - numpy.min(vector.astype(float))
    vector_norm = vector_min.astype(float) / numpy.abs(vector_min.astype(float)).max()
    return vector_norm

def norm_vectx(vector,minv,maxv):
    vector_min = vector.astype(float) - minv
    vector_norm = vector_min.astype(float) /maxv
    return vector_norm


def get_track_vects(input_track, *args):

    try:
        downs = args[0]
    except IndexError:
        downs = 1

    print " Loading VTK  file..."

    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(input_track)
    reader.Update()

    polydata = reader.GetOutput()



    xx_max=0
    yy_max=0
    zz_max=0
    xx_min=0
    yy_min=0
    zz_min=0

    nS=polydata.GetNumberOfCells()/1
    cont=0
    sub=30
    from scipy.ndimage.interpolation import zoom


    print " Convert VTK to vectors..."


    xx=numpy.zeros(1)
    yy=numpy.zeros(1)
    zz=numpy.zeros(1)

    xx_l=np.array([])
    yy_l=np.array([])
    zz_l=np.array([])
    xc_l=np.array([])
    yc_l=np.array([])
    zc_l=np.array([])

    for s in range(nS):
       pts = polydata.GetCell(s).GetPoints()
       np_pts = numpy.array([pts.GetPoint(i) for i in range(pts.GetNumberOfPoints())])

       np_shape=np_pts.shape
       #print np_pts

       xx = np_pts[:, 0]
       yy = np_pts[:, 1]
       zz = np_pts[:, 2]



       xx = zoom(xx, downs)
       yy = zoom(yy, downs)
       zz = zoom(zz, downs)


       vx = get_vect(np_pts)[:, 0]
       vy = get_vect(np_pts)[:, 1]
       vz = get_vect(np_pts)[:, 2]

       vx = zoom(vx, downs)
       vy = zoom(vy, downs)
       vz = zoom(vz, downs)

       xx_l=numpy.concatenate((xx_l,xx))
       yy_l=numpy.concatenate((yy_l,yy))
       zz_l=numpy.concatenate((zz_l, zz))

       xc_l = numpy.concatenate((xc_l, vx))
       yc_l = numpy.concatenate((yc_l, vy))
       zc_l = numpy.concatenate((zc_l, vz))

    ##########################


    xc_l=numpy.abs(xc_l)
    xc_l=norm_vect(xc_l)
    yc_l=numpy.abs(yc_l)
    yc_l=norm_vect(yc_l)
    zc_l=numpy.abs(zc_l)
    zc_l=norm_vect(zc_l)
    rgb_l=[]
    for i in xrange(len(xc_l)):
        rgb_l.append((xc_l[i], yc_l[i], zc_l[i]))

    return xx_l, yy_l, zz_l, rgb_l


def plot_track(track_path,*args):

    try:
        downs = args[0]
    except IndexError:
        downs = 1

    xx_l, yy_l, zz_l, rgb_l = get_track_vects(track_path, downs)

    print "plot track..."

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d',axis_bgcolor='black')

    ax.scatter(xx_l, yy_l, zz_l, c=rgb_l, depthshade=False, lw=0.01, s=5, alpha=1, antialiased=True)
    ax.axis("off")
