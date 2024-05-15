import cv2
import vtk
import config
import random
import py_plantbox as rb

from rb_tools import *
from vtk.util.numpy_support import vtk_to_numpy


def vtkPoints(p):
    """ Creates vtkPoints from an numpy array
    """
    da = vtk.vtkDataArray.CreateDataArray(vtk.VTK_DOUBLE)
    da.SetNumberOfComponents(3)  # vtk point dimension is always 3
    da.SetNumberOfTuples(p.shape[0])
    for i in range(0, p.shape[0]):
        if p.shape[1] == 2:
            da.InsertTuple3(i, p[i, 0], p[i, 1], 0.)
        elif p.shape[1] == 3:
            da.InsertTuple3(i, p[i, 0], p[i, 1], p[i, 2])
    points = vtk.vtkPoints()
    points.SetData(da)
    return points


def vtkCells(t):
    """ Creates vtkCells from an numpy array
    """
    cellArray = vtk.vtkCellArray()
    for vert in t:
        if t.shape[1] == 2:
            tetra = vtk.vtkLine()
        if t.shape[1] == 3:
            tetra = vtk.vtkTriangle()
        elif t.shape[1] == 4:
            tetra = vtk.vtkTetra()
        for i, v in enumerate(vert):
            tetra.GetPointIds().SetId(i, int(v))
        cellArray.InsertNextCell(tetra)
    return cellArray


def vtkData(d):
    """ Creates a vtkDataArray from an numpy array, e.g. grid.GetCellData().SetScalars(vtkData(celldata))
    """
    da = vtk.vtkDataArray.CreateDataArray(vtk.VTK_DOUBLE)
    noc = d.shape[0]
    da.SetNumberOfComponents(noc)
    da.SetNumberOfTuples(d.shape[0])
    for i in range(0, d.shape[0]):
        if noc == 1:
            da.InsertTuple1(i, d[i, 0])
            print('here')
        elif noc == 2:
            da.InsertTuple2(i, d[i, 0], d[i, 1])
        elif noc == 3:
            da.InsertTuple3(i, d[i, 0], d[i, 1], d[i, 2])
        elif noc == 4:
            da.InsertTuple4(i, d[i, 0], d[i, 1], d[i, 2], d[i, 3])
    return da


def simulate_root_system(system_name, type):
    rs = rb.RootSystem()
    rs.openFile(system_name)
    rs.initialize()

    random_day_growth = random.uniform(type[0], type[1])
    rs.simulate(random_day_growth, False)
    return rs


def extract_poly_data(root_system):
    nodes = vv2a(root_system.getNodes())
    segs = seg2a(root_system.getSegments(-1))  # -1 all
    types = v2ai(root_system.getParameter("type", 1, []))  # -1 all

    points = vtkPoints(nodes)
    cells = vtkCells(segs)
    data = vtkData(types)
    print(nodes.shape[0])
    print(segs.shape)
    print(types.shape[0])

    pd = vtk.vtkPolyData()
    pd.SetPoints(points)
    pd.SetLines(cells)
    pd.GetCellData().SetScalars(data)
    return pd


def vtk_polydata_to_numpy(pd):
    render_window = vtk.vtkRenderWindow()
    renderer = vtk.vtkRenderer()
    render_window.AddRenderer(renderer)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(pd)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.RotateX(90.0)

    renderer.AddActor(actor)

    render_window.SetSize(800, 800)
    render_window.SetOffScreenRendering(1)
    render_window.Render()

    window_to_image_filter = vtk.vtkWindowToImageFilter()
    window_to_image_filter.SetInput(render_window)
    window_to_image_filter.SetInputBufferTypeToRGB()
    window_to_image_filter.ReadFrontBufferOff()
    window_to_image_filter.Update()

    vtk_image = window_to_image_filter.GetOutput()
    height, width, _ = vtk_image.GetDimensions()
    vtk_array = vtk_image.GetPointData().GetScalars()
    image_array = vtk_to_numpy(vtk_array).reshape((height, width, 3))

    return image_array


def increase_resolution(img, root_width):
    rescaled_image = cv2.resize(img, config.output_image_dimensions, interpolation=cv2.INTER_LANCZOS4)

    kernel = np.ones((3, 3), np.uint8)

    random_root_width = random.randint(root_width[0], root_width[1])
    img_dilation = cv2.dilate(rescaled_image, kernel, iterations=random_root_width)

    ret, thresh = cv2.threshold(img_dilation, 120, 255, cv2.THRESH_BINARY)
    return thresh


def generate_plant(root_type, **type):
    if len(type.keys()) > 0:
        if 'type' not in type.keys():
            return None
        type = type['type']
    else:
        type = random.choice(config.option_arr)

    rs = simulate_root_system(root_type, type)
    pd = extract_poly_data(rs)
    image_rgb = vtk_polydata_to_numpy(pd)
    large_res_image = increase_resolution(image_rgb, config.root_width)

    return large_res_image
