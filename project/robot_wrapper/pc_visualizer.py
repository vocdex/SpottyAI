import vtk


class PointCloudVisualizer:
    """
    A class to visualize a point cloud using VTK
    """

    def __init__(self):
        """
        Initialize the point cloud visualizer
        """
        # Create a vtkPoints object to hold the initial point coordinates
        vtk_points = vtk.vtkPoints()

        # Create a vtkPolyData object to hold the points
        self.point_cloud = vtk.vtkPolyData()
        self.point_cloud.SetPoints(vtk_points)

        # Create a vtkSphereSource to represent the points as spheres
        sphere_source = vtk.vtkSphereSource()
        sphere_source.SetRadius(0.01)  # Adjust the sphere radius as needed
        sphere_source.SetPhiResolution(10)
        sphere_source.SetThetaResolution(10)

        # Combine the points and spheres using vtkGlyph3D
        glyph = vtk.vtkGlyph3D()
        glyph.SetInputData(self.point_cloud)
        glyph.SetSourceConnection(sphere_source.GetOutputPort())

        # Mapper
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(glyph.GetOutputPort())

        # Actor
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        # Renderer
        renderer = vtk.vtkRenderer()
        renderer.AddActor(actor)
        renderer.SetBackground(0, 0, 0)  # Set background color to black

        # Render Window
        self.render_window = vtk.vtkRenderWindow()
        self.render_window.AddRenderer(renderer)

        # Render Window Interactor
        self.render_interactor = vtk.vtkRenderWindowInteractor()
        self.render_interactor.SetRenderWindow(self.render_window)

        self.points = None

    # Function to update the point cloud data (replace this with your data source)
    def update_vtk_points(self):
        """
        Update the point cloud data
        """
        if self.points is None:
            return

        vtk_points = vtk.vtkPoints()
        for point in self.points:
            vtk_points.InsertNextPoint(point)
        self.point_cloud.SetPoints(vtk_points)
        self.render_window.Render()

    def show_point_cloud(self):
        """
        Show the point cloud
        """

        # Function to be called by the timer
        def timer_callback(obj, event):
            self.update_vtk_points()

        timer_id = self.render_interactor.CreateRepeatingTimer(200)  # 200 milliseconds = 0.2 second
        self.render_interactor.AddObserver('TimerEvent', timer_callback)

        # Start the interaction
        self.render_window.Render()
        self.render_interactor.Start()

    def update_points(self, points):
        """
        Update the points
        """
        self.points = points
