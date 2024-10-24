import wx
from vtk.wx.wxVTKRenderWindowInteractor import wxVTKRenderWindowInteractor
import vedo

class VedoApp(wx.Frame):
    def __init__(self, *args, **kw):
        super(VedoApp, self).__init__(*args, **kw)

        # Set up the wx Frame and sizer
        self.SetTitle("Vedo 3D Viewer with wxPython")
        self.SetSize((1000, 600))

        # Create the wxVTKRenderWindowInteractor (for VTK rendering inside wxPython)
        self.widget = wxVTKRenderWindowInteractor(self, -1)
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.SetSizer(sizer)
        sizer.Add(self.widget, 1, wx.EXPAND)

        # Initialize the interactor
        self.widget.Enable(1)
        self.widget.AddObserver("ExitEvent", lambda o, e, f=self: f.Close())

        # Set up the vedo plotter with the wx widget
        self.plotter = vedo.Plotter(bg='white', wx_widget=self.widget)

        # add lines
        self.z_axis = vedo.Line([0,0,0], [0,0,2], lw=3).c("Blue")
        self.y_axis = vedo.Line([0,0,0], [0,2,0], lw=3).c("Green")
        self.x_axis = vedo.Line([0,0,0], [2,0,0], lw=3).c("Red")

        # Create sidebar checkboxes and buttons
        self.create_sidebar()

        # Add default unit box
        self.unit_box = vedo.Box(pos=(0, 0, 0), width=1, height=1, length=1).c("Black").wireframe(True)
        self.plotter.add(self.unit_box)

        # Finalize and show everything
        self.plotter.show(interactive=False)
        self.Layout()
        self.Centre()
        self.Show()

    def create_sidebar(self):
        """Create a sidebar on the left with checkboxes and a button for loading meshes."""
        # Create a vertical sizer for the sidebar
        sidebar_sizer = wx.BoxSizer(wx.VERTICAL)

        # Grid checkbox
        self.grid_cb = wx.CheckBox(self, label="Show Grid")
        self.grid_cb.SetValue(True)
        self.grid_cb.Bind(wx.EVT_CHECKBOX, self.on_toggle_grid)
        sidebar_sizer.Add(self.grid_cb, flag=wx.EXPAND | wx.ALL, border=5)

        # Axes checkbox
        self.axes_cb = wx.CheckBox(self, label="Show Unit Axis")
        self.axes_cb.SetValue(True)
        self.axes_cb.Bind(wx.EVT_CHECKBOX, self.on_toggle_axes)
        sidebar_sizer.Add(self.axes_cb, flag=wx.EXPAND | wx.ALL, border=5)

        # Unit box checkbox
        self.unit_box_cb = wx.CheckBox(self, label="Show Unit Box")
        self.unit_box_cb.SetValue(True)
        self.unit_box_cb.Bind(wx.EVT_CHECKBOX, self.on_toggle_unit_box)
        sidebar_sizer.Add(self.unit_box_cb, flag=wx.EXPAND | wx.ALL, border=5)

        # Button for loading mesh
        self.load_mesh_btn = wx.Button(self, label="Load Mesh")
        self.load_mesh_btn.Bind(wx.EVT_BUTTON, self.on_load_mesh)
        sidebar_sizer.Add(self.load_mesh_btn, flag=wx.EXPAND | wx.ALL, border=5)

        #Dropdown for selecting view
        self.view_choices = ["flat shading", "smooth shading", "random colors", "hide"]
        self.view_dropdown = wx.Choice(self, choices=self.view_choices)
        self.view_dropdown.SetSelection(0)  # Default to first choice
        self.view_dropdown.Bind(wx.EVT_CHOICE, self.on_change_view)
        sidebar_sizer.Add(self.view_dropdown, flag=wx.EXPAND | wx.ALL, border=5)

        # Add sidebar sizer to the frame
        self.GetSizer().Add(sidebar_sizer, flag=wx.EXPAND)

    def on_toggle_grid(self, event):
        """Show or hide the grid based on the checkbox state."""
        if self.grid_cb.GetValue():
            self.plotter.show_grid()
        else:
            self.plotter.hide_grid()

        self.plotter.render()

    def on_toggle_axes(self, event):
        """Show or hide the unit axes based on the checkbox state."""
        
        if self.axes_cb.GetValue():
            self.plotter.add(self.x_axis,self.y_axis,self.z_axis)
        else:
            self.plotter.remove(self.x_axis,self.y_axis,self.z_axis)
        self.plotter.render()

    def on_toggle_unit_box(self, event):
        """Show or hide the unit box based on the checkbox state."""
        if self.unit_box_cb.GetValue():
            self.plotter.add(self.unit_box)
        else:
            self.plotter.remove(self.unit_box)

        self.plotter.render()

    def on_load_mesh(self, event):
        """Open a file dialog to load a mesh and display it."""
        wildcard = "Mesh files (*.stl;*.ply;*.vtk; *.obj)|*.stl;*.ply;*.vtk;*.obj|" \
                   "All files (*.*)|*.*"

        # Open the file dialog
        dialog = wx.FileDialog(self, "Open Mesh File", wildcard=wildcard, style=wx.FD_OPEN)
        if dialog.ShowModal() == wx.ID_OK:
            file_path = dialog.GetPath()

            # Try to load the mesh
            try:
                self.plotter.remove(self.mesh)  # Optionally clear previous objects
                self.mesh = vedo.Mesh(file_path)
                # Add mesh to the plotter and render it
                self.plotter.add(self.mesh)
                self.plotter.render()

            except Exception as e:
                wx.MessageBox(f"Failed to load mesh: {str(e)}", "Error", wx.ICON_ERROR)

        dialog.Destroy()
    
    def on_change_view(self, event):
        """Change the shading, view, or hide the mesh based on the dropdown selection."""
        status = self.view_dropdown.GetStringSelection()
        if hasattr(self, 'mesh'):  # Make sure the mesh is loaded
            if status == "flat shading":
                self.mesh.flat()
                self.mesh.wireframe(False)
                self.mesh.alpha(0.5)
                self.mesh.c("violet")
                self.hidden = False

            elif status == "smooth shading":
                self.mesh.phong()
                self.mesh.wireframe(False)
                self.hidden = False

            elif status == "random colors":
                self.mesh.wireframe(False)
                self.mesh.cellcolors = self.rgba
                self.hidden = False

            elif status == "hide":
                self.hidden = True
                if self.lines:
                    self.mesh.wireframe(True)
                else:
                    self.mesh.alpha(0)

            self.plotter.render()  # Update the plotter view


# Main function to start the wxPython app
if __name__ == "__main__":
    app = wx.App(False)
    frame = VedoApp(None)
    app.MainLoop()
