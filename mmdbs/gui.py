import wx
from vtk.wx.wxVTKRenderWindowInteractor import wxVTKRenderWindowInteractor
import vedo
import numpy as np
from normalize import normalize_shape


class MyPanel(wx.Panel):
    """This is the custom panel class containing a dropdown and two buttons."""
    def __init__(self, parent, scroll_sizer, meshRef, pltRef):
        super(MyPanel, self).__init__(parent)
        
        self.scroll_sizer = scroll_sizer  # Reference to the scrollable panel's sizer
        self.parent = parent  # Reference to the scrolling panel (wxScrolledWindow)

        self.mesh = meshRef
        self.plt = pltRef

        self.rgba = np.random.rand(self.mesh.ncells, 4) * 255
        # Horizontal sizer to contain dropdown and buttons
        self.panel_sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        choices = ["flat shading", "smooth shading", "random colors", "hide", "wireframe", "shaded lines"]

        self.dropdown = wx.Choice(self, choices=choices)
        self.dropdown.SetSelection(3)  # Default to first choice
        self.dropdown.Bind(wx.EVT_CHOICE, self.on_change_view)

        self.panel_sizer.Add(self.dropdown, 1, wx.ALL | wx.EXPAND, 5)

        # "Remove" button
        self.remove_button = wx.Button(self, label="Remove")
        self.remove_button.Bind(wx.EVT_BUTTON, self.on_remove)
        self.panel_sizer.Add(self.remove_button, 0, wx.ALL, 5)

        # "Normalize" button
        self.normalize_button = wx.Button(self, label="Normalize")
        self.panel_sizer.Add(self.normalize_button, 0, wx.ALL, 5)
        # Set the sizer and layout for this panel
        self.SetSizer(self.panel_sizer)

    def on_change_view(self, event):
        """Change the shading, view, or hide the mesh based on the dropdown selection."""
        status = self.dropdown.GetStringSelection()
        if hasattr(self, 'mesh'):  # Make sure the mesh is loaded
            if status == "flat shading":
                self.mesh.flat()
                self.mesh.wireframe(False)
                self.mesh.alpha(0.5)
                self.mesh.c("violet")
                self.hidden = False
                self.mesh.linewidth(0)

            elif status == "smooth shading":
                self.mesh.wireframe(False)
                self.hidden = False
                self.mesh.c("violet")
                self.mesh.phong()
                
                self.mesh.linewidth(0)

            elif status == "random colors":
                self.mesh.wireframe(False)
                self.mesh.cellcolors = self.rgba
                self.hidden = False
                self.mesh.flat()
                self.mesh.linewidth(0)

            elif status == "hide":
                self.mesh.alpha(0)
                self.mesh.linewidth(0)
            
            elif status == "wireframe":
                self.mesh.alpha(1)
                self.mesh.linewidth(1)
                self.mesh.c("Black").wireframe(True)
                

            elif status == "shaded lines":
                self.mesh.wireframe(False)
                self.mesh.alpha(0.5)
                self.mesh.flat()
                self.mesh.c("violet")
                self.mesh.linewidth(1)

            self.plt.render()  # Update the plotter view
            
    def on_remove(self, event):
        """Handles removing this panel from the scrollable sizer."""
        if(self.mesh!=None and self.plt!=None):
            self.plt.remove(self.mesh)
            self.plt.render()  # Update the plotter view

        #self.scroll_sizer.Remove(self)  # Remove panel from sizer
        self.Destroy()  # Destroy the panelinstance 
        self.parent.Layout()  # Refresh the parent layout
        self.parent.FitInside()  # Adjust scrollbars
    
    def on_normalize(self,event):
        self.mesh = normalize_shape(self.mesh)
        self.plt.render()
        

class VedoApp(wx.Frame):
    def __init__(self, *args, **kw):
        super(VedoApp, self).__init__(*args, **kw)
        self.lines=False
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
        self.mesh = vedo.Sphere()
        # Finalize and show everything
        self.plotter.show(interactive=False)
        self.Layout()
        self.Centre()
        self.Show()

    def on_reset_camera(self, event):
        self.plotter.reset_camera()
        self.plotter.render()
    
    def create_sidebar(self):
        """Create a sidebar on the left with checkboxes, buttons, and dynamic panels."""

        # Create a vertical sizer for the sidebar
        self.sidebar_sizer = wx.BoxSizer(wx.VERTICAL)

        # Axes checkbox
        self.axes_cb = wx.CheckBox(self, label="Show Unit Axis")
        self.axes_cb.SetValue(False)
        self.axes_cb.Bind(wx.EVT_CHECKBOX, self.on_toggle_axes)
        self.sidebar_sizer.Add(self.axes_cb, flag=wx.EXPAND | wx.ALL, border=5)

        # Unit box checkbox
        self.unit_box_cb = wx.CheckBox(self, label="Show Unit Box")
        self.unit_box_cb.SetValue(True)
        self.unit_box_cb.Bind(wx.EVT_CHECKBOX, self.on_toggle_unit_box)
        self.sidebar_sizer.Add(self.unit_box_cb, flag=wx.EXPAND | wx.ALL, border=5)

        # Reset Camera button
        self.reset_btn = wx.Button(self, label="Reset Camera")
        self.reset_btn.Bind(wx.EVT_BUTTON, self.on_reset_camera)
        self.sidebar_sizer.Add(self.reset_btn, flag=wx.EXPAND | wx.ALL, border=5)

        # Load Mesh button
        self.load_mesh_btn = wx.Button(self, label="Load Mesh")
        self.load_mesh_btn.Bind(wx.EVT_BUTTON, self.on_load_mesh)
        self.sidebar_sizer.Add(self.load_mesh_btn, flag=wx.EXPAND | wx.ALL, border=5)


        # Button to add new dynamic panels
        self.search_btn = wx.Button(self, label="Search")
        self.search_btn.Bind(wx.EVT_BUTTON, self.on_query)
        self.sidebar_sizer.Add(self.search_btn, flag=wx.EXPAND | wx.ALL, border=5)

        # Create a scrolling panel to hold dynamic panels
        self.scroll_panel = wx.ScrolledWindow(self, style=wx.VSCROLL)
        self.scroll_panel.SetScrollRate(5, 5)
        self.scroll_sizer = wx.BoxSizer(wx.VERTICAL)
        self.scroll_panel.SetSizer(self.scroll_sizer)
        self.sidebar_sizer.Add(self.scroll_panel, proportion=1, flag=wx.EXPAND | wx.ALL, border=5)

        # Add sidebar sizer to the main panel
        self.GetSizer().Add(self.sidebar_sizer, proportion=1, flag=wx.EXPAND | wx.ALL, border=5)

    def on_query(self, event):
        return



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

    def add_panel(self,mesh):
        new_panel = MyPanel(self.scroll_panel, self.scroll_sizer,mesh, self.plotter)
        self.scroll_sizer.Add(new_panel, 0, wx.EXPAND | wx.ALL, 5)

        # Refresh the scroll panel
        self.scroll_panel.Layout()
        self.scroll_panel.FitInside()
        self.scroll_panel.Scroll(0, self.scroll_panel.GetScrollRange(wx.VERTICAL))
        
    

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
                mesh = vedo.Mesh(file_path)
                # Add mesh to the plotter and render it
                self.plotter.add(mesh)
                self.plotter.render()
                self.add_panel(mesh)
                
            except Exception as e:
                wx.MessageBox(f"Failed to load mesh: {str(e)}", "Error", wx.ICON_ERROR)
        
        """Adds a new instance of MyPanel to the scrolling panel."""
        dialog.Destroy()



# Main function to start the wxPython app
if __name__ == "__main__":
    app = wx.App(False)
    frame = VedoApp(None)
    app.MainLoop()
