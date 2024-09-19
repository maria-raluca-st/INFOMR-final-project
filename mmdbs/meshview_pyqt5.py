from vedo import Plotter, Mesh, dataurl
import numpy as np
from PyQt5.QtWidgets import QApplication, QFileDialog  
import sys  

# Define a function that toggles the transparency of a mesh and changes the button state
import time

file_low = "../shapes/Door/D01062.obj"
file_high = "../shapes/Apartment/D00045.obj"
file_non_uniform = "../shapes/Tool/m1108.obj"

def file_dialog():
    try:
        import crossfiledialog
        return crossfiledialog.open_file()
    except:
        app = QApplication(sys.argv)  
        return QFileDialog.getOpenFileName(None, "Open 3D Model", "", "3D Files (*.obj *.stl *.ply *.vtk)")


class MeshViewer:
    def __init__(self, file=None):
        if not file:  # Some default mesh from online
            self.mesh = Mesh(dataurl + "magnolia.vtk").c("violet").flat()
        else:
            self.mesh = Mesh(file).c("violet")
        self.rgba = np.random.rand(self.mesh.ncells, 4) * 255
        self.show()

    def show(self):
        self.plt = Plotter(axes=11)
        self.orig_camera = self.plt.camera.DeepCopy(self.plt.camera)

        self.buildGui()
        self.plt.show(self.mesh, __doc__)

    def importObject(self, obj, ename):
        file, _ = file_dialog()#QFileDialog.getOpenFileName(None, "Open 3D Model", "", "3D Files (*.obj *.stl *.ply *.vtk)")
        if file:
            try:
                print(f"Importing {file}")
                self.plt.remove(self.mesh)
                self.mesh = Mesh(file).c("violet").flat()
                self.rgba = np.random.rand(self.mesh.ncells, 4) * 255
                self.plt.add(self.mesh)
            except Exception as e:
                print(f"Unable to add mesh from file {file}: {e}")

    def switchView(self, obj, ename):
        status = self.view_btn.status()
        if status == "click to hide":
            self.mesh.alpha(0)
            self.mesh.linewidth(0)
        elif status == "click to show obj":
            self.mesh.alpha(1)
            self.mesh.linewidth(0)
            self.mesh.c("violet")
        elif status == "click to show mesh":
            self.mesh.force_opaque().linewidth(1)
            self.mesh.cellcolors = self.rgba
        self.view_btn.switch()

        self.camera_btn = self.plt.add_button(
            self.resetCamera,
            pos=(0.15, 0.85),
            states=["reset camera"],
            c=["w"],
            bc=["dg"],
            font="courier",
            size=20,
            bold=False,
            italic=False,
        )
    
    def hideGui(self):
        self.plt.remove(self.view_btn)
        self.plt.remove(self.import_btn)
        self.plt.remove(self.camera_btn)
        self.plt.remove(self.screenshot_btn)
        
    def resetCamera(self, obj, ename):
     # Reset to the original camera settings
     self.plt.camera = self.orig_camera  # Use saved original camera
     self.plt.render()

    def buildGui(self):
        # Add a button to the plotter with buttonfunc as the callback function
        self.view_btn = self.plt.add_button(
            self.switchView,
            pos=(0.15, 0.95),  # x,y fraction from bottom left corner
            states=[
                "click to hide",
                "click to show obj",
                "click to show mesh",
            ],  # text for each state
            c=["w", "w", "w"],  # font color for each state
            bc=["dg", "dv", "dr"],  # background color for each state
            font="courier",  
            size=20,  
            bold=False,  
            italic=False,  
        )

        self.import_btn = self.plt.add_button(
            self.importObject,
            pos=(0.15, 0.9),
            states=["import object"],
            c=["w"],
            bc=["dg"],
            font="courier",
            size=20,
            bold=False,
            italic=False,
        )

        self.camera_btn = self.plt.add_button(
            self.resetCamera,
            pos=(0.15, 0.85),
            states=["reset camera"],
            c=["w"],
            bc=["dg"],
            font="courier",
            size=20,
            bold=False,
            italic=False,
        )

if __name__ == "__main__":
    print("Starting Mesh Viewer")
    mv = MeshViewer(file = file_non_uniform)
