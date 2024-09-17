from vedo import Plotter, Mesh, dataurl
import numpy as np
import crossfiledialog

# Define a function that toggles the transparency of a mesh
#  and changes the button state
import time


class MeshViewer:
    def __init__(self, file=None):
        if not file:  # Some default mesh from online
            self.mesh = Mesh(dataurl + "magnolia.vtk").c("violet").flat()
        else:
            self.mesh = Mesh(file).c("violet").flat()
        self.rgba = np.random.rand(self.mesh.ncells, 4) * 255
        self.show()

    def show(self):
        self.plt = Plotter(axes=11)
        self.orig_camera = self.plt.camera.DeepCopy(self.plt.camera)

        self.buildGui()
        self.plt.show(self.mesh, __doc__)

    def importObject(self, obj, ename):
        file = crossfiledialog.open_file()
        if file is not None:
            try:
                print(f"importing {file}")
                self.plt.remove(self.mesh)
                self.mesh = Mesh(file).c("violet").flat()
                self.rgba = np.random.rand(self.mesh.ncells, 4) * 255
                self.plt.add(self.mesh)
            except:
                print("Unable to add mesh from file", file)

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

    def resetCamera(self, obj, ename):
        self.plt.reset_camera()

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
            font="courier",  # font type
            size=20,  # font size
            bold=False,  # bold font
            italic=False,  # non-italic font style
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
    print("Starting Mesh View")
    mv = MeshViewer()
