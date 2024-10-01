from vedo import Plotter, Mesh, dataurl,Line, Box, Sphere, LinearTransform
import numpy as np
from normalize import normalize_pose,normalize_position,normalize_vertices,normalize_scale,normalize_flip,normalize_shape, get_center_of_mass
# Define a function that toggles the transparency of a mesh
#  and changes the button state
import datetime
#from PyQt5.QtWidgets import QApplication, QFileDialog  
import sys


def file_dialog():
    try:
        import crossfiledialog
        return crossfiledialog.open_file()
    except:
        app = QApplication(sys.argv)  
        return QFileDialog.getOpenFileName(None, "Open 3D Model", "", "3D Files (*.obj *.stl *.ply *.vtk)")


class MeshViewer:
    def __init__(self, file=None):

        self.hidden = False
        self.lines = False

        if not file:  # Some default mesh from online
            self.mesh = Mesh(dataurl + "magnolia.vtk").c("violet").flat()
        else:
            self.mesh = Mesh(file).c("violet").flat()
        self.rgba = np.random.rand(self.mesh.ncells, 4) * 255
        com = get_center_of_mass(self.mesh)
        self.ball = Sphere(pos=com, r=0.02, res=24, quads=False, c='red', alpha=1.0)
        #Help Display 
        self.z_axis = Line([0,0,0], [0,0,2], lw=3).c("Blue")
        self.y_axis = Line([0,0,0], [0,2,0], lw=3).c("Green")
        self.x_axis = Line([0,0,0], [2,0,0], lw=3).c("Red")
        self.unit_box = Box(width=1,height=1,length=1).c("Black").wireframe(True)
        self.origMesh = self.mesh.copy().c("Black").wireframe(True)

        self.show()

    def show(self):

        self.plt = Plotter(axes=11)
        self.orig_camera = self.plt.camera.DeepCopy(self.plt.camera)

        self.buildGui()
        self.plt.show(self.mesh, self.ball,__doc__)

    def importObject(self, obj, ename):
        # file = crossfiledialog.open_file()
        file = file_dialog()
        if file is not None:
            try:
                print(f"importing {file}")
                self.plt.remove(self.mesh)
                self.plt.remove(self.origMesh)
                self.mesh = Mesh(file).c("violet").flat().alpha(0.5)
                self.origMesh = self.mesh.copy().c("black").wireframe(True)
                self.rgba = np.random.rand(self.mesh.ncells, 4) * 255
                self.plt.add(self.mesh,self.origMesh)
                self.norm_btn.status("normalize position")
                self.reset_com_ball()
            except:
                print("Unable to add mesh from file", file)

    def switchView(self, obj, ename):
        status = self.view_btn.status()
        if status == "click to hide":
            self.hidden = True
            if self.lines:
                self.mesh.wireframe(True)
            else:
                self.mesh.alpha(0)

        elif status == "flat shading":
            self.mesh.flat()
            self.hidden = False
            self.mesh.wireframe(False)
            self.mesh.alpha(0.5)
            self.mesh.c("violet")

        elif status == "smooth shading":
            self.mesh.phong()
            self.mesh.wireframe(False)

        elif status == "random colors":
            self.mesh.wireframe(False)
            self.mesh.cellcolors = self.rgba

        self.view_btn.switch()

    def triggerMesh(self, obj, ename):
        if self.mesh_btn.status() == "show edges":
            self.mesh.alpha(1)
            self.lines = True
            if self.hidden:
                self.mesh.wireframe(True)
            self.mesh.linewidth(1)
        else:
            if self.hidden:
                self.mesh.alpha(0)
            self.lines = False
            self.mesh.wireframe(False)
            self.mesh.linewidth(0)
        self.mesh_btn.switch()

    def resetCamera(self, obj, ename):
        self.plt.reset_camera()

    def screenshotPlot(self, obj, ename):
        print("Screenshot")
        self.hideGui()
        self.plt.screenshot(f"image.png")
        self.buildGui()

    def hideGui(self):
        self.plt.remove(self.view_btn)
        self.plt.remove(self.import_btn)
        self.plt.remove(self.camera_btn)
        self.plt.remove(self.mesh_btn)
        self.plt.remove(self.norm_btn)
        self.plt.remove(self.set_btn)
        self.plt.remove(self.screenshot_btn)
        self.plt.remove(self.orig_btn)
    
    def reset_com_ball(self):
        npos = get_center_of_mass(self.mesh)
        LT = LinearTransform()
        LT.translate(-self.ball.transform.position+npos)
        LT.move(self.ball)

    def normalize(self,obj,ename):
        print(self.mesh)
        status = self.norm_btn.status()
        if status == "normalize position":
            normalize_position(self.mesh)
        elif status == "normalize pose":
            normalize_pose(self.mesh)
        elif status == "normalize vertices":
            normalize_vertices(self.mesh)
        elif status == "normalize orientation":
            normalize_flip(self.mesh)
        elif status == "normalize size":
            normalize_scale(self.mesh)
        self.reset_com_ball()
        self.norm_btn.switch()

    def set_options(self, obj, ename):
        status = self.set_btn.status()
        if status == "show axis":
            self.plt.add(self.x_axis, self.y_axis, self.z_axis)
        elif status == "show unit box":
            self.plt.add(self.unit_box)
        elif status == "hide axis":
            self.plt.remove(self.x_axis, self.y_axis, self.z_axis)
        elif status == "hide unit box":
            self.plt.remove(self.unit_box)
        self.set_btn.switch()

    def trigger_orig(self, obj, ename):
        status = self.orig_btn.status()
        if status == "show original":
            self.plt.add(self.origMesh)
        elif status == "hide original":
            self.plt.remove(self.origMesh)
        self.orig_btn.switch()

    def buildGui(self):

        # Add a button to the plotter with buttonfunc as the callback function
        self.view_btn = self.plt.add_button(
            self.switchView,
            pos=(0.15, 0.95),  # x,y fraction from bottom left corner
            states=[
                "click to hide",
                "flat shading",
                "smooth shading",
                "random colors",
            ],  # text for each state
            c=["w", "w", "w", "w"],  # font color for each state
            bc=["dv", "dv", "dv", "dv"],  # background color for each state
            font="courier",  # font type
            size=20,  # font size
            bold=False,  # bold font
            italic=False,  # non-italic font style
        )
        self.mesh_btn = self.plt.add_button(
            self.triggerMesh,
            pos=(0.15, 0.9),  # x,y fraction from bottom left corner
            states=[
                "show edges",
                "hide edges",
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
            pos=(0.15, 0.85),
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
            pos=(0.15, 0.8),
            states=["reset camera"],
            c=["w"],
            bc=["dg"],
            font="courier",
            size=20,
            bold=False,
            italic=False,
        )
        self.norm_btn = self.plt.add_button(
            self.normalize,
            pos=(0.15, 0.75),
            states=[
                "normalize vertices",
                "normalize position",
                "normalize pose",
                "normalize orientation",
                "normalize size",
            ],
            c=["w"],
            bc=["dg"],
            font="courier",
            size=20,
            bold=False,
            italic=False,
        )
        self.set_btn = self.plt.add_button(
            self.set_options,
            pos=(0.15, 0.70),
            states=["show axis", "show unit box", "hide axis", "hide unit box"],
            c=["w"],
            bc=["dg"],
            font="courier",
            size=20,
            bold=False,
            italic=False,
        )
        self.orig_btn = self.plt.add_button(
            self.trigger_orig,
            pos=(0.15, 0.65),
            states=["show original", "hide original"],
            c=["w"],
            bc=["dg"],
            font="courier",
            size=20,
            bold=False,
            italic=False,
        )

        self.screenshot_btn = self.plt.add_button(
            self.screenshotPlot,
            pos=(0.15, 0.60),
            states=["screenshot"],
            c=["w"],
            bc=["dg"],
            font="courier",
            size=20,
            bold=False,
            italic=False,
        )

"""
Example Files 
Off Center Door: D01104
Off Center Lamp: m619
Off Center Guitar\D00534.obj

"""


if __name__ == "__main__":
    print("Starting Mesh View")
    file = "../shapes/FloorLamp/m619.obj"
    mv = MeshViewer(file=file)
