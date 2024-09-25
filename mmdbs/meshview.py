from vedo import Plotter, Mesh, dataurl,Line, Box
import numpy as np
import crossfiledialog

# Define a function that toggles the transparency of a mesh
#  and changes the button state
import datetime


class MeshViewer:
    def __init__(self, file=None):
        
        self.hidden = False
        self.lines = False

        if not file:  # Some default mesh from online
            self.mesh = Mesh(dataurl + "magnolia.vtk").c("violet").flat()
        else:
            self.mesh = Mesh(file).c("violet").flat()
        self.rgba = np.random.rand(self.mesh.ncells, 4) * 255

        self.z_axis = Line([0,0,0], [0,0,2], lw=3).c("Blue")
        self.y_axis = Line([0,0,0], [0,2,0], lw=3).c("Green")
        self.x_axis = Line([0,0,0], [2,0,0], lw=3).c("Red")

        self.unit_box = Box(width=1,height=1,length=1).c("Black").wireframe(True)
        
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
            self.hidden=True
            if(self.lines):
                self.mesh.wireframe(True)
            else:
                self.mesh.alpha(0)
            
        elif status == "flat shading":
            self.mesh.flat()
            self.hidden=False
            self.mesh.wireframe(False)
            self.mesh.alpha(1)
            self.mesh.c("violet")

        elif status == "smooth shading":
            self.mesh.phong()
            self.mesh.wireframe(False)

        elif status == "random colors":
            self.mesh.wireframe(False)
            self.mesh.cellcolors = self.rgba

        self.view_btn.switch()
    
    def triggerMesh(self, obj, ename):
        if(self.mesh_btn.status()=="show edges"):
            self.mesh.alpha(1)
            self.lines=True
            if(self.hidden):
                self.mesh.wireframe(True)
            self.mesh.linewidth(1)
        else:
            if(self.hidden):
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
    
    def normalize(self,obj,ename):
        status = self.norm_btn.status()
        if(status =="normalize position"):
            print("Normalizing Position")
        elif(status =="normalize pose"):
            print("Normalizing Pose")
        elif(status =="normalize vertices"):
            print("Normalizing Vertices")
        elif(status == "normalize orientation"):
            print("Normalizing Orientation")
        elif(status == "normalize size"):
            print("Normalizing Size")
        self.norm_btn.switch()

    def set_options(self,obj,ename):
        status = self.set_btn.status()
        if(status =="show axis"):
            self.plt.add(self.x_axis,self.y_axis,self.z_axis)
        elif(status =="show unit box"):
            self.plt.add(self.unit_box)
        elif(status =="hide axis"):
            self.plt.remove(self.x_axis,self.y_axis,self.z_axis)
        elif(status == "hide unit box"):
            self.plt.remove(self.unit_box)
        self.set_btn.switch()



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
            c=["w", "w", "w","w"],  # font color for each state
            bc=["dv", "dv", "dv","dv"],  # background color for each state
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
            states=["normalize position","normalize pose","normalize vertices","normalize orientation","normalize size"],
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
            states=["show axis","show unit box","hide axis","hide unit box"],
            c=["w"],
            bc=["dg"],
            font="courier",
            size=20,
            bold=False,
            italic=False,
        )


        self.screenshot_btn = self.plt.add_button(
            self.screenshotPlot,
            pos=(0.15, 0.65),
            states=["screenshot"],
            c=["w"],
            bc=["dg"],
            font="courier",
            size=20,
            bold=False,
            italic=False,
        )


if __name__ == "__main__":
    print("Starting Mesh View")
    file = "..\shapes\AircraftBuoyant\m1343.obj"
    mv = MeshViewer(file=file)
