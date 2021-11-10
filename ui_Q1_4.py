import sys
from PyQt5.QtWidgets import QMainWindow, QApplication

from PyQt5.QtGui import QIntValidator, QDoubleValidator
from PyQt5.QtWidgets import QLabel, QWidget, QLineEdit
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout
from PyQt5.QtWidgets import QPushButton, QGroupBox
from PyQt5.QtCore import Qt, QMetaObject
from numpy.lib.function_base import angle
import utils as u
__appname__ = "2021 Opencvdl Hw1"

class windowUI(object):
    """
    Set up UI
    please don't edit
    """
    def setupUI(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setWindowTitle(__appname__)


        # 1. Image Processing Group
        Image_Processing_Group = QGroupBox("1. Image Processing")
        group_V1_vBoxLayout = QVBoxLayout(Image_Processing_Group)

        self.button1_1 = QPushButton("1.1 Load Image")
        self.button1_1.setStyleSheet("text-align:left;")
        self.button1_2 = QPushButton("1.2 Color Seperation")
        self.button1_2.setStyleSheet("text-align:left;")
        self.button1_3 = QPushButton("1.3 Color Transformations")
        self.button1_3.setStyleSheet("text-align:left;")
        self.button1_4 = QPushButton("1.4 Blending")
        self.button1_4.setStyleSheet("text-align:left;")

        group_V1_vBoxLayout.addWidget(self.button1_1)
        group_V1_vBoxLayout.addWidget(self.button1_2)
        group_V1_vBoxLayout.addWidget(self.button1_3)
        group_V1_vBoxLayout.addWidget(self.button1_4)

        # 2. Image Processing Group
        Image_Smoothing_Group = QGroupBox("2. Image Smoothing")
        group_V2_vBoxLayout = QVBoxLayout(Image_Smoothing_Group)

        self.button2_1 = QPushButton("2.1 Gaussian Blur")
        self.button2_2 = QPushButton("2.2 Bilateral Filter")
        self.button2_3 = QPushButton("2.3 Median Filter")
        self.button2_1.setStyleSheet("text-align:left;")
        self.button2_2.setStyleSheet("text-align:left;")
        self.button2_3.setStyleSheet("text-align:left;")

        group_V2_vBoxLayout.addWidget(self.button2_1)
        group_V2_vBoxLayout.addWidget(self.button2_2)
        group_V2_vBoxLayout.addWidget(self.button2_3)
        
        # 3. Edge Detection Group
        Edge_Detection_Group = QGroupBox("3. Edge Detection")
        group_V3_vBoxLayout = QVBoxLayout(Edge_Detection_Group)

        self.button3_1 = QPushButton("3.1 Gaussian Blur")
        self.button3_2 = QPushButton("3.2 Sobel X")
        self.button3_3 = QPushButton("3.3 Sobel Y")
        self.button3_4 = QPushButton("3.4 Magnitude")
        self.button3_1.setStyleSheet("text-align:left;")
        self.button3_2.setStyleSheet("text-align:left;")
        self.button3_3.setStyleSheet("text-align:left;")
        self.button3_4.setStyleSheet("text-align:left;")

        group_V3_vBoxLayout.addWidget(self.button3_1)
        group_V3_vBoxLayout.addWidget(self.button3_2)
        group_V3_vBoxLayout.addWidget(self.button3_3)
        group_V3_vBoxLayout.addWidget(self.button3_4)

        # 4. Transformation Group
        Transformation_Group = QGroupBox("4. Transformation")
        group_V4_vBoxLayout = QVBoxLayout(Transformation_Group)
        layout_w, self.edit_4_1_w = self.edit_Text("width: ", "256", "pixels", showUnit=True)
        layout_h, self.edit_4_1_h = self.edit_Text("height: ", "256", "pixels", showUnit=True)
        self.button4_1 = QPushButton("4.1 Resize")
        layout_x, self.edit_4_2_x = self.edit_Text("x: ", "0", "pixels", showUnit=True)
        layout_y, self.edit_4_2_y = self.edit_Text("y: ", "28", "pixels", showUnit=True)
        self.button4_2 = QPushButton("4.2 Translation")
        layout_angle, self.edit_4_3_angle = self.edit_Text("Angle: ", "10", "degrees", showUnit=True)
        layout_scale, self.edit_4_3_scale = self.edit_Text("Scale: ", "0.5", "", showUnit = True, validator= "float")
        self.button4_3 = QPushButton("4.3 Rotation, Scaling")
        self.button4_4 = QPushButton("4.4 Shearing")
        self.button4_1.setStyleSheet("text-align:left;")
        self.button4_2.setStyleSheet("text-align:left;")
        self.button4_3.setStyleSheet("text-align:left;")
        self.button4_4.setStyleSheet("text-align:left;")

        group_V4_vBoxLayout.addLayout(layout_w)
        group_V4_vBoxLayout.addLayout(layout_h)
        group_V4_vBoxLayout.addWidget(self.button4_1)

        group_V4_vBoxLayout.addLayout(layout_x)
        group_V4_vBoxLayout.addLayout(layout_y)
        group_V4_vBoxLayout.addWidget(self.button4_2)
        group_V4_vBoxLayout.addLayout(layout_angle)
        group_V4_vBoxLayout.addLayout(layout_scale)
        group_V4_vBoxLayout.addWidget(self.button4_3)
        group_V4_vBoxLayout.addWidget(self.button4_4)
        
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        vLayout = QHBoxLayout()
        vLayout.addWidget(Image_Processing_Group)
        vLayout.addWidget(Image_Smoothing_Group)
        vLayout.addWidget(Edge_Detection_Group)
        vLayout.addWidget(Transformation_Group)
        self.centralwidget.setLayout(vLayout)
        MainWindow.setCentralWidget(self.centralwidget)
        QMetaObject.connectSlotsByName(MainWindow)
    @staticmethod
    def edit_Text(title:str, default:str, unit = "", showUnit= False, validator:str = "int"):
        hLayout = QHBoxLayout()

        title_label = QLabel(title)
        title_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        title_label.setFixedWidth(60)
        unit_label = QLabel(unit)
        unit_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        unit_label.setFixedWidth(30)
        editText = QLineEdit(default)
        editText.setFixedWidth(50)
        editText.setAlignment(Qt.AlignRight)
        if validator.lower() == "float":
            editText.setValidator(QDoubleValidator())
        else:
            editText.setValidator(QIntValidator())

        hLayout.addWidget(title_label, alignment=Qt.AlignLeft)
        hLayout.addWidget(editText)
        if showUnit:
            hLayout.addWidget(unit_label)
        return hLayout, editText

class MainWindow(QMainWindow, windowUI):

    def __init__(self, parent = None):
        super(MainWindow, self).__init__(parent=parent)
        self.setupUI(self)
        self.intial_value()
        self.buildUi()

    def buildUi(self):
        self.button1_1.clicked.connect(self.load_image)
        self.button1_2.clicked.connect(self.color_seperation)
        self.button1_3.clicked.connect(self.color_transformation)
        self.button1_4.clicked.connect(self.blending)

        self.button2_1.clicked.connect(self.using_gaussian)
        self.button2_2.clicked.connect(self.using_bilaterial)
        self.button2_3.clicked.connect(self.using_median)

        self.button3_1.clicked.connect(self.edge_gaussian)
        self.button3_2.clicked.connect(self.sobelx)
        self.button3_3.clicked.connect(self.sobely)
        self.button3_4.clicked.connect(self.magnitude)

        self.button4_1.clicked.connect(self.resize)
        self.button4_2.clicked.connect(self.translation)
        self.button4_3.clicked.connect(self.rotation)
        self.button4_4.clicked.connect(self.shearing)
        
    def intial_value(self):
        self.q1 = u.Q1("./Q1_Image/Sun.jpg")
        self.q2 = u.Q2("./Q2_Image/Lenna_whiteNoise.jpg")
        self.q3 = u.Q3("./Q3_Image/House.jpg")
        self.q4 = u.Q4("./Q4_Image/SQUARE-01.png")

    def load_image(self):
        self.q1.image = self.q1.load_image(show_image=True)
        pass

    def color_seperation(self):
        self.q1.color_seperation(True)
        pass

    def color_transformation(self):
        self.q1.color_traformations()
        pass

    def blending(self):
        self.q1.blending()
        pass

    def using_median(self):
        image = self.q2.load_image("./Q2_Image/Lenna_pepperSalt.jpg")
        self.q2.median_filter(image, kernel_size= 3)
        self.q2.median_filter(image, kernel_size= 5)
        self.q2.call_waitkey()
        pass

    def using_gaussian(self):
        self.q2.gaussian_filter()
        pass

    def using_bilaterial(self):
        self.q2.bilateral_filter(kernel_size=9, gammaColor=90, gammaSpace=90)
        pass

    def edge_gaussian(self):
        self.q3.image = self.q3.gaussian_blur()
        self.q3.waitkey()
        pass

    def sobelx(self):
        self.q3.sobel_X = self.q3.sobelX(True)
        self.q3.waitkey()
        pass

    def sobely(self):
        self.q3.sobel_Y = self.q3.sobelY(True)
        self.q3.waitkey()
        pass

    def magnitude(self):
        self.q3.magnitude()
        pass

    def resize(self):
        w = int(self.edit_4_1_w.text())
        h = int(self.edit_4_1_h.text())
        self.q4.image = self.q4.resize((w, h))
        print(self.q4.image.shape)
        self.q4.waitkey()
        pass

    def translation(self):
        x = int(self.edit_4_2_x.text())
        y = int(self.edit_4_2_y.text())
        self.q4.image = self.q4.translations(x, y)
        print(self.q4.image.shape)
        self.q4.waitkey()
        pass

    def rotation(self):
        scale = float(self.edit_4_3_scale.text())
        angle = int(self.edit_4_3_angle.text())
        self.q4.image = self.q4.rotate(angle, scale)
        print(self.q4.image.shape)
        self.q4.waitkey()
        pass

    def shearing(self):
        self.q4.image = self.q4.shearing()
        self.q4.waitkey()
        pass

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.setGeometry(500, 150, 500, 300)
    window.show()
    sys.exit(app.exec_())

