from PyQt5 import QtWidgets, QtGui, QtCore 
from PyQt5.QtWidgets import QMessageBox
from UI import Ui_MainWindow
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        # in python3, super(Class, self).xxx = super().xxx
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setup_control()
        self.Image = None
        self.ImageQ3ImL = None
        self.ImageQ3ImR = None
        self.ImageQ3ImL2 = None
        self.ImageQ3ImR2 = None
        
        self.objPoints = dict() 
        self.objPoints['Q1'] = None
        self.objPoints['Q2'] = None
        self.imgPoints = dict()
        self.imgPoints['Q1'] = None
        self.imgPoints['Q2'] = None 
        self.intrinsicMatrix = dict() 
        self.intrinsicMatrix['Q1'] = None
        self.intrinsicMatrix['Q2'] = None
        self.distortionMatrix = dict() 
        self.distortionMatrix['Q1'] = None
        self.distortionMatrix['Q2'] = None
        self.extrinsicMatrix = dict() 
        self.extrinsicMatrix['Q1'] = None
        self.extrinsicMatrix['Q2'] = None
        self.rvecs = dict() 
        self.rvecs['Q1'] = None
        self.rvecs['Q2'] = None
        self.tvecs = dict() 
        self.tvecs['Q1'] = None
        self.tvecs['Q2'] = None
        
    def setup_control(self):
        # TODO
        self.ui.pushButton.clicked.connect(self.load_click)
        self.ui.pushButton_2.clicked.connect(self.load_imageL_click)
        self.ui.pushButton_3.clicked.connect(self.load_imageR_click)
        self.ui.pushButton_4.clicked.connect(self.btn1_1_click)
        self.ui.pushButton_5.clicked.connect(self.btn1_2_click)
        self.ui.pushButton_6.clicked.connect(self.btn1_3_click)
        self.ui.pushButton_7.clicked.connect(self.btn1_4_click)
        self.ui.pushButton_8.clicked.connect(self.btn1_5_click)
        self.ui.pushButton_9.clicked.connect(self.btn2_1_click)
        self.ui.pushButton_10.clicked.connect(self.btn2_2_click)
        self.ui.pushButton_11.clicked.connect(self.btn3_1_click)

    def load_click(self):
        path = os.getcwd()
        self.Image = dict()
        self.Image['Q1'] = []
        self.Image['Q2'] = []

        print(path)
        filePathQ1 = '/Dataset_CvDl_Hw1/Q1_Image/'
        filePathQ2 = '/Dataset_CvDl_Hw1/Q2_Image/'
        Q1ImageFiles = os.listdir(path+filePathQ1)
        Q2ImageFiles = os.listdir(path+filePathQ2) 
        print('Q1',os.listdir(path+filePathQ1))
        print('Q2',os.listdir(path+filePathQ2))
        

        # if img in Q1ImageFiles : 

        for i in range(len(Q1ImageFiles)):
            img = str(i+1) +".bmp"
            if os.path.exists('.'+filePathQ1+ img) == True:
                self.Image['Q1'].append(cv2.imread('.'+filePathQ1+ img))
        for i in range(len(Q2ImageFiles)):
            img = str(i+1) +".bmp"
            if os.path.exists('.'+filePathQ2+ img) == True:
                self.Image['Q2'].append(cv2.imread('.'+filePathQ2+ img))

    def load_imageL_click(self):
        path = os.getcwd()
        filePath = '/Dataset_CvDl_Hw1/Q3_Image/'
        self.ImageQ3ImL  = cv2.imread('.'+filePath+'imL.png',0)
        self.ImageQ3ImL2  = cv2.imread('.'+filePath+'imL.png')
        self.ImageQ3ImL2 = cv2.cvtColor(self.ImageQ3ImL2,cv2.COLOR_BGR2RGB)

    def load_imageR_click(self):
        path = os.getcwd()
        filePath = '/Dataset_CvDl_Hw1/Q3_Image/'
        self.ImageQ3ImR= cv2.imread('.'+filePath+'imR.png',0)
        self.ImageQ3ImR2= cv2.imread('.'+filePath+'imR.png')
        self.ImageQ3ImR2 = cv2.cvtColor(self.ImageQ3ImR2,cv2.COLOR_BGR2RGB)
    
    def check_image(self):
        if self.Image == None :
            QMessageBox.about(self, "check", "No image,Please confirm the loading image")
            return  False
        return True

    def btn1_1_click(self):
        if self.check_image() == False:
            return 
        objPoints, imgPoints = self.find_corners(self.Image['Q1'],True)
        print('Finish')
        
    def find_corners(self,Images,show = False):
        #number of  corners x,y 
        x = 11
        y = 8

        # Arrays to store object points and image points from all the images.
        objPoints = [] # 3d point in real world space
        imgPoints = [] # 2d points in image plane.

        objp = np.zeros((11 * 8, 3), np.float32)
        objp[:, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2) # Z =0


        for item in Images :
            # Convert to gray
            image = item.copy()
            grayImage = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            # Find checkerboard corners
            #findChessboardCorners(InputArray image, Size patternSize, OutputArray corners, int flags = CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE )
            ret, corners = cv2.findChessboardCorners(grayImage, (x, y), None)

            if ret == True:
                
                objPoints.append(objp)

                #Find the best corner position
                #cv2.cornerSubPix(image, corners, winSize, zeroZone, criteria)
                corners2 = cv2.cornerSubPix(grayImage,corners, (11,11), (-1,-1), (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001))
                
                if [corners2]: 
                    imgPoints.append(corners2)
                else:
                    imgPoints.append(corners)

                if show: 
                    cv2.drawChessboardCorners(image,(x, y), corners2,ret)

                    # show image
                    image = cv2.resize(image, (960, 960))
                    cv2.imshow('Image', image)

                    # wait 0.5S
                    cv2.waitKey(500)

                    # close window
                    cv2.destroyAllWindows()

        return objPoints, imgPoints

    def calibrate_camera(self,Images,string,show = False):
        if self.objPoints[string] is None or self.imgPoints[string] is None :
            self.objPoints[string], self.imgPoints[string] = self.find_corners(Images,show)

        if self.intrinsicMatrix[string] is None or self.distortionMatrix[string] is None or self.rvecs[string] is None or self.tvecs[string] is None :
            ret, self.intrinsicMatrix[string], self.distortionMatrix[string],self.rvecs[string], self.tvecs[string] = cv2.calibrateCamera(self.objPoints[string], self.imgPoints[string], (Images[0].shape[1],Images[0].shape[0]), None, None)       
        
    def btn1_2_click(self):
        if self.check_image() == False:
            return 
        # Calibrate Camera
        if self.intrinsicMatrix['Q1'] is None :
            self.calibrate_camera(self.Image['Q1'],"Q1")
        print('Intrinsic')
        print(self.intrinsicMatrix['Q1'])

    def btn1_3_click(self):
        if self.check_image() == False:
            return 
        choicePic = self.ui.comboBox.currentText()
        # Calibrate Camera
        if self.intrinsicMatrix['Q1'] is None :
            self.calibrate_camera(self.Image['Q1'],'Q1',False)

        if self.extrinsicMatrix['Q1'] is None:
            self.extrinsicMatrix['Q1'] = []
            for i in range(len(self.rvecs['Q1'])):
                rvec, temp = cv2.Rodrigues(self.rvecs['Q1'][i])
                self.extrinsicMatrix['Q1'].append(np.concatenate((rvec, self.tvecs['Q1'][i]), axis=1))

        print('Extrinsic')
        print(self.extrinsicMatrix['Q1'][int(choicePic)-1])

    def btn1_4_click(self):
        if self.check_image() == False:
            return 
        if self.distortionMatrix['Q1'] is None :
            self.calibrate_camera(self.Image['Q1'],'Q1',False)
        print('Distortion')
        print(self.distortionMatrix['Q1'])

    def btn1_5_click(self):
        if self.check_image() == False:
            return 
        if self.intrinsicMatrix['Q1'] is None or self.distortionMatrix['Q1'] is None :
            self.calibrate_camera(self.Image['Q1'],'Q1',False)

        for item in self.Image['Q1']:
            image = item.copy()
            h,  w = image.shape[:2]
            cameraMtx, roi = cv2.getOptimalNewCameraMatrix(self.intrinsicMatrix['Q1'], self.distortionMatrix['Q1'], (w,h), 1, (w,h))
            dst = cv2.undistort(image, self.intrinsicMatrix['Q1'], self.distortionMatrix['Q1'], None, cameraMtx)
            x, y, w, h = roi
            dst = dst[y:y + h, x:x + w]
            image = cv2.resize(image, (960, 960))
            dst = cv2.resize(dst, (960, 960))
            result = np.hstack((image,dst))
            cv2.imshow('Image', result)
            cv2.waitKey(500)
            cv2.destroyAllWindows() 

    def draw(self, img, corners, imgpts):
        corners = corners.astype(int)
        imgpts = imgpts.astype(int)
    
        for i in range(0,len(imgpts),2):
            img = cv2.line(img,tuple(imgpts[i].ravel()),tuple(imgpts[i+1].ravel()), (255,255,0), 10)
        return img

    def augmented_reality(self,wordList) : 

        self.calibrate_camera(self.Image['Q2'],'Q2',False)
        #number of  corners x,y 
        x = 11
        y = 8

        for image in self.Image['Q2']:
            img = image.copy()
            wordPoint = [62, 59, 56, 29, 26, 23]
            wordPoint = np.array(wordPoint)
            # Convert to gray
            grayImage = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            # Find checkerboard corners
            ret, corners = cv2.findChessboardCorners(grayImage, (x, y), None)
            
            if ret == True:
                corners2 = cv2.cornerSubPix(grayImage,corners, (11,11), (-1,-1), (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001))
                objp = np.zeros((11 * 8, 3), np.float32)
                objp[:, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2)
                ret,rvecs, tvecs = cv2.solvePnP(objp, corners2, self.intrinsicMatrix['Q2'], self.distortionMatrix['Q2'])
                rvecs = np.float32(rvecs)
                tvecs = np.float32(tvecs)
                data = [[7,5,0],[4,5,0],[1,5,0],[7,2,0],[4,2,0],[1,2,0]]

                for item in  range(len(wordList)):
                    line = np.float32(wordList[item] + data[item]).reshape(-1, 3)
                    imgpts, jac = cv2.projectPoints(line, rvecs, tvecs, self.intrinsicMatrix['Q2'], self.distortionMatrix['Q2'])
                    img = self.draw(img, corners2, imgpts)
                img = cv2.resize(img, (960, 960))
                cv2.imshow("Image", img)
                cv2.waitKey(1000)
                cv2.destroyAllWindows()
        print("Finish")

    def check_input_text(self):
        if len(self.ui.textEdit.toPlainText()) <= 0 or len(self.ui.textEdit.toPlainText()) > 6 :
            QMessageBox.about(self, "check the input", "Please input 1-6 characters")
            return

    def btn2_1_click(self):
        if self.check_image() == False:
            return 
        fs = cv2.FileStorage('./Dataset_CvDl_Hw1/Q2_Image/Q2_Lib/alphabet_lib_onboard.txt', cv2.FILE_STORAGE_READ)
        self.check_input_text()
        wordList = []
        for chr in self.ui.textEdit.toPlainText():
            wordList.append(fs.getNode(chr).mat())
        self.augmented_reality(wordList)

    def btn2_2_click(self):
        if self.check_image() == False:
            return        
        fs = cv2.FileStorage('./Dataset_CvDl_Hw1/Q2_Image/Q2_Lib/alphabet_lib_vertical.txt', cv2.FILE_STORAGE_READ)
        self.check_input_text()     
        wordList = []
        for chr in self.ui.textEdit.toPlainText():
            wordList.append(fs.getNode(chr).mat())
        self.augmented_reality(wordList)

    def btn3_1_click(self):
        if self.ImageQ3ImL is None :
            QMessageBox.about(self, "check", "No image,Please confirm the loading image")
            return
        if self.ImageQ3ImR is None :
            QMessageBox.about(self, "check", "No image,Please confirm the loading image")
            return

        stereo = cv2.StereoBM_create(numDisparities=16*15, blockSize=11)

        # Compute disparity at full resolution and downsample
        disparity = stereo.compute(self.ImageQ3ImL,self.ImageQ3ImR)
        disparity = cv2.normalize(disparity, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        print('normalize',disparity.dtype)
        disparity = np.uint8(disparity)

        dis = cv2.resize(disparity, (1000, 600))

        
        def onclick(event):
            circle1 = plt.Circle((event.xdata,event.ydata), 10, color='green')
            focalLength = 4019.284   # pixels
            baseline = 342.789    # mm
            dis = disparity[int(event.ydata)][int(event.xdata)]
            doffs = 279.184
            if dis ==0:
                return
            x = event.xdata  - dis
            y = event.ydata
            circle2 = plt.Circle((x, y), 10, color='green')
            #ax1.add_patch(circle1)
            ax2.add_patch(circle2)
            figL.canvas.draw()
            figR.canvas.draw()
        
            print('Disparity: ' + str(dis) + ' pixels\n' + 'Depth: ' + str(int(focalLength * baseline / (dis + doffs ))) + ' mm')
            
        figL = plt.figure(figsize=(12, 8)) 
        imgL = self.ImageQ3ImL2
        ax1 = figL.add_subplot(111)
        plt.axis('off')
        plt.imshow(imgL)
        print('type L',imgL.dtype)
        figL.suptitle('This is a left figure', fontsize=16)
        
        figR = plt.figure(figsize=(12, 8))
        imgR = self.ImageQ3ImR2
        ax2 = figR.add_subplot(111)
        plt.axis('off')
        plt.imshow(imgR)
        print('type R', imgR.dtype)
        figR.suptitle('This is a right figure', fontsize=16)
        
        figG = plt.figure(figsize=(12, 8))
        ax3 = figG.add_subplot(111)
        plt.axis('off')
        plt.imshow(dis, 'gray')
        print('type dis',dis.dtype)

        cid = figL.canvas.mpl_connect('button_press_event', onclick)
        
        plt.axis('off')
        plt.show()

if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
