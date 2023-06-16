import math
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
def rotateShift(pos,theta,xs,ys):
    x = pos[0]
    y = pos[1]
    x1 = int(x*np.cos(theta) - y*np.sin(theta) + xs)
    y1 = int(y*np.cos(theta) + x*np.sin(theta) + ys)
    return [x1,y1]

def drawSquare(img,c,theta,col):
    xs = (c%3)*200
    ys = (c//3)*200
    x0 = 100
    y0 = 100
    pos1 = rotateShift([30-x0,30-y0],theta,x0+xs,y0+ys)
    pos2 = rotateShift([170-x0,30-y0],theta,x0+xs,y0+ys)
    pos3 = rotateShift([170-x0,170-y0],theta,x0+xs,y0+ys)
    pos4 = rotateShift([30-x0,170-y0],theta,x0+xs,y0+ys)
    pointsx = np.array([pos1,pos2,pos3,pos4])
    img = cv2.fillPoly(img, pts=[pointsx], color=col)
    return img

def drawTriangle(img,c,theta,col):
    xs = (c%3)*200
    ys = (c//3)*200
    x0 = 100
    y0 = 100
    pos1 = rotateShift([100-x0,195-y0],theta,x0+xs,y0+ys)
    pos2 = rotateShift([18-x0,53-y0],theta,x0+xs,y0+ys)
    pos3 = rotateShift([182-x0,53-y0],theta,x0+xs,y0+ys)
    pointsx = np.array([pos1,pos2,pos3])
    img = cv2.fillPoly(img, pts=[pointsx], color=col)
    return img

def drawCircle(img,c,rad,col):
    xs = (c%3)*200
    ys = (c//3)*200
    x0 = 100
    y0 = 100
    img = cv2.circle(img, (x0+xs,y0+ys), rad, color=col, thickness=-1)
    return img

def drawSquare_contour(img,c,theta,col):
    xs = (c%3)*200
    ys = (c//3)*200
    x0 = 100
    y0 = 100
    pos1 = rotateShift([30-x0,30-y0],theta,x0+xs,y0+ys)
    pos2 = rotateShift([170-x0,30-y0],theta,x0+xs,y0+ys)
    pos3 = rotateShift([170-x0,170-y0],theta,x0+xs,y0+ys)
    pos4 = rotateShift([30-x0,170-y0],theta,x0+xs,y0+ys)
    pointsx = np.array([pos1,pos2,pos3,pos4])
    img = cv2.drawContours(img, [pointsx], -1, color=col, thickness=5)
    # img = cv2.fillPoly(img, pts=[pointsx], color=col)
    return img

def drawTriangle_contour(img,c,theta,col):
    xs = (c%3)*200
    ys = (c//3)*200
    x0 = 100
    y0 = 100
    pos1 = rotateShift([100-x0,195-y0],theta,x0+xs,y0+ys)
    pos2 = rotateShift([18-x0,53-y0],theta,x0+xs,y0+ys)
    pos3 = rotateShift([182-x0,53-y0],theta,x0+xs,y0+ys)
    pointsx = np.array([pos1,pos2,pos3])
    img = cv2.drawContours(img, [pointsx], -1, color=col, thickness=5)
    # img = cv2.fillPoly(img, pts=[pointsx], color=col)
    return img

def drawCircle_contour(img,c,rad,col):
    xs = (c%3)*200
    ys = (c//3)*200
    x0 = 100
    y0 = 100
    img = cv2.circle(img, (x0+xs,y0+ys), rad, color=col, thickness=5)
    return img

def drawSquare_large(img,c,theta,col):
    xs = (c%2)*300
    ys = (c//2)*300
    x0 = 150
    y0 = 150
    pos1 = rotateShift([45-x0,45-y0],theta,x0+xs,y0+ys)
    pos2 = rotateShift([255-x0,45-y0],theta,x0+xs,y0+ys)
    pos3 = rotateShift([255-x0,255-y0],theta,x0+xs,y0+ys)
    pos4 = rotateShift([45-x0,255-y0],theta,x0+xs,y0+ys)
    pointsx = np.array([pos1,pos2,pos3,pos4])
    img = cv2.fillPoly(img, pts=[pointsx], color=col)
    return img

def drawTriangle_large(img,c,theta,col):
    xs = (c%2)*300
    ys = (c//2)*300
    x0 = 150
    y0 = 150
    pos1 = rotateShift([150-x0,293-y0],theta,x0+xs,y0+ys)
    pos2 = rotateShift([27-x0,80-y0],theta,x0+xs,y0+ys)
    pos3 = rotateShift([273-x0,80-y0],theta,x0+xs,y0+ys)
    pointsx = np.array([pos1,pos2,pos3])
    img = cv2.fillPoly(img, pts=[pointsx], color=col)
    return img

def drawCircle_large(img,c,rad,col):
    xs = (c%2)*300
    ys = (c//2)*300
    x0 = 150
    y0 = 150
    img = cv2.circle(img, (x0+xs,y0+ys), rad, color=col, thickness=-1)
    return img

def drawSquare_large_contour(img,c,theta,col):
    xs = (c%2)*300
    ys = (c//2)*300
    x0 = 150
    y0 = 150
    pos1 = rotateShift([45-x0,45-y0],theta,x0+xs,y0+ys)
    pos2 = rotateShift([255-x0,45-y0],theta,x0+xs,y0+ys)
    pos3 = rotateShift([255-x0,255-y0],theta,x0+xs,y0+ys)
    pos4 = rotateShift([45-x0,255-y0],theta,x0+xs,y0+ys)
    pointsx = np.array([pos1,pos2,pos3,pos4])
    img = cv2.drawContours(img, [pointsx], -1, color=col, thickness=5)
    # img = cv2.fillPoly(img, pts=[pointsx], color=col)
    return img

def drawTriangle_large_contour(img,c,theta,col):
    xs = (c%2)*300
    ys = (c//2)*300
    x0 = 150
    y0 = 150
    pos1 = rotateShift([150-x0,293-y0],theta,x0+xs,y0+ys)
    pos2 = rotateShift([27-x0,80-y0],theta,x0+xs,y0+ys)
    pos3 = rotateShift([273-x0,80-y0],theta,x0+xs,y0+ys)
    pointsx = np.array([pos1,pos2,pos3])
    img = cv2.drawContours(img, [pointsx], -1, color=col, thickness=5)
    # img = cv2.fillPoly(img, pts=[pointsx], color=col)
    return img

def drawCircle_large_contour(img,c,rad,col):
    xs = (c%2)*300
    ys = (c//2)*300
    x0 = 150
    y0 = 150
    img = cv2.circle(img, (x0+xs,y0+ys), rad, color=col, thickness=5)
    return img

def threeShapeSetA():
    img = np.zeros((600,600), dtype = "uint8")
    mask = np.zeros((600,600), dtype = "uint8")
    col = 255
    seq = list(np.arange(0,9))
    c = random.sample(seq,3)
    
    theta1 = 2*math.pi*random.random()
    col1 = int(120+50*random.random())
    img = drawSquare(img,c[0],theta1,col1)
    mask = drawSquare(mask,c[0],theta1,col)

    theta2 = 2*math.pi*random.random()
    col2 = int(120+50*random.random())
    img = drawTriangle(img,c[1],theta2,col2)
    mask = drawTriangle(mask,c[1],theta2,col)

    rad = int(90 + 10*random.random())
    col3 = int(120+50*random.random())
    img = drawCircle(img,c[2],rad,col3)
    mask = drawCircle(mask,c[2],rad,col)
    
    return img,mask

def threeShapeSetB():
    img = np.zeros((600,600), dtype = "uint8")
    mask = np.zeros((600,600), dtype = "uint8")
    col = 255
    seq = list(np.arange(0,9))
    c = random.sample(seq,3)
    
    theta1 = 2*math.pi*random.random()
    col1 = int(120+50*random.random())
    img = drawSquare(img,c[0],theta1,col1)
    #mask = drawSquare(mask,c[0],theta1,col)

    theta2 = 2*math.pi*random.random()
    col2 = int(120+50*random.random())
    img = drawTriangle(img,c[1],theta2,col2)
    mask = drawTriangle(mask,c[1],theta2,col)

    rad = int(90 + 10*random.random())
    col3 = int(120+50*random.random())
    img = drawCircle(img,c[2],rad,col3)
    mask = drawCircle(mask,c[2],rad,col)
    
    return img,mask

def threeShapeSetA_contour():
    img = np.zeros((600,600), dtype = "uint8")
    mask = np.zeros((600,600), dtype = "uint8")
    col = 255
    seq = list(np.arange(0,9))
    c = random.sample(seq,3)
    
    theta1 = 2*math.pi*random.random()
    col1 = int(120+50*random.random())
    img = drawSquare_contour(img,c[0],theta1,col1)
    mask = drawSquare(mask,c[0],theta1,col)

    theta2 = 2*math.pi*random.random()
    col2 = int(120+50*random.random())
    img = drawTriangle_contour(img,c[1],theta2,col2)
    mask = drawTriangle(mask,c[1],theta2,col)

    rad = int(90 + 10*random.random())
    col3 = int(120+50*random.random())
    img = drawCircle_contour(img,c[2],rad,col3)
    mask = drawCircle(mask,c[2],rad,col)
    
    return img,mask

def threeShapeSetB_contour():
    img = np.zeros((600,600), dtype = "uint8")
    mask = np.zeros((600,600), dtype = "uint8")
    col = 255
    seq = list(np.arange(0,9))
    c = random.sample(seq,3)
    
    theta1 = 2*math.pi*random.random()
    col1 = int(120+50*random.random())
    img = drawSquare_contour(img,c[0],theta1,col1)
    #mask = drawSquare(mask,c[0],theta1,col)

    theta2 = 2*math.pi*random.random()
    col2 = int(120+50*random.random())
    img = drawTriangle_contour(img,c[1],theta2,col2)
    mask = drawTriangle(mask,c[1],theta2,col)

    rad = int(90 + 10*random.random())
    col3 = int(120+50*random.random())
    img = drawCircle_contour(img,c[2],rad,col3)
    mask = drawCircle(mask,c[2],rad,col)
    
    return img,mask

def threeShapeSetA_large():
    img = np.zeros((600,600), dtype = "uint8")
    mask = np.zeros((600,600), dtype = "uint8")
    col = 255
    seq = list(np.arange(0,4))
    c = random.sample(seq,3)
    
    theta1 = 2*math.pi*random.random()
    col1 = int(120+50*random.random())
    img = drawSquare_large(img,c[0],theta1,col1)
    mask = drawSquare_large(mask,c[0],theta1,col)

    theta2 = 2*math.pi*random.random()
    col2 = int(120+50*random.random())
    img = drawTriangle_large(img,c[1],theta2,col2)
    mask = drawTriangle_large(mask,c[1],theta2,col)

    rad = int(120 + 15*random.random())
    col3 = int(120+50*random.random())
    img = drawCircle_large(img,c[2],rad,col3)
    mask = drawCircle_large(mask,c[2],rad,col)
    
    return img,mask

def threeShapeSetB_large():
    img = np.zeros((600,600), dtype = "uint8")
    mask = np.zeros((600,600), dtype = "uint8")
    col = 255
    seq = list(np.arange(0,4))
    c = random.sample(seq,3)
    
    theta1 = 2*math.pi*random.random()
    col1 = int(120+50*random.random())
    img = drawSquare_large(img,c[0],theta1,col1)
    #mask = drawSquare(mask,c[0],theta1,col)

    theta2 = 2*math.pi*random.random()
    col2 = int(120+50*random.random())
    img = drawTriangle_large(img,c[1],theta2,col2)
    mask = drawTriangle_large(mask,c[1],theta2,col)

    rad = int(120 + 15*random.random())
    col3 = int(120+50*random.random())
    img = drawCircle_large(img,c[2],rad,col3)
    mask = drawCircle_large(mask,c[2],rad,col)
    
    return img,mask

def threeShapeSetA_large_contour():
    img = np.zeros((600,600), dtype = "uint8")
    mask = np.zeros((600,600), dtype = "uint8")
    col = 255
    seq = list(np.arange(0,4))
    c = random.sample(seq,3)
    
    theta1 = 2*math.pi*random.random()
    col1 = int(120+50*random.random())
    img = drawSquare_large_contour(img,c[0],theta1,col1)
    mask = drawSquare_large(mask,c[0],theta1,col)

    theta2 = 2*math.pi*random.random()
    col2 = int(120+50*random.random())
    img = drawTriangle_large_contour(img,c[1],theta2,col2)
    mask = drawTriangle_large(mask,c[1],theta2,col)

    rad = int(120 + 15*random.random())
    col3 = int(120+50*random.random())
    img = drawCircle_large_contour(img,c[2],rad,col3)
    mask = drawCircle_large(mask,c[2],rad,col)
    
    return img,mask

def threeShapeSetB_large_contour():
    img = np.zeros((600,600), dtype = "uint8")
    mask = np.zeros((600,600), dtype = "uint8")
    col = 255
    seq = list(np.arange(0,4))
    c = random.sample(seq,3)
    
    theta1 = 2*math.pi*random.random()
    col1 = int(120+50*random.random())
    img = drawSquare_large_contour(img,c[0],theta1,col1)
    #mask = drawSquare(mask,c[0],theta1,col)

    theta2 = 2*math.pi*random.random()
    col2 = int(120+50*random.random())
    img = drawTriangle_large_contour(img,c[1],theta2,col2)
    mask = drawTriangle_large(mask,c[1],theta2,col)

    rad = int(120 + 15*random.random())
    col3 = int(120+50*random.random())
    img = drawCircle_large_contour(img,c[2],rad,col3)
    mask = drawCircle_large(mask,c[2],rad,col)
    
    return img,mask

def drawSquareNew(img,theta,col):
    x0 = 300
    y0 = 300
    xs = 0
    ys = 0
    #theta = 2*math.pi*random.random()
    pos1 = rotateShift([100-x0,100-y0],theta,x0+xs,y0+ys)
    pos2 = rotateShift([100-x0,500-y0],theta,x0+xs,y0+ys)
    pos3 = rotateShift([500-x0,500-y0],theta,x0+xs,y0+ys)
    pos4 = rotateShift([500-x0,100-y0],theta,x0+xs,y0+ys)
    pointsx = np.array([pos1,pos2,pos3,pos4])
    #print(pointsx)
    #print(np.shape(pointsx))
    img = cv2.fillPoly(img, pts=[pointsx], color=col)
    return img

def drawTriangleNew(img,theta,col):
    xs = 0
    ys = 0
    x0 = 300
    y0 = 300
    #theta = 2*math.pi*random.random()
    pos1 = rotateShift([100-x0,100-y0],theta,x0+xs,y0+ys)
    pos2 = rotateShift([100-x0,500-y0],theta,x0+xs,y0+ys)
    pos3 = rotateShift([500-x0,500-y0],theta,x0+xs,y0+ys)
    pointsx = np.array([pos1,pos2,pos3])
    img = cv2.fillPoly(img, pts=[pointsx], color=col)
    return img

def drawSquareContourNew(img,theta,col):
    x0 = 300
    y0 = 300
    xs = 0
    ys = 0
    #theta = 2*math.pi*random.random()
    pos1 = rotateShift([100-x0,100-y0],theta,x0+xs,y0+ys)
    pos2 = rotateShift([100-x0,500-y0],theta,x0+xs,y0+ys)
    pos3 = rotateShift([500-x0,500-y0],theta,x0+xs,y0+ys)
    pos4 = rotateShift([500-x0,100-y0],theta,x0+xs,y0+ys)
    pointsx = np.array([pos1,pos2,pos3,pos4])
    #print(pointsx)
    #print(np.shape(pointsx))
    img = cv2.drawContours(img, [pointsx], -1, color=col, thickness=10)
    return img

def drawTriangleContourNew(img,theta,col):
    xs = 0
    ys = 0
    x0 = 300
    y0 = 300
    #theta = 2*math.pi*random.random()
    pos1 = rotateShift([100-x0,100-y0],theta,x0+xs,y0+ys)
    pos2 = rotateShift([100-x0,500-y0],theta,x0+xs,y0+ys)
    pos3 = rotateShift([500-x0,500-y0],theta,x0+xs,y0+ys)
    pointsx = np.array([pos1,pos2,pos3])
    img = cv2.drawContours(img, [pointsx], -1, color=col, thickness=10)
    return img

def singleShapeSetC():
    img = np.zeros((600,600), dtype = "uint8")
    mask = np.zeros((600,600), dtype = "uint8")
    col = 255
    choice = random.random()
    if(choice<0.5):
        theta1 = 2*math.pi*random.random()
        col1 = int(120+50*random.random())
        img = drawSquareNew(img,theta1,col1)
        mask = drawSquareNew(mask,theta1,col)
    if(choice>=0.5):
        theta1 = 2*math.pi*random.random()
        col1 = int(120+50*random.random())
        img = drawTriangleNew(img,theta1,col1)
        mask = drawTriangleNew(mask,theta1,col)
    return img,mask

def singleShapeSetD():
    img = np.zeros((600,600), dtype = "uint8")
    mask = np.zeros((600,600), dtype = "uint8")
    col = 255
    choice = random.random()
    if(choice<0.5):
        theta1 = 2*math.pi*random.random()
        col1 = int(120+50*random.random())
        img = drawSquareNew(img,theta1,col1)
        mask = drawSquareNew(mask,theta1,col)
    if(choice>=0.5):
        theta1 = 2*math.pi*random.random()
        col1 = int(120+50*random.random())
        img = drawTriangleNew(img,theta1,col1)
    return img,mask


def singleShapeSetE():
    img = np.zeros((600,600), dtype = "uint8")
    mask = np.zeros((600,600), dtype = "uint8")
    col = 255
    choice = random.random()
    if(choice<0.5):
        theta1 = 2*math.pi*random.random()
        col1 = int(120+50*random.random())
        img = drawSquareContourNew(img,theta1,col1)
        mask = drawSquareContourNew(mask,theta1,col)
    if(choice>=0.5):
        theta1 = 2*math.pi*random.random()
        col1 = int(120+50*random.random())
        img = drawTriangleContourNew(img,theta1,col1)
        mask = drawTriangleContourNew(mask,theta1,col)
    return img,mask

def singleShapeSetF():
    img = np.zeros((600,600), dtype = "uint8")
    mask = np.zeros((600,600), dtype = "uint8")
    col = 255
    choice = random.random()
    if(choice<0.5):
        theta1 = 2*math.pi*random.random()
        col1 = int(120+50*random.random())
        img = drawSquareContourNew(img,theta1,col1)
        mask = drawSquareContourNew(mask,theta1,col)
    if(choice>=0.5):
        theta1 = 2*math.pi*random.random()
        col1 = int(120+50*random.random())
        img = drawTriangleContourNew(img,theta1,col1)
    return img,mask

