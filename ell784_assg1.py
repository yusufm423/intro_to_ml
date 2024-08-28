# -*- coding: utf-8 -*-
"""
ELL784 Introduction to Machine Learning
Assignment 1 - Stauffer and Grimson Background Substraction Model

Submitted by - Mohammad Yusuf (2023EET2757)
             - Satyendra Gehlot (2023EET2191)

"""

import numpy as np
import cv2
import os

#define the model(pdf)
def gaussian_pdf(x, mean, stddev):
    return (1 / (np.sqrt(2 * 3.141) * stddev)) * (np.exp(-0.5 * (((x - mean) / stddev) ** 2)))

def update_parameters(alpha, gray_frame, mean, stddev, weight, gaussian_pdf, v, compare_value, gaussian_match, gaussian_not_match):
    for j in range(3):
        rho = alpha * gaussian_pdf(gray_frame[gaussian_match[j]], mean[j][gaussian_match[j]], stddev[j][gaussian_match[j]])
        C = rho * ((gray_frame[gaussian_match[j]] - mean[j][gaussian_match[j]]) ** 2)

        mean[j][gaussian_match[j]] = (1 - rho) * mean[j][gaussian_match[j]] + rho * gray_frame[gaussian_match[j]]
        covariance[j][gaussian_match[j]] = (1 - rho) * covariance[j][gaussian_match[j]] + C
        weight[j][gaussian_match[j]] = (1 - alpha) * weight[j][gaussian_match[j]] + alpha

        weight[j][gaussian_not_match[j]] = (1 - alpha) * weight[j][gaussian_not_match[j]]
    return mean, covariance, weight

def gauss_match_not_match(compare_value,weight_by_stddev, weight , stddev):
    match = []
    not_match = []
    # calculate values for weight/stddev and find 
    for i in range(3):
        compare_value[i] = np.abs(gray_frame - mean[i])

        weight_by_stddev[i] = weight[i]/stddev[i]
                    
        gaussian_not_fit = np.where(compare_value[i] > 2.5* stddev[i])
        not_match.append(gaussian_not_fit)

        gaussian_fit = np.where(compare_value[i] <= 2.5* stddev[i])
        match.append(gaussian_fit)
    return match, not_match, weight_by_stddev, compare_value
    
def get_bg_indices(gaussian_match, weight, T, height, width):
    # indices where T< of weight of most probable gauss model
    condition1 = np.where(weight[2] >= T)

    # indices where T< of sum of weight of (middle + most) probable gauss model
    condition2 = np.where(((weight[2] + weight[1]) > T) & (weight[2] < T))

    y = gaussian_match[2]
    # updatingindices>threshold
    temp = np.zeros([height, width])
    temp[condition1] = 1
    temp[y] = temp[y] + 1
    index1 = np.where(temp == 2)

    # updatingindices<threshold
    temp = np.zeros([height, width])
    temp[condition2] = 1
    index = np.where((compare_value[2] <= 2.5 * stddev[2]) | (compare_value[1] <= 2.5 * stddev[1]))
    temp[index] = temp[index] + 1
    index2 = np.where(temp == 2)
    
    return index1, index2

def sort_params(mean, covariance, weight, weight_by_stddev):
    # getting index order for sorting weight, mean, covariance by weight/stddev
    index = np.argsort(weight_by_stddev, axis=0)

    mean = np.take_along_axis(mean, index, axis=0)
    covariance = np.take_along_axis(covariance, index, axis=0)
    weight = np.take_along_axis(weight, index, axis=0)

    return mean, covariance, weight


cap = cv2.VideoCapture("umcp.mpg")

# Parameters
alpha = 0.01
T = 0.85

_, frame = cap.read()
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#Get height and width
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object for background video
background_out = cv2.VideoWriter('bg_vid.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (width, height), isColor=False)

# Define the codec and create VideoWriter object for foreground video
foreground_out = cv2.VideoWriter('fg_vid.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (width, height), isColor=False)

#initialise sample mean matrix
mean = np.zeros([3, height, width], np.float64)
mean[1,:,:] = frame

#initialise sample covariance matrix
covariance = np.zeros([3, height, width], np.float64)
covariance[:, :, :] = 5

#initialise sample weight matrix
weight = np.zeros([3, height, width], np.float64)
weight[0, :, :], weight[1, :, :], weight[2, :, :] =0, 0, 1

#initailise weight/stddev matrix to sort K gaussians
weight_by_stddev = np.zeros([3, height, width], np.float64)

#initialise background matrix
background = np.zeros([height, width], np.uint8)

# initialise stddev matrix: standard_deviation
stddev = np.zeros([3,height,width])
# initialise v matrix: 2.5(standard_deviation)
v = np.zeros([3, height, width])
# initialise compare_value matrix: Difference of pixel intensity of current frame and mean value
compare_value = np.zeros([3, height, width])


while True:
    ret, frame = cap.read()

    if ret:

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        gaussian_match = []
        gaussian_not_match = []

        gray_frame = gray_frame.astype(np.float64)

        # stddev = sqrt(covariance) and v = 2.5(stddev)
        for i in range(3):
            covariance[i][np.where(covariance[i]<1)] = 5
            stddev[i] = np.sqrt(covariance[i])
            v[i]= 2.5 * stddev[i]

        # to find the indices where (Xt - ut) <= 2.5* stddev
        gaussian_match, gaussian_not_match, weight_by_stddev, compare_value= gauss_match_not_match(compare_value, weight_by_stddev, weight , stddev)

        # update_parameters for the matched Gaussians
        mean, covariance, weight = update_parameters(alpha, gray_frame, mean, stddev, weight, gaussian_pdf, v, compare_value, gaussian_match, gaussian_not_match)

        not_match_index = np.zeros([3, height, width])
        match_index = np.zeros([height,width])

        match_index[gaussian_match[0]] = 1
        match_index[gaussian_match[1]] = 1
        match_index[gaussian_match[2]] = 1

        not_match_index = np.where(match_index == 0)

        # update least probable gaussian
        mean[0][not_match_index] = gray_frame[not_match_index]
        covariance[0][not_match_index] = 100
        weight[0][not_match_index] = 0.1

        #Normalizing weights
        summation = np.sum(weight, axis =0)
        weight = weight / summation

        #finding omega by sigma for ordering of the gaussian
        for i in range(3):
            weight_by_stddev[i] = weight[i] / stddev[i]

        # sorting based on stddev
        mean, covariance, weight=sort_params(mean, covariance, weight, weight_by_stddev)

        # convert back to integer type to display
        gray_frame = gray_frame.astype(np.uint8)

        #background model estimation

        index1, index2 = get_bg_indices(gaussian_match, weight, T, height, width)
        # index2 = np.where(weight_by_stddev[1]>T) 
        # index3 = np.where(weight_by_stddev[2]>T) 

        # background in index3 and index2
        background[index1] = gray_frame[index1]
        background[index2] = gray_frame[index2]

        # Save background frame
        background_out.write(background)
        
        # Save foreground frame
        foreground_frame = np.abs(gray_frame - background)
        foreground_out.write(foreground_frame)

        # background and foreground video
        cv2.imshow('Background', background)
        cv2.imshow('Foreground',foreground_frame)
        cv2.imshow('Original',gray_frame)

        if cv2.waitKey(30) == 27:
            break
    else:
        break

cap.release()
background_out.release()
foreground_out.release()
cv2.destroyAllWindows()
