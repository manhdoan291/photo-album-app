from django.shortcuts import render, redirect
from .models import Category, Photo
# Create your views here.
from django.shortcuts import render
from django.http import HttpResponse
import pandas as pd
from PIL import Image
from PyPDF2 import PdfFileReader
import pdfplumber
import cv2
import numpy as np
import glob
import os
from PIL import Image
import pytesseract as pt
from math import atan
from tqdm import tqdm
def gallery(request):
    category = request.GET.get('category')
    if category == None:
        photos = Photo.objects.all()
    else:
        photos = Photo.objects.filter(category__name=category)

    categories = Category.objects.all()
    context = {'categories': categories, 'photos': photos}
    return render(request, 'photos/gallery.html', context)


def viewPhoto(request, pk):
    photo = Photo.objects.get(id=pk)
    return render(request, 'photos/photo.html', {'photo': photo})


def addPhoto(request):
    categories = Category.objects.all()

    if request.method == 'POST':
        data = request.POST
        images = request.FILES.getlist('images')

        if data['category'] != 'none':
            category = Category.objects.get(id=data['category'])
        elif data['category_new'] != '':
            category, created = Category.objects.get_or_create(
                name=data['category_new'])
        else:
            category = None

        for image in images:
            photo = Photo.objects.create(
                category=category,
                description=data['description'],
                image=image,
            )

        return redirect('gallery')

    context = {'categories': categories}
    return render(request, 'photos/add.html', context)

class Cropper_class:
    def cropper(image_input):
        # Part 1: Extracting only olored regions from the image
        # Chuyen anh thanh gray
        gray = cv2.cvtColor(image_input, cv2.COLOR_BGR2GRAY)
        # Lam mo anh bang ma tran 5 x 5
        blurred = cv2.GaussianBlur(image_input, (3, 3), 0)
        # Tao mot adaptive thresh co gia tri 178 voi method la thresh mean c
        # Tao kernel lay hinh vuong
        thresh = cv2.adaptiveThreshold(gray, 178, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                       cv2.THRESH_BINARY, 11, 2)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
        # Dan no anh lay theo pixel da so
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        # Find the index of the largest contour
        (cnts, _) = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        areas = [cv2.contourArea(c) for c in cnts]
        max_index = np.argmax(areas)
        for cnt in cnts:
            cnt = cnts[max_index]
            x, y, w, h = cv2.boundingRect(cnt)
        cropped_image = image_input[y:y + h, x:x + w]
        cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
        return cropped_image


class Detector_class(Cropper_class):
    def detector(cropped_image):
        def image_resize(cropped_image, width=None, height=None, inter=cv2.INTER_AREA):
            # initialize the dimensions of the image to be resized and
            # grab the image size
            dim = None
            (h, w) = cropped_image.shape[:2]
            # if both the width and height are None, then return the
            # original image
            if width is None and height is None:
                return cropped_image
                # check to see if the width is None
            if width is None:
                # calculate the ratio of the height and construct the
                # dimensions
                r = height / float(h)
                dim = (int(w * r), height)
                # otherwise, the height is None
            else:
                # calculate the ratio of the width and construct the
                # dimensions
                r = width / float(w)
                dim = (width, int(h * r))

            # resize the image
            resized = cv2.resize(cropped_image, dim, interpolation=inter)
            # return the resized image
            return resized

        image = image_resize(cropped_image, width=640, height=720)
        # Anh chua mat cmtnd
        crop_img1 = image[150:350, 30:190]  # Crop from {x, y, w, h } => {0, 0, 300, 400}
        # Có giá trị đến
        crop_img2 = image[105:150, 210:570]  # Crop from {x, y, w, h } => {0, 0, 300, 400}
        # Ho va ten 220
        crop_img3 = image[140:190, 340:570]  # Crop from {x, y, w, h } => {0, 0, 300, 400}
        # Ngay thang nam sinh
        crop_img4 = image[190:220, 225:550]  # Crop from {x, y, w, h } => {0, 0, 300, 400}
        # Gioi tinh
        crop_img5 = image[225:265, 225:350]  # Crop from {x, y, w, h } => {0, 0, 300, 400}
        # Quoc tich
        crop_img6 = image[225:265, 360:620]  # Crop from {x, y, w, h } => {0, 0, 300, 400}
        # Que quan
        crop_img7 = image[260:315, 225:620]  # Crop from {x, y, w, h } => {0, 0, 300, 400}
        # Noi thuong tru
        crop_img8 = image[315:365, 225:620]
        # Co gia tri den
        crop_img9 = image[360:420, 10:240]  # Crop from {x, y, w, h } => {0, 0, 300, 400}
        detected_images = [crop_img1, crop_img2, crop_img3, crop_img4, crop_img5, crop_img6, crop_img7, crop_img8,
                           crop_img9]
        return detected_images


class Ocr_class(Detector_class):
    def ocr(detected_images):
        text1 = detected_images[0]
        text2 = pt.image_to_string(detected_images[1], lang="vie")
        text3 = pt.image_to_string(detected_images[2], lang="vie")
        text4 = pt.image_to_string(detected_images[3], lang="vie")
        text5 = pt.image_to_string(detected_images[4], lang="vie")
        text6 = pt.image_to_string(detected_images[5], lang="vie")
        text7 = pt.image_to_string(detected_images[6], lang="vie")
        text8 = pt.image_to_string(detected_images[7], lang="vie")
        text9 = pt.image_to_string(detected_images[8], lang="vie")
        text_images = (text1, text2, text3, text4, text5, text6, text7, text8, text9)
        return text_images
# MAIN
image_path = './photos/cccd3.jpg'
image_input = cv2.imread(image_path)
Cropper_class.cropper(image_input)
cropped_image = Cropper_class.cropper(image_input)
Detector_class.detector(cropped_image)
detected_images = Detector_class.detector(cropped_image)
selectarea_image = Detector_class.detector(cropped_image)
Ocr_class.ocr(detected_images)
text_images = Ocr_class.ocr(detected_images)
#test voi extract request
def extract_info(request,pk):
    photo = Photo.objects.get(id=pk)
    avatar = text_images[0]
    text_number = text_images[1]
    text_name = text_images[2]
    text_birthday = text_images[3]
    text_male = text_images[4]
    text_national = text_images[5]
    text_hometown = text_images[6]
    text_home = text_images[7]
    text_time = text_images[8]
    return render(request, 'photos/photo.html', {'text_number': text_number,'text_name':text_name,
    'text_birthday':text_birthday,'text_male':text_male,'text_national':text_national,'text_hometown':text_hometown,
    'text_home':text_home,'text_time':text_time,'avatar':avatar,'photo': photo})