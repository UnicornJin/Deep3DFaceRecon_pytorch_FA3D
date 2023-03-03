"""
This script loads model for Deep3DFaceRecon_pytorch,
Then stand by, wait for signal of generation
"""

import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import MyVisualizer
from util.preprocess import align_img
from PIL import Image
import numpy as np
from util.load_mats import load_lm3d
import torch 
from data.flist_dataset import default_flist_reader
from scipy.io import loadmat, savemat
import socket
import re
import math
import argparse
from mtcnn import MTCNN
import cv2

def find_between(s, first, last):
    try:
        start = s.index(first) + len(first)
        end = s.index(last, start)
        return s[start:end]
    except ValueError:
        return ""

def get_data_path(root='examples'):
    
    im_path = [os.path.join(root, i) for i in sorted(os.listdir(root)) if i.endswith('png') or i.endswith('jpg')]
    lm_path = [i.replace('png', 'txt').replace('jpg', 'txt') for i in im_path]
    lm_path = [os.path.join(i.replace(i.split(os.path.sep)[-1],''),'detections',i.split(os.path.sep)[-1]) for i in lm_path]

    return im_path, lm_path

def read_data(im_path, lm_path, lm3d_std, to_tensor=True):
    # conver image to RGB 
    im = Image.open(im_path).convert('RGB')
    W,H = im.size
    lm = np.loadtxt(lm_path).astype(np.float32)
    lm = lm.reshape([-1, 2])
    lm[:, -1] = H - 1 - lm[:, -1]
    _, im, lm, _ = align_img(im, lm, lm3d_std)
    if to_tensor:
        im = torch.tensor(np.array(im)/255., dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        lm = torch.tensor(lm).unsqueeze(0)
    return im, lm

def key_point_detection(detector, im_path):
    # img = cv2.cvtColor(cv2.imread("ivan.jpg"), cv2.COLOR_BGR2RGB)

    directory = os.path.abspath(im_path)
    totals = len(os.listdir(directory))
    j = 0
    for file in os.listdir(directory):
        filename = os.path.join(directory, file)
        if filename.endswith(".jpg"):
            img = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
            detect = detector.detect_faces(img)
            confidence = detect[0]['confidence']
            bbox = detect[0]['box']
            if j < 1: print("[Model \ KeyPointDetection] Detection progress: " + str(round((j / totals) * 100, 2)) + "%")
            if float(confidence) > 0.95 or j < 1:
                Leye = find_between(str(detect), '\'left_eye\': (', ')').replace(', ', '\t')
                Reye = find_between(str(detect), '\'right_eye\': (', ')').replace(', ', '\t')
                nose = find_between(str(detect), '\'nose\': (', ')').replace(', ', '\t')
                Lmouth = find_between(str(detect), '\'mouth_left\': (', ')').replace(', ', '\t')
                Rmouth = find_between(str(detect), '\'mouth_right\': (', ')').replace(', ', '\t')
                centerX = (float(bbox[0]) + (float(bbox[2]) / 2.0))  # center X
                centerY = (float(bbox[1]) + (float(bbox[3]) / 2.0))  # center Y
                LeyeB = Leye
                ReyeB = Reye
                noseB = nose
                LmouthB = Lmouth
                RmouthB = Rmouth
                centerXB = centerX
                centerYB = centerY
            else:
                print("[Model \ KeyPointDetection] Low Confidence")
                Leye = LeyeB
                Reye = ReyeB
                nose = noseB
                Lmouth = LmouthB
                Rmouth = RmouthB
                centerX = centerXB
                centerY = centerYB
            print("[Model \ KeyPointDetection] Result:\nLeft Eye: " + Leye + '\n' + "Right Eye: " + Reye + '\n' + "Nose: " + nose + '\n' + "Mouth Left: " + Lmouth + '\n' + "Mouth Right: " + Rmouth)
            if not os.path.exists(os.path.join(directory, 'detections/')):
                os.mkdir(os.path.join(directory, 'detections/'))
            f = open(os.path.join(directory, 'detections/', file.replace('.jpg', '.txt')), "w+")
            f.write(Leye + '\n' + Reye + '\n' + nose + '\n' + Lmouth + '\n' + Rmouth)
            f.close()
            f2 = open(os.path.join(directory, 'detections/', file.replace('.jpg', '.center')), "w+")
            f2.write(str(centerX) + '\n' + str(centerY))
            f2.close()
            j += 1
            print("[Model \ KeyPointDetection] Detection progress: " + str(round((j / totals) * 100, 2)) + "%")

            continue
            # break
        else:
            continue

def main(rank, opt):
    print("[Model] Model launched, start loading...")
    # Load model
    device = torch.device(rank)
    torch.cuda.set_device(device)
    model = create_model(opt)
    model.setup(opt)
    model.device = device
    model.parallelize()
    model.eval()
    visualizer = MyVisualizer(opt)
    lm3d_std = load_lm3d(opt.bfm_folder) 
    face_detector = MTCNN()

    print("[Model] Model loaded")

    # Set Up Socket Listening
    s = socket.socket()
    host = socket.gethostname()
    port = 11451
    s.bind((host, port))
    print("[Model] Model start listining")

    # Start Listen to Message
    s.listen(5)
    while True:
        # Received connection
        connection, address = s.accept()
        print("[Model] Receive Model Generation Request:", address)
        
        imagePath = connection.recv(10240).decode()
        print("[Model] ImageName:", imagePath)

        try:
            print("[Model] Start detecting face key points...")
            key_point_detection(face_detector, imagePath)
            print("[Model] Face key points detection finished...")

            im_path, lm_path = get_data_path(imagePath)

            for i in range(len(im_path)):
                print(i, im_path[i])
                img_name = im_path[i].split(os.path.sep)[-1].replace('.png','').replace('.jpg','')
                if not os.path.isfile(lm_path[i]):
                    print("[Model] Not a path: " + lm_path[i])
                    continue
                im_tensor, lm_tensor = read_data(im_path[i], lm_path[i], lm3d_std)
                data = {
                    'imgs': im_tensor,
                    'lms': lm_tensor
                }
                model.set_input(data)  # unpack data from data loader
                print("[Model] set input data")
                model.test()           # run inference
                print("[Model] done model test")
                visuals = model.get_current_visuals()  # get image results
                visualizer.display_current_results(visuals, 0, opt.epoch, dataset=imagePath.split(os.path.sep)[-1], 
                    save_results=True, count=i, name=img_name, add_image=False)

                save_path = os.path.join(imagePath, 'models')
                print("[Model] Save Path: ", save_path)
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                model.save_mesh(os.path.join(save_path,img_name+'.obj')) # save reconstruction meshes
                model.save_coeff(os.path.join(save_path,img_name+'.mat')) # save predicted coefficients

                connection.sendall('200'.encode())
                print("[Model] Model generation Finished")
    
                connection.close()
        except Exception as err:
            print("[Model] Encountered Error:", err)
            connection.sendall('400'.encode())
            connection.close()


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    main(0, opt)