import os
import numpy as np
from sklearn.model_selection import train_test_split
from data_augmentation import load_images  
import matplotlib.pyplot as plt

def label_encoder(labels):
    unique_labels = np.unique(labels)
    label_to_index = {label: index for index, label in enumerate(unique_labels)}
    encoded_labels = np.array([label_to_index[label] for label in labels])  
    return encoded_labels, label_to_index

def split_data_by_label(imgs, labels, test_size=0.2, random_st=42):
    unique_labels = np.unique(labels)
    imgs_train = []
    imgs_test = []
    labels_train = []
    labels_test = []
    for label in unique_labels:
        imgs_of_label = [img for img, lbl in zip(imgs, labels) if lbl == label]
        labels_of_label = [lbl for lbl in labels if lbl == label]
        
        imgs_train_label, imgs_test_label, labels_train_label, labels_test_label = train_test_split(
            imgs_of_label, labels_of_label, test_size=test_size, random_state=random_st
        )
        
        imgs_train.extend(imgs_train_label)
        imgs_test.extend(imgs_test_label)
        labels_train.extend(labels_train_label)
        labels_test.extend(labels_test_label)
        print(f"Letra '{label}': {len(labels_test_label)} elementos prueba")

    return imgs_train, imgs_test, labels_train, labels_test

def process_imgs(data_dir):
    imgs, labels = load_images(data_dir)  
    encoded_labels, label_to_index = label_encoder(labels)  
    imgs_train, imgs_test, labels_train, labels_test = split_data_by_label(imgs, encoded_labels)
    
    imgs_train = np.array(imgs_train)
    imgs_test = np.array(imgs_test)
    labels_train = np.array(labels_train)
    labels_test = np.array(labels_test)

    return imgs_train, imgs_test, labels_train, labels_test, label_to_index

