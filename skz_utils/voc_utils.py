import numpy as np
import os
from xml.etree import ElementTree
import pickle
import random

class XML_preprocessor(object):

    def __init__(self, data_path):
        self.path_prefix = data_path
        self.data = dict()
        self.clsList = ['background']
        self.ignoreList = ['HKD','DCD','BATH','GKSN']
        # self.clsList = self.getClassesFromFile()
        self.num_classes = len(self.clsList)
        self.clsCountList = [0] * self.num_classes
        self.annotationList = []
        self._preprocess_XML()


    def getClassesFromFile(self):
        f = open('classes.txt')
        lines = f.readlines()
        f.close()
        clsList = []
        for line in lines:
            line = line.replace('\n', '')
            line = line.replace('\r', '')
            clsList.append(line)
        return clsList

    def _preprocess_XML(self):
        filenames = os.listdir(self.path_prefix)
        for filename in filenames:
            if not (".xml" in filename):
                print(self.path_prefix + filename)
                continue
            tree = ElementTree.parse(self.path_prefix + filename)
            root = tree.getroot()
            bounding_boxes = []
            one_hot_classes = []
            size_tree = root.find('size')
            width = float(size_tree.find('width').text)
            height = float(size_tree.find('height').text)
            for object_tree in root.findall('object'):
                for bounding_box in object_tree.iter('bndbox'):
                    xmin = float(bounding_box.find('xmin').text)
                    ymin = float(bounding_box.find('ymin').text)
                    xmax = float(bounding_box.find('xmax').text)
                    ymax = float(bounding_box.find('ymax').text)
                bounding_box = [int(xmin),int(ymin),int(xmax),int(ymax)]
                bounding_boxes.append(bounding_box)
                class_name = object_tree.find('name').text
            if class_name in self.ignoreList:
                continue
            """
                one_hot_class = self._to_one_hot(class_name)
                one_hot_classes.append(one_hot_class)
            """
            image_name = root.find('filename').text
            frame = os.path.basename(image_name)
            if not class_name in self.clsList:
                self.clsList.append(class_name)

            class_id = self.clsList.index(class_name)

            annotation = frame + "," + str(int(xmin)) + "," + str(int(xmax)) + "," + str(int(ymin)) + "," + str(int(ymax)) + "," + str(class_id)
            self.annotationList.append(annotation)
            print(annotation)

            """
            bounding_boxes = np.asarray(bounding_boxes)
            one_hot_classes = np.asarray(one_hot_classes)
            image_data = np.hstack((bounding_boxes, one_hot_classes))
            self.data[image_name] = image_data
            """
        '''
        save classe names to file
        '''
        f = open("classes.txt", "w")
        for class_name in self.clsList:
            f.write(class_name + "\n")
        f.close

        '''
        save annotations to file
        '''
        f = open("annotations.txt", "w")
        for annotation in self.annotationList:
            f.write(annotation + "\n")
        f.close

    def _to_one_hot(self,name):
        one_hot_vector = [0] * self.num_classes
        if name in self.clsList:
            i = self.clsList.index(name)
            one_hot_vector[i] = 1
            self.clsCountList[i] = self.clsCountList[i] + 1
        else:
            print('label skipped: %s' %name)

        return one_hot_vector

def generate(annotationsRootDirPath, kplSavePath):
    print(annotationsRootDirPath)
    xmlProc = XML_preprocessor(annotationsRootDirPath)
    f = open(kplSavePath, 'wb')
    pickle.dump(xmlProc.data, f)
    f.close()
    
    return (xmlProc.num_classes + 1), xmlProc.clsList, xmlProc.clsCountList

def writeList2csv(dataList, fileName):
    f = open(fileName, 'w')
    for line in dataList:
        #f.write(line + "\n")
        f.write(line)
    f.close()

def createTrainValFiles(annotationFilePath):
    f = open(annotationFilePath)
    lines = f.readlines()
    f.close()

    classIDList = []
    fileList = {}
    for line in lines:
        lineAry = line.split(',')
        classID = lineAry[-1]
        if not classID in classIDList:
            classIDList.append(classID)
            fileList[classID] = []
        fileList[classID].append(line)

    trainList = []
    valList = []
    ratio = 0.8
    for key in fileList.keys():
        tmpList = fileList[key]
        random.shuffle(tmpList)
        devLength = int(len(tmpList) * ratio)
        trainList.extend(tmpList[0:devLength])
        valList.extend(tmpList[devLength:])

    trainList.insert(0, ("frame,xmin,xmax,ymin,ymax,class_id" + "\n"))
    valList.insert(0, ("frame,xmin,xmax,ymin,ymax,class_id" + "\n"))

    writeList2csv(trainList, "train.csv")
    writeList2csv(valList, "val.csv")


import os_utils
def get_object_str_list(annotation_xml_file_path, img_ext="jpg"):
    object_str_list = []
    # img file name
    img_file_name = os_utils.get_fileName_no_ext(annotation_xml_file_path) + "." + img_ext
    # parse xml
    tree = ElementTree.parse(annotation_xml_file_path)
    root = tree.getroot()
    for object_tree in root.findall('object'):
        for bounding_box in object_tree.iter('bndbox'):
            xmin = float(bounding_box.find('xmin').text)
            ymin = float(bounding_box.find('ymin').text)
            xmax = float(bounding_box.find('xmax').text)
            ymax = float(bounding_box.find('ymax').text)
        bounding_box = [int(xmin),int(ymin),int(xmax),int(ymax)]
        object_name = object_tree.find('name').text
        object_str = ','.join([str(img_file_name), str(bounding_box[0]), str(bounding_box[2]), str(bounding_box[1]), str(bounding_box[3]), str(object_name)])
        object_str_list.append(object_str)
    return object_str_list

def get_object_list_from_dir(xml_dir_path):
    xml_file_path_list = os_utils.get_all_files_pathList(xml_dir_path, "xml")
    obj_list  = []
    for xml_file_path in xml_file_path_list:
        print("[info] {}".format(xml_file_path))
        obj_list.extend(get_object_str_list(xml_file_path))
    obj_list = [obj_str.split(',') for obj_str in obj_list]

    return obj_list

if __name__ == '__main__':
    import jup_def
    import cv2
    import matplotlib.pyplot as plt
    from random import shuffle
    obj_list = get_object_list_from_dir(jup_def.voc_ann_dir_path)
    label_list = list(set([row[5] for row in obj_list]))
    target_label_list = ['background', 'person']
    target_count= np.zeros(len(target_label_list))
    target_obj_list = []
    for row in obj_list:
        label = row[5]
        if label in target_label_list:
            label_no = target_label_list.index(label)
            row[5] = str(label_no)
            target_obj_list.append(row)
            target_count[label_no] += 1
    print("[info] target_obj_list:{}".format(target_obj_list[0]))
    # output result
    result_dir_path = "../../../exaBase/Detection/input/dataset/" + os_utils.get_dateTime_str()
    os_utils.createDir(result_dir_path)
    # classes.txt
    classes_txt_file_path = os.path.join(result_dir_path, "classes.txt")
    with open(classes_txt_file_path, 'w') as f:
        lines = '\n'.join(target_label_list)
        f.writelines(lines)
    # train/val.csv
    header = "frame,xmin,xmax,ymin,ymax,class_id"
    train_csv_file_path = os.path.join(result_dir_path, "train.csv")
    val_csv_file_path = os.path.join(result_dir_path, "val.csv")
    train_writer = open(train_csv_file_path, 'w')
    train_writer.write(header+'\n')
    val_writer = open(val_csv_file_path, 'w')
    val_writer.write(header+'\n')
    # random order
    shuffle(target_obj_list)
    # temp
    dataset_limit = 1000 if len(target_obj_list) >= 1000 else len(target_obj_list)
    target_obj_list = target_obj_list[0:dataset_limit]

    train_vs_val = 0.8
    train_limit = (target_count * train_vs_val).astype(int)
    for obj in target_obj_list:
        obj_id = int(obj[5])
        if train_limit[obj_id] > 0:
            train_writer.write(','.join(obj) + '\n')
            train_limit[obj_id] -= 1
        else:
            val_writer.write(','.join(obj) + '\n')
    train_writer.close()
    val_writer.close()
    print("[info] target: {}".format(target_label_list))
    print("[info] count: {}".format(target_count))
    exit()
    tree = ElementTree.parse(jup_def.voc_sample_xml_file_path)
    root = tree.getroot()
    size_tree = root.find('size')
    width = float(size_tree.find('width').text)
    height = float(size_tree.find('height').text)
    img = cv2.imread(jup_def.voc_sample_img_file_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for object_tree in root.findall('object'):
        for bounding_box in object_tree.iter('bndbox'):
            xmin = float(bounding_box.find('xmin').text)
            ymin = float(bounding_box.find('ymin').text)
            xmax = float(bounding_box.find('xmax').text)
            ymax = float(bounding_box.find('ymax').text)
        bounding_box = [int(xmin),int(ymin),int(xmax),int(ymax)]
        class_name = object_tree.find('name').text
        img = cv2.rectangle(img, (bounding_box[0], bounding_box[1]), (bounding_box[2], bounding_box[3]), (255,0,0), 1)
        cv2.putText(img,class_name,(bounding_box[0],bounding_box[1]), font, 0.5,(255,0,0), 1, cv2.LINE_AA)
        print("[info] {}: {}".format(class_name, bounding_box))
    plt.imshow(img)
    plt.show()
    print("[info] Done")