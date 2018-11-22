from darkflow.net.build import TFNet
import os
import json
import shutil
import glob
import xml.etree.ElementTree as ET

MINOVERLAP = 0.5

default_valImagesPath = os.path.join('./valData', 'images')

validationImagesPath = default_valImagesPath
predictionsDir = os.path.join(validationImagesPath, 'out')
groundTruthDir = os.path.join('valData', 'groundTruth')
results_files_path = os.path.join('./valData', 'metricResults')

if os.path.isdir(predictionsDir):
    shutil.rmtree(predictionsDir)
    os.mkdir(predictionsDir)
else:
    os.mkdir(predictionsDir)

def voc_ap(rec, prec):

  rec.insert(0, 0.0) # insert 0.0 at begining of list
  rec.append(1.0) # insert 1.0 at end of list
  mrec = rec[:]
  prec.insert(0, 0.0) # insert 0.0 at begining of list
  prec.append(0.0) # insert 0.0 at end of list
  mpre = prec[:]
  """
   This part makes the precision monotonically decreasing
    (goes from the end to the beginning)
    matlab:  for i=numel(mpre)-1:-1:1
                mpre(i)=max(mpre(i),mpre(i+1));
  """

  for i in range(len(mpre)-2, -1, -1):
    mpre[i] = max(mpre[i], mpre[i+1])
  """
   This part creates a list of indexes where the recall changes
  """
  i_list = []
  for i in range(1, len(mrec)):
    if mrec[i] != mrec[i-1]:
      i_list.append(i)
  """
   The Average Precision (AP) is the area under the curve
    (numerical integration)
  """
  ap = 0.0
  for i in i_list:
    ap += ((mrec[i]-mrec[i-1])*mpre[i])
  return ap, mrec, mpre

'''
{'file_id': 'raccoon-197', 'bbox': [142, 55, 939, 672], 'confidence': 0.86}, {'file_id': 'raccoon-134', 'bbox': [133, 83, 194, 175], 'confidence': 0.85},...

19
'''
def createPredictionsList(className):
    predicted_files_list = glob.glob(predictionsDir + '/*.json')
    allPredictions = list()
    if len(predicted_files_list) == 0:
        print('No predicted files were found in the specified path')
        exit(-1)

    predicted_files_list.sort()
    predictionFilesCount = 0
    predictionsCount = 0
    for fi in predicted_files_list:
        predictionFilesCount += 1
        predictionFileContents = json.load(open(fi))
        fileID = os.path.basename(fi).split('.json')[0]
        for prediction in predictionFileContents:
            if not prediction['label'] == className:
                continue
            leftX = prediction['topleft']['x']
            topY = prediction['topleft']['y']
            rightX = prediction['bottomright']['x']
            bottomY = prediction['bottomright']['y']
            bbox = [leftX, topY, rightX, bottomY]
            conf = prediction['confidence']
            allPredictions.append({'file_id': fileID, 'confidence': conf, 'bbox': bbox})
            predictionsCount +=1
    allPredictions.sort(key=lambda x: x['confidence'], reverse=True)
    return allPredictions,predictionsCount



'''
'raccoon-168': [{'bbox': [81, 96, 365, 314], 'used': False, 'class_name': 'raccoon'}, {'bbox': [184, 1, 481, 295], 'used': False, 'class_name': 'raccoon'}]

{'raccoon': 43}
'''

def createGroundTruthList():

    ground_truth_files_list = glob.glob(groundTruthDir + '/*.xml')

    if len(ground_truth_files_list) == 0:
        print('No ground truth files were found in the specified path')
        exit(-1)

    groundTruth = {}
    gt_files_count = 0
    gt_classes_count = {}


    for fi in ground_truth_files_list:
        gt_files_count += 1
        fileID = os.path.basename(fi).split('.xml')[0]
        groundTruth[fileID] = []

        groundTruthFileContents = ET.parse(fi)
        root = groundTruthFileContents.getroot()

        for obj in root.findall('object'):
            gtBox = {}
            className = obj.find('name').text
            if className in gt_classes_count.keys():
                gt_classes_count[className] += 1
            else:
                gt_classes_count[className] = 1
            bb = obj.find('bndbox')
            leftX = int(bb.find('xmin').text)
            topY = int(bb.find('ymin').text)
            rightX = int(bb.find('xmax').text)
            bottomY = int(bb.find('ymax').text)
            bbox = [leftX,topY,rightX,bottomY]
            gtBox['used'] = False
            gtBox['class_name'] = className
            gtBox['bbox'] = bbox
            groundTruth[fileID].append(gtBox)
    return groundTruth,gt_classes_count

def validate_predictions():

    gt_boxes_per_file, gt_classes_count = createGroundTruthList()

    gt_classes = gt_classes_count.keys()

    n_classes = len(gt_classes)

    sum_AP = 0.0
    ap_dictionary = {}

    with open(results_files_path + "/results.txt", 'w') as results_file:
        results_file.write("# AP and precision/recall per class\n")
        count_true_positives = {}
        for class_index, class_name in enumerate(gt_classes):
            count_true_positives[class_name] = 0
            """
             Load predictions of that class
            """
            predictions_data, predictions_count = createPredictionsList(class_name)

            """
             Assign predictions to ground truth objects
            """
            nd = len(predictions_data)
            tp = [0] * nd  # creates an array of zeros of size nd
            fp = [0] * nd
            for idx, prediction in enumerate(predictions_data):
                file_id = prediction['file_id']

                # assign prediction to ground truth object if any
                #   get ground-truth with that file_id
                ground_truth_data = gt_boxes_per_file[file_id]
                ovmax = -1
                gt_match = -1
                # load prediction bounding-box
                bb = prediction["bbox"]
                for obj in ground_truth_data:
                    # look for a class_name match
                    if obj["class_name"] == class_name:
                        bbgt = obj["bbox"]
                        bi = [max(bb[0], bbgt[0]), max(bb[1], bbgt[1]), min(bb[2], bbgt[2]),
                              min(bb[3], bbgt[3])]  # Finding the coordinates of intersection
                        iw = bi[2] - bi[0] + 1
                        ih = bi[3] - bi[1] + 1
                        if iw > 0 and ih > 0:
                            # compute overlap (IoU) = area of intersection / area of union
                            ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + (bbgt[2] - bbgt[0]
                                                                              + 1) * (bbgt[3] - bbgt[1] + 1) - iw * ih
                            ov = iw * ih / ua
                            if ov > ovmax:
                                ovmax = ov
                                gt_match = obj

                # assign prediction as true positive/don't care/false positive
                if ovmax >= MINOVERLAP:

                    if not bool(gt_match["used"]):
                        # true positive
                        tp[idx] = 1
                        gt_match["used"] = True
                        count_true_positives[class_name] += 1

                    else:
                        # false positive (multiple detection)
                        fp[idx] = 1
                else:
                    # false positive
                    fp[idx] = 1

                """
                 Draw image to show animation
                """

            # print(tp)
            # compute precision/recall
            cumsum = 0
            for idx, val in enumerate(fp):
                fp[idx] += cumsum
                cumsum += val
            cumsum = 0
            for idx, val in enumerate(tp):
                tp[idx] += cumsum
                cumsum += val
            # print(tp)
            rec = tp[:]
            for idx, val in enumerate(tp):
                rec[idx] = float(tp[idx]) / gt_classes_count[class_name]
            # print(rec)
            prec = tp[:]
            for idx, val in enumerate(tp):
                prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])
            # print(prec)

            ap, mrec, mprec = voc_ap(rec, prec)
            sum_AP += ap
            text = "{0:.2f}%".format(
                ap * 100) + " = " + class_name + " AP  "  # class_name + " AP = {0:.2f}%".format(ap*100)
            """
             Write to results.txt
            """
            rounded_prec = ['%.2f' % elem for elem in prec]
            rounded_rec = ['%.2f' % elem for elem in rec]
            results_file.write(
                text + "\n Precision: " + str(rounded_prec) + "\n Recall   :" + str(rounded_rec) + "\n\n")
            print(text)
            ap_dictionary[class_name] = ap

        results_file.write("\n# mAP of all classes\n")
        mAP = sum_AP / n_classes
        text = "mAP = {0:.2f}%".format(mAP * 100)
        results_file.write(text + "\n")
        print(text)

        return mAP


def predict():

    testOptions = {'model': 'cfg/yolo-voc-1c.cfg',
                   'load': -1,
                   'gpu': 0.0,
                   'threshold': 0.5,
                   'labels': 'labels.txt',
                   'json': True,
                   'imgdir': validationImagesPath}

    network = TFNet(testOptions)

    network.predict()

def error(msg):
    print(msg)
    exit(-1)

def init(val_ImagesPath,groundTruthLabelsPath,resultsPath):

    '''Accessing the global variables'''
    global validationImagesPath,groundTruthDir,results_files_path

    '''validating and initializing the path of validation images'''
    if os.path.exists(val_ImagesPath) and os.path.isdir(val_ImagesPath):
        if len(os.listdir(val_ImagesPath)) != 0:
            validationImagesPath = val_ImagesPath
        else:
            msg = 'No validation images were found'
            error(msg)
    else:
        msg = 'Validation Folder does not exist'
        error(msg)

    '''validating and initializing the folder of ground truth images'''
    if os.path.exists(groundTruthLabelsPath) and os.path.isdir(groundTruthLabelsPath):
        if len(os.listdir(groundTruthLabelsPath)) != 0:
            groundTruthDir = groundTruthLabelsPath
        else:
            msg = 'No ground truth labels were found'
            error(msg)
    else:
        msg = 'Ground truth labels Folder does not exist'
        error(msg)

    '''validating and initializing the folder of ground truth images'''

    if os.path.exists(resultsPath):  # if it exist already
        # reset the results directory
        shutil.rmtree(resultsPath)

    os.makedirs(resultsPath)
    results_files_path = resultsPath


def validate(valImagesPath,groundTruthLabelsPath,resultsPath):
    init(valImagesPath,groundTruthLabelsPath,resultsPath)
    predict()
    mAP = validate_predictions()
    return mAP