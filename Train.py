from skimage.feature import hog
import joblib, glob, os, cv2
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn import svm, metrics
import numpy as np
from sklearn.preprocessing import LabelEncoder

pos_data = []
pos_lbl = []
neg_data = []
neg_lbl = []

pos_im_path = 'C:/Users/1/Desktop/UNI/7/computer vision/project/myproj/DATAIMAGE/positive/train/'
neg_im_path = 'C:/Users/1/Desktop/UNI/7/computer vision/project/myproj/DATAIMAGE/negative/train/'
model_path = 'models/models.dat'

# Load the positive features
for filename in glob.glob(os.path.join(pos_im_path, "*.png")):
    fd = cv2.imread(filename, 0)
    fd = cv2.resize(fd, (64, 128))
    fd = hog(fd, orientations=9, pixels_per_cell=(8, 8), visualize=False, cells_per_block=(3, 3))
    pos_data.append(fd)
    pos_lbl.append(1)

# Load the negative features
for filename in glob.glob(os.path.join(neg_im_path, "*.jpg")):
    fd = cv2.imread(filename, 0)
    fd = cv2.resize(fd, (64, 128))
    fd = hog(fd, orientations=9, pixels_per_cell=(8, 8), visualize=False, cells_per_block=(3, 3))
    neg_data.append(fd)
    neg_lbl.append(0)

# numpy array
pos_data = np.float32(pos_data)
pos_lbl = np.array(pos_lbl)

neg_data = np.float32(neg_data)
neg_lbl = np.array(neg_lbl)

# train_test_split
pos_data_train, pos_data_test, pos_lbl_train, pos_lbl_test = train_test_split(pos_data, pos_lbl, test_size=0.2,
                                                                              random_state=1381)
neg_data_train, neg_data_test, neg_lbl_train, neg_lbl_test = train_test_split(neg_data, neg_lbl, test_size=0.2,
                                                                              random_state=1381)

# train data
train_data = np.append(pos_data_train, neg_data_train, axis=0)
train_labels = np.append(pos_lbl_train, neg_lbl_train, axis=0)

# test data
test_data = np.append(pos_data_test, neg_data_test, axis=0)
test_labels = np.append(pos_lbl_test, neg_lbl_test, axis=0)

print('Train Data:', len(train_data))
print('Train Labels (1,0)', len(train_labels))

model = LinearSVC()
print('Training...... Support Vector Machine')
model.fit(train_data, train_labels)

print("Fit Score:", model.score(train_data, train_labels))
print("Test Score:", model.score(test_data, test_labels))

joblib.dump(model, 'models/models.dat')
print('Model saved : {}'.format('models/models.dat'))
