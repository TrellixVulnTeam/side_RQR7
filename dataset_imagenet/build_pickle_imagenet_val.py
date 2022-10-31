import numpy as np
import cv2


dataset_path = '/datasets/Imagenet/'
dataset_list_path = '/datasets/val_map.txt'

x_test = []
y_test = []


with open(dataset_list_path) as f:
    cnt = 0
    content = f.readlines()
    for x in content:
        if cnt % 500 == 0:
            print(cnt) 
        x = x.strip().split(' ')
        data_path = dataset_path + x[0]
  
        img = cv2.imread(data_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)

        x_test.append(img_resized)
        y_test.append(int(x[1]))
        cnt += 1
        if cnt == 1000:
            break

np.savez('/datasets/ILSVRC2012_val_224_1000.npz', x_test=x_test, y_test=y_test)

print("Save finish!")



# dataset_path = '/datasets/ILSVRC2012_val_224.npz'
# val_dataset = np.load(dataset_path)

# x_test = val_dataset['x_test']
# y_test = val_dataset['y_test']

# print(type(x_test))
# print(x_test.shape)

# print(type(y_test))
# print(y_test.shape)

# val_images = x_test / 255.0
# val_images -= 0.5
# val_images *= 2

# val_labels = y_test + 1

