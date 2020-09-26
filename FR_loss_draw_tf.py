import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--train_log', type=str, 
                    default='None', help='Training log path.')
parser.add_argument('--eval_log', type=str, 
                    default='None', help='Evaluation log path.')
parser.add_argument('--eval_start', type=int,
                    default=0, help='Evaluation start steps.')

args = parser.parse_args()

'''
python ./FR_loss_draw_tf.py \
--train_log ../logs/QAT_FR_softmax_mobilefacenet_relu.log \
--eval_log ../logs/QAT_FR_softmax_mobilefacenet_relu_eval.log  \
--eval_start=60
'''

train_path_root = args.train_log
eval_path_root = args.eval_log
e_start = args.eval_start
model_name = train_path_root.split('/')[-1].split('.')[0]

training_data = []
validation_data = []
evaluation_data = []
with open(train_path_root, 'r') as f:
    for line in f:
        if 'Training Loss' in line:
            row = line.split(',')
            for i, col in enumerate(row):
                if i == 2:
                    col = col.split('(')[0]
                if 'Loss' in col:
                    if 'arcface' in train_path_root:
                        loss = float(col.split('=')[-1])
                    else:
                        loss = float((col.split('[')[-1]).split(']')[0])
                elif 'Step' in col:
                    steps = int(col.split('=')[-1])
                elif 'Rate' in col:
                    lr = float(col.split('=')[-1])
            training_data.append([steps, loss, lr])
with open(eval_path_root, 'r') as f:
    for line in f:
        if 'lfw_eu1.19_acc' in line:
            # line_count += 1
            row = line.split(',')
            # acc = 0
            for i, col in enumerate(row):
                if i == 4:
                    col = col.split('(')[0]
                if 'loss' in col:
                    loss = float(col.split('=')[-1])
                elif 'global_step' in col:
                    steps = int(col.split('=')[-1])
                # elif 'Rate' in col:
                #     lr = float(col.split('=')[-1])
                elif 'acc' in col:
                    acc = float(col.split('=')[-1])
            evaluation_data.append([steps, loss, acc])

training_data = np.array(training_data)
# validation_data = np.array(validation_data)
evaluation_data = np.array(evaluation_data)
print('Model name: {}'.format(model_name))
print('Training Data:', training_data.shape)
print('Evaluation Data:', evaluation_data.shape)

P = plt.figure(1)

p1 = plt.subplot(211)

plt.plot(training_data[:, 0], training_data[:, 1], 'b')
plt.plot(evaluation_data[e_start:, 0], evaluation_data[e_start:, 1], 'k')

min_e_loss = np.min(evaluation_data[e_start:, 1])
min_e_ind = evaluation_data[np.argmin(evaluation_data[e_start:, 1])+e_start, 0]

min_t_loss = np.min(training_data[:, 1])
min_t_ind = training_data[np.argmin(training_data[:, 1]), 0]
print('Minimum eval loss: {}, at steps: {}'.format(min_e_loss, min_e_ind))
print('Minimum train loss: {}, at steps: {}'.format(min_t_loss, min_t_ind))
plt.plot(min_e_ind, min_e_loss, 'rx')
plt.plot(min_t_ind, min_t_loss, 'r*')

plt.ylabel("Loss")
plt.title(
    model_name + '\n' + 'Training Loss: blue line   ' +
    ', Evaluation Loss: black line')
P.text(0.48,0.875,'-', ha='center', va='bottom', size=24,color='blue')
P.text(0.87,0.875,'-', ha='center', va='bottom', size=24,color='black')
p2 = plt.subplot(212)

plt.plot(evaluation_data[e_start:, 0], evaluation_data[e_start:, 2], 'g')

max_e_acc = np.max(evaluation_data[e_start:, 2])
max_e_ind_acc = evaluation_data[np.argmax(evaluation_data[e_start:, 2])+e_start, 0]

print('Maxmum eval lfw_eu1.19_acc: {}, at steps: {}'.format(max_e_acc, max_e_ind_acc))
plt.plot(max_e_ind_acc, max_e_acc, 'rx')

plt.xlabel("Steps")
plt.ylabel("Evaluation LFW Acc")
plt.grid()
plt.show()
