import torch
import torch.nn as nn
import torch.optim as optim
import os
import logging
import numpy as np
from tqdm import tqdm
import random
import shutil
import time
import pickle
from itertools import product
from SAC.train_helper import Network, create_data_loader_ros, check_folder
from datetime import datetime
import matplotlib.pyplot as plt
from config.config import load_config
from torch.utils.tensorboard import SummaryWriter
import subprocess

config_name = "./config/ground_robot.yaml"
config = load_config(config_name)

dataset_folder = config['data_folder']
action_file_name = "action_map.pkl"
prep_log = 'data_collection.log'
with open(os.path.join(dataset_folder, action_file_name), "rb") as f:
    data = pickle.load(f)
action_map = {}
for k, v in data.items():
    action_map[v] = k
    print(k, v)

#####################
random.seed(0)
INPUT_SIZE = config['bias_input']
OUTPUT_SIZE = len(action_map)
BATCH_SIZE = config['bias_batch']
EVAL_EPOCH = config['bias_eval']
NUM_EPOCHS = config['bias_epochs']
SAVE_EPOCH = EVAL_EPOCH
LEARNING_RATE = config['bias_lr']
SAMPLE_RATE = config['sample_rate']  # ts
gazebo_workspace_size = config['ws_size']  # meter
##############
# net_dims = [INPUT_SIZE, 4096, 2048, OUTPUT_SIZE]
net_dims = [INPUT_SIZE] + config['bias_net_dim'] + [OUTPUT_SIZE]
############## 
data_name = 'data.csv'
result_folder = config['bias_output_folder']
check_folder(result_folder)
now = datetime.now()
current_time = now.strftime("%Y_%m_%d_%H_%M_%S")
cur_folder_name = os.path.join(result_folder, current_time)
check_folder(cur_folder_name)
shutil.copy(os.path.join(dataset_folder, data_name), os.path.join(cur_folder_name, data_name))
shutil.copy(os.path.join(dataset_folder, action_file_name), os.path.join(cur_folder_name, action_file_name))
shutil.copy(os.path.join(dataset_folder, prep_log), os.path.join(cur_folder_name, prep_log))
shutil.copy(config_name, cur_folder_name)
shutil.copy(__file__, cur_folder_name)


def run_eval(model, loader):
    corrects = 0
    cnt = 0
    model.eval()
    for data, label in loader:
        data, label = data.to('cuda', non_blocking=True), label.to('cuda', non_blocking=True)
        output = model(data)
        _, pred = torch.max(output, 1)
        # print(pred)
        if pred == label:
            corrects += 1
        cnt += 1
        if cnt < 10:
            logging.info("data: {} | label: {} | pred: {}".format(data.cpu().detach().numpy(),
                                                                  label.cpu().detach().numpy(),
                                                                  pred.cpu().detach().numpy()))
    return corrects / cnt


def main(writer):
    logging.info("Batch: {} | Learning Rate: {}".format(BATCH_SIZE, LEARNING_RATE))
    logging.info("Gazebo workspace size: {}".format(gazebo_workspace_size))
    train_loader, test_loader = create_data_loader_ros(BATCH_SIZE, os.path.join(dataset_folder,data_name), gazebo_workspace_size)
    model = Network(net_dims, activation=nn.ReLU).net
    logging.info("\nModel:\n{}".format(model))
    torch.backends.cudnn.benchmark = True
    model.to('cuda')
    criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    logging.info("Criterion: {}".format(criterion))
    loss_list = []
    x_axis = []
    global_cnt = 0
    evl_cnt = 0
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config['bias_lr_scheduler_step'], gamma=config['bias_lr_scheduler_gamma'])
    for epoch_num in tqdm(range(1, NUM_EPOCHS + 1)):
        mean_loss, cnt = 0, 0
        mean_acc = 0
        for batch_idx, (data, label) in enumerate(train_loader):
            model.train()
            data, label = data.cuda(), label.cuda()
            output = model(data)
            loss = criterion(output, label)
            writer.add_scalar("Loss/train", loss, global_cnt)
            _, preds = torch.max(output, 1)
            train_acc = sum(preds==label)/len(label)
            writer.add_scalar("Acc/train", train_acc, global_cnt)
            mean_acc += train_acc
            mean_loss += loss.cpu().detach().numpy()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            cnt += 1
            global_cnt += 1
        scheduler.step()
        val = mean_loss / cnt
        acc_val = mean_acc / cnt
        x_axis.append(epoch_num)
        logging.info("Epoch->{} | avg_loss->{:.3f} | avg_acc->{:.3f}".format(epoch_num, val, acc_val))
        loss_list.append(val)
        if epoch_num % EVAL_EPOCH == 0:
            accuracy = run_eval(model, test_loader)
            logging.info("Accuracy at evaluation of epoch {} is {}".format(epoch_num, accuracy))
            writer.add_scalar("Accuracy/test", torch.tensor(accuracy), evl_cnt)
            evl_cnt += 1
        if epoch_num % SAVE_EPOCH == 0 or epoch_num == NUM_EPOCHS:
            save_name = 'model_' + 'epoch_' + str(epoch_num) + '.pth'
            save_model_position = os.path.join(cur_folder_name, save_name)
            torch.save(model.state_dict(), save_model_position)
            logging.info("Saving model at epoch: {}\n".format(epoch_num))
        writer.flush()  # flush to disk every episode
    
    plt.title("Training Loss of Biased Network")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    # plt.legend()
    plt.plot(x_axis, loss_list)
    plt.savefig(os.path.join(cur_folder_name, "loss.png"))


if __name__ == '__main__':
    log_name = os.path.join(cur_folder_name, "train.log")
    level = logging.INFO
    format = '%(message)s'
    handlers = [logging.FileHandler(log_name), logging.StreamHandler()]
    # noinspection PyArgumentList
    logging.basicConfig(level=level, format=format, handlers=handlers)
    logging.info("Batch -> {}\nTotal Epochs -> {}\nSample Rate -> {}".format(BATCH_SIZE, NUM_EPOCHS, SAMPLE_RATE))
    logging.info("Loading data from {}".format(data_name))
    logging.info("Trained model can be found in {}".format(cur_folder_name))
    logging.info("Label mapping rule:")
    for k, v in action_map.items():
        logging.info("{}: {}".format(k, v))
    start_time = time.time()

    writer_dir = os.path.join(cur_folder_name, "runs")
    writer = SummaryWriter(log_dir=writer_dir)
    command = "tensorboard --logdir=" + writer_dir
    # subprocess.run(["gnome-terminal", "-x", "sh", "-c", command])

    main(writer)
    logging.info("Program Time Elapsed: {} (minutes)".format(int((time.time()-start_time)/60)))
    writer.close()
