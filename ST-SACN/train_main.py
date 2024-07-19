import os
import random
import time

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from model.ST_SACN import ST_SACN
from util.data import load_custom_data
import csv




if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_path = os.path.abspath(os.path.join(os.getcwd(), './result/weigth'))
    path = "./data/DBLP3.npz"
    dataset = "DBLP3"
    adj_matrix, feature_matrix, labels, idx_train, idx_val, idx_test = load_custom_data(path, dataset)
    labels = labels.argmax(dim=1)
    adj_matrix = [matrix.to(device) for matrix in adj_matrix]
    feature_matrix = feature_matrix.to(device)
    labels = labels.to(device)
    idx_val = torch.cat([idx_val,idx_test])
    idx_train = idx_train.to(device)
    idx_val = idx_val.to(device)


    # 定义和初始化模型
    model = ST_SACN(in_feats=feature_matrix.shape[2], out_feats = 64,hidden_feats=16, num_layers=2, time_steps=feature_matrix.shape[1] ,classes=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # 生成mini-batch
    batch_size = 128
    num_epochs = 1000


    # 将训练数据封装成数据加载器
    train_data = TensorDataset(feature_matrix[idx_train], labels[idx_train])
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    # 训练循环
    Train_Losses = []
    val_Losses = []
    ACC = []
    F1 = []
    Precision = []
    Recall = []
    best_result = {"iter": 0, "val_acc": 0.}
    with open('./result/loss_acc_f1/metrics_11.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Loss', 'Accuracy', 'F1', 'Precision', 'Recall'])
        for epoch in range(num_epochs):
            start_time_t = time.time()
            print("Epoch %d Training starts at %s" % (epoch, time.asctime(time.localtime(time.time()))))

            losses = []
            model.train()
            epoch_loss = 0.0
            with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch') as pbar:
                for batch_features, batch_labels in train_loader:
                    batch_nodes = random.sample(range(batch_features.shape[0]), min(batch_size, batch_features.shape[0]))
                    optimizer.zero_grad()
                    output = model(batch_features, adj_matrix, num_samples_list=[15, 20], batch_nodes=batch_nodes)
                    loss = criterion(output, batch_labels)
                    loss.backward()
                    optimizer.step()
                    losses.append(loss.detach().cpu().numpy())
                    pbar.set_postfix({'loss': loss.item()})
                    pbar.update(1)
            print('[learning] epoch %i >> %3.2f%%' % (epoch, 100),
                  'completed in %.2f (sec) <<' % (time.time() - start_time_t))
            print('Training:\tEpoch : %d\tTime : %.4fs\t Loss: %.5f' % (epoch, time.time() - start_time_t, sum(losses) / len(losses)))

            Train_Losses.append(sum(losses) / len(losses))
            # 验证模型
            model.eval()
            val_features = feature_matrix[idx_val]
            val_labels = labels[idx_val]
            with torch.no_grad():
                start_time_v = time.time()
                print("Validating Epoch %d starts at %s" % (epoch, time.asctime(time.localtime(time.time()))))

                batch_nodes = list(range(val_features.shape[0]))
                val_output = model(val_features, adj_matrix, num_samples_list=[15, 20], batch_nodes=batch_nodes)
                val_loss = criterion(val_output, val_labels).item()
                acc = accuracy_score(val_output.argmax(dim=1).cpu(), val_labels.cpu())
                f1 = f1_score(val_output.argmax(dim=1).cpu(), val_labels.cpu(), average='macro')
                precision = precision_score(val_output.argmax(dim=1).cpu(), val_labels.cpu(), average='macro')
                recall = recall_score(val_output.argmax(dim=1).cpu(), val_labels.cpu(), average='macro')
                val_Losses.append(val_loss)
                ACC.append(acc)
                F1.append(f1)
                Precision.append(precision)
                Recall.append(recall)
                print('[Validating] epoch %i >> %3.2f%%' % (epoch, 100),
                      'completed in %.2f (sec) <<' % (time.time() - start_time_v))
                print('Validating:\tEpoch : %d\tTime : %.4fs\t Loss: %.5f' % (
                epoch, time.time() - start_time_v, val_loss))
                print('Validating:\tEpoch : %d\tACC : %.5f\t F1: %.5f\tPrecision : %.5f\t Recall: %.5f' % (
                epoch, acc, f1,precision,recall))

                if acc >= best_result["val_acc"]:
                    torch.save(model.state_dict(), os.path.join(save_path, 'Epoch%03d_Loss%.5f_ACC%.5f_F1%.5f.pth' % (
                        (epoch + 1, Train_Losses[-1], acc, f1))))
                    print('NEW BEST:\tEpoch %d\t Train_Loss: %.5f\t ACC: %.5f\t F1: %.5f' % (
                        epoch, Train_Losses[-1], acc, f1))
                    print('save successfully!')
                    best_result["val_acc"] = acc
                    best_result["iter"] = epoch
                writer.writerow([epoch + 1, sum(losses) / len(losses), acc, f1, precision, recall])
        print('FINAL BEST RESULT: \tEpoch : %d\tBest Val (ACC : %.4f)'
              % (best_result['iter'], best_result['val_acc']))

