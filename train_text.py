#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataset import get_text_dataloaders
from model import build_model
from torchmetrics import Accuracy
import numpy as np
import json
import matplotlib.pyplot as plt
from utils import calculate_flops, model_parameters, LatencyCounter
#%%
def train_one_epoch(epoch, model, optimizer, criterion, train_loader, dev_loader, perf_counter):
    running_train_loss = 0.0
    train_acc = 0
    model.train()
    print(f'Epoch: {epoch}')
    print('--------------------------------------------------')
    print('Training...')
    model.cuda(1)
    accuracy = Accuracy().cuda(1)
    for i, data in enumerate(train_loader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.cuda(1)
        labels = labels.cuda(1)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_train_loss += loss.item()
        train_acc += accuracy(torch.argmax(outputs, dim=1), torch.argmax(labels, dim=1))
    train_acc = train_acc / len(train_loader)
    train_loss = running_train_loss / len(train_loader)
    print(f'[{epoch + 1}] train loss: {train_loss:.3f}')
    print(f'[{epoch + 1}] train acc: {train_acc:.3f}')
    print('--------------------------------------------------')
    print('Testing...')
    running_test_loss = 0.0
    test_acc = 0
    model.eval()
    model.cpu()
    accuracy = Accuracy()
    for i, data in enumerate(dev_loader):
        inputs, labels = data
        inputs = inputs#.cuda(1)
        labels = labels#.cuda(1)
        perf_counter.tic()
        outputs = model(inputs)
        perf_counter.toc()
        loss = criterion(outputs, labels)
        running_test_loss += loss.item()
        test_acc += accuracy(torch.argmax(outputs, dim=1), torch.argmax(labels, dim=1))
    test_acc = test_acc / len(dev_loader)
    test_loss = running_test_loss / len(dev_loader)
    print(f'[{epoch + 1}] test loss: {test_loss:.3f}')
    print(f'[{epoch + 1}] test acc: {test_acc:.3f}')
    return model, train_loss, test_loss, train_acc.cpu().numpy(), test_acc.cpu().numpy()
#%%
def run_experiment(train_loader, dev_loader, hidden_size, depth, vocab):
    model_time_counter = LatencyCounter()
    input_size = len(vocab)
    results = {}
    output_size = 2
    epochs = 3
    model = build_model(input_size,output_size, hidden_size, depth=depth)
    print(model)
    model_flops = calculate_flops(model)
    model_params = model_parameters(model)
    model#.cuda(1)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    for epoch in range(epochs):
        model, train_loss, test_loss, train_acc, test_acc = train_one_epoch(
            epoch, model, optimizer, criterion, train_loader, dev_loader, perf_counter=model_time_counter
        )
    print('Finished Training')
    model_times = model_time_counter.report_times() 
    print(model_times)
    model_time_counter.reset()
    results['model_params'] = model_params
    results['model_flops'] = model_flops
    results['model_times'] = model_times
    results['train_loss'] = train_loss
    results['test_loss'] = test_loss
    results['test_acc'] = test_acc.item(0)
    results['train_acc'] = train_acc.item(0)
    return results
#%%
[train_loader, dev_loader, vocab] = get_text_dataloaders(test_batch_size=10)
experiments = []
hidden_sizes = [128,256,512,1000,5000,15000]
depths = [1]
vocab_lengths = [int(len(vocab)*(i/100)) for i in [100]]
for length in vocab_lengths:
    [train_loader, dev_loader, vocab] = get_text_dataloaders(train_batch_size=50,test_batch_size=10, topk=length)
    for depth in depths:
        for hidden_size in hidden_sizes:
            
            setting = {
                'depth':depth,
                'hidden_size':hidden_size,
                'vocab_length':length
            }
            print(setting)
            results = run_experiment(train_loader, dev_loader, hidden_size, depth,vocab)
            experiment = {
                'setting': setting,
                'results':results
            }
            experiments.append(experiment)
            with open('all_mix_experiments_text.json', 'w') as f:
                f.write(json.dumps(experiments))
print(experiments)
with open('all_mix_experiments_text.json', 'w') as f:
    f.write(json.dumps(experiments))

#%%
[train_loader, dev_loader, vocab] = get_text_dataloaders(test_batch_size=10)
experiments = []
hidden_sizes = [512]
depths = [1]
vocab_lengths = [int(len(vocab)*(i/100)) for i in [10,25, 50, 75, 90, 100]]
for length in vocab_lengths:
    [train_loader, dev_loader, vocab] = get_text_dataloaders(train_batch_size=50,test_batch_size=10, topk=length)
    for depth in depths:
        for hidden_size in hidden_sizes:
            setting = {
                'depth':depth,
                'hidden_size':hidden_size,
                'vocab_length':length
            }
            print(setting)
            model = build_model(len(vocab),2, hidden_size, depth=depth)
            print(model)
            model_flops = calculate_flops(model)
            model_params = model_parameters(model)
            results = {}
            results['model_params'] = model_params
            results['model_flops'] = model_flops

            experiment = {
                'setting': setting,
                'results':results
            }
            experiments.append(experiment)

#%%
# # %%
# flops = []
# latency = []
# accuracy = []
# for i in vocab_lengths:
#     flops.append(experiments[i]['model_flops'])
#     latency.append(experiments[i]['model_times'][0])
#     accuracy.append(experiments[i]['test_acc'])
    
# # %%
# plt.plot(flops, accuracy)
# # depths = [128,256, 512, 1000, 5000, 15000]
# for i, d in enumerate(vocab_lengths):
#     plt.text(flops[i],accuracy[i],str(d))
# plt.title('FLOPs vs Accuracy')
# plt.xticks(rotation = 45)
# plt.xlabel('FLOPs')
# plt.xscale('log')
# plt.ylabel('Accuracy')
# # %%
# plt.plot(latency, accuracy)
# for i, d in enumerate(vocab_lengths):
#     plt.text(latency[i],accuracy[i],str(d))
# plt.title('Latency vs Accuracy')
# plt.xlabel('Latency')
# plt.ylabel('Accuracy')

# # %%
# plt.plot(flops, latency)
# for i, d in enumerate(vocab_lengths):
#     plt.text(flops[i],latency[i],str(d))
# plt.title('FLOPs vs Latency')
# plt.xlabel('FLOPs')
# plt.xscale('log')
# plt.ylabel('Latency')
# # %%