#%%
from cProfile import label
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from vision_dataset import get_dataloaders
from model import build_model
from torchmetrics import Accuracy
import numpy as np
import matplotlib.pyplot as plt
from utils import calculate_flops, model_parameters, LatencyCounter
import json
#%%
def train_one_epoch(epoch, model, optimizer, criterion, train_loader, dev_loader, perf_counter):
    running_train_loss = 0.0
    train_acc = 0
    model.train()
    print(f'Epoch: {epoch}')
    print('--------------------------------------------------')
    print('Training...')
    model.cuda(2)
    accuracy = Accuracy().cuda(2)
    for i, data in enumerate(train_loader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.cuda(2)
        labels = labels.cuda(2)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_train_loss += loss.item()
        train_acc += accuracy(torch.argmax(outputs, dim=1), labels)
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
        test_acc += accuracy(torch.argmax(outputs, dim=1), labels)
    test_acc = test_acc / len(dev_loader)
    test_loss = running_test_loss / len(dev_loader)
    
    return model, train_loss, test_loss, train_acc.cpu().numpy(), test_acc.cpu().numpy() 
#%%
def run_experiment(train_loader, dev_loader, hidden_size, depth):
    model_time_counter = LatencyCounter()
    for batch in train_loader:
        x, y = batch
        input_size = x.shape[1]
        break
    # input_size = len(vocab)
    # print(f'Width: {input_size}')
    results = {}
    output_size = 10
    epochs = 3
    model = build_model(input_size,output_size, hidden_size, depth=depth)
    print(model)
    model_flops = calculate_flops(model)
    model_params = model_parameters(model)
    model#.cuda(1)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
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
experiments = []
hidden_sizes = [512]
sampling_rates = [0.5,0.6,0.7,0.8,0.9,1.0]
depths = [1]
num_cycles = 1
train_counter = LatencyCounter()
for donsample in sampling_rates:
    train_loader, dev_loader = get_dataloaders(
        test_batch_size=10, downsample=donsample,use_transform=True
    )
    for depth in depths:
        for hidden_size in hidden_sizes:
            for i in range(num_cycles):
                train_counter.tic()
                setting = {
                    'depth':depth,
                    'hidden_size':hidden_size,
                    'downsample':donsample
                }
                print(setting)
                results = run_experiment(train_loader, dev_loader, hidden_size, depth)
                experiment = {
                    'setting': setting,
                    'results':results
                }
                experiments.append(experiment)
                train_counter.toc()
train_counter.report_times()
#%%
plt.plot(train_counter.times)
plt.title('Training times for Vision model')
plt.xlabel('Training Cycle')
plt.ylabel('Time (s)')
#%%
print(len(experiments))
print(experiments)
with open('crop_experiments_vision.json', 'w') as f:
    f.write(json.dumps(experiments))

fig,(ax1, ax2, ax3) = plt.subplots(1,3, figsize=(15,5))

flops = []
latency = []
accuracy = []
key = sampling_rates
exps ={k:{'flops':[],'latency':[],'accuracy':[]} for k in key}
for exp in experiments:
    flops.append(exp['results']['model_flops'])
    latency.append(exp['results']['model_times'][0])
    accuracy.append(exp['results']['test_acc'])

ax1.scatter(latency, accuracy)
for i, d in enumerate(key):
    ax1.text(latency[i],accuracy[i],str(d))
ax1.set_title('Latency vs Accuracy')
ax1.set_xlabel('Latency')
ax1.set_xticklabels(np.round(ax2.get_xticks(), 5), rotation =45)
ax1.set_ylabel('Accuracy')

ax2.scatter(latency, accuracy)
for i, d in enumerate(key):
    ax2.text(latency[i],accuracy[i],str(d))
ax2.set_title('Latency vs Accuracy')
ax2.set_xlabel('Latency')
ax2.set_xticklabels(np.round(ax2.get_xticks(), 5), rotation =45)
ax2.set_ylabel('Accuracy')

ax3.scatter(flops, latency)
for i, d in enumerate(key):
    ax3.text(flops[i],latency[i],str(d))
ax3.set_title('FLOPs vs Latency')
ax3.set_xlabel('FLOPs')
ax3.set_xscale('log')
ax3.set_ylabel('Latency')
plt.tight_layout()