import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import time

device = torch.device("cuda")
criterion = nn.CrossEntropyLoss()
sigmoid=nn.Sigmoid()
PATH = '/scratch/shared/nfs1/mohita/ufonia/speech_rec_pytorch/saved/models/speech_net_2.pth'
writer = SummaryWriter('/scratch/shared/nfs1/mohita/ufonia/speech_rec_pytorch/saved/log/runs_2') 

def train(train_gen, max_epochs, n_batches, model, optimizer):
    # Train model
    # Loop over epochs
    
    initial_time = time.time()
    for epoch in range(max_epochs):
        initial_time_epoch = time.time()
        running_loss = 0.0
        print('Epoch #', epoch)
        for i, data in enumerate(tqdm(train_gen)):
            # Transfer to GPU
            # local_batch, local_labels = local_batch.to(device), local_labels.to(device)
            local_batch, local_labels =  data
            optimizer.zero_grad()

            # Model computation
            predicted_labels = model(local_batch)
            loss = criterion(sigmoid(predicted_labels), local_labels)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()
        time_elapsed_epoch = time.time() - initial_time_epoch
        print('This epoch took {:.0f}h {:.0f}m {:0f}s'.format(time_elapsed_epoch //3600, (time_elapsed_epoch % 3600) // 60,((time_elapsed_epoch % 3600)%60)%60))
        writer.add_scalars('Training_Loss', {'Loss':running_loss}, epoch)
        print('This epoch loss:', running_loss)
    print('Finished Training')
    
    torch.save(model.state_dict(), PATH)
    writer.close()
    time_elapsed = time.time() - initial_time

    print('The Training took {:.0f}h {:.0f}m {:0f}s'.format(time_elapsed //3600, (time_elapsed % 3600) // 60,((time_elapsed % 3600)%60)%60))
