import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import time

device = torch.device("cuda")
criterion = nn.CrossEntropyLoss()
sigmoid=nn.Sigmoid()
folder_name = '/scratch/shared/nfs1/mohita/ufonia/speech_rec_pytorch/saved/models'
PATH = 'speech_net_aug_30.pth.tar'
writer = SummaryWriter('/scratch/shared/nfs1/mohita/ufonia/speech_rec_pytorch/saved/log/runs__aug_30') 

def train(train_gen, max_epochs, n_batches, model, optimizer):
    # Train model
    # Loop over epochs
    
    initial_time = time.time()
    min_running_loss = 100000
    for epoch in range(max_epochs):
        initial_time_epoch = time.time()
        running_loss = 0.0
        print('Epoch #', epoch)
        for i, data in enumerate(tqdm(train_gen)):
            # Transfer to GPU
            # local_batch, local_labels = local_batch.to(device), local_labels.to(device)
            local_batch, local_labels =  data
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)
            optimizer.zero_grad()

            # import pdb; pdb.set_trace()
            # Model computation
            predicted_labels = model(local_batch)
            loss = criterion(torch.unsqueeze(predicted_labels,0), local_labels)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()
            

            if running_loss < min_running_loss:
                min_running_loss = running_loss
                torch.save(model.state_dict(), folder_name +'/'+'bestcheckpoint_aug_30.pth.tar')
        

        time_elapsed_epoch = time.time() - initial_time_epoch
        print('This epoch took {:.0f}h {:.0f}m {:0f}s'.format(time_elapsed_epoch //3600, (time_elapsed_epoch % 3600) // 60,((time_elapsed_epoch % 3600)%60)%60))
        

        writer.add_scalars('Training_Loss', {'Loss':running_loss}, epoch)
        print('This epoch loss:', running_loss)
    
    print('Finished Training')
    
    torch.save(model.state_dict(), folder_name + '/' + PATH)
    writer.close()
    time_elapsed = time.time() - initial_time

    print('The Training took {:.0f}h {:.0f}m {:0f}s'.format(time_elapsed //3600, (time_elapsed % 3600) // 60,((time_elapsed % 3600)%60)%60))
