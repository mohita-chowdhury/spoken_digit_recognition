import torch
import model

PATH = '/scratch/shared/nfs1/mohita/ufonia/speech_rec_pytorch/saved/models/speech_net.pth'

def test(test_gen, model):
	# dataiter = iter(test_gen)
	# images, labels = dataiter.next()

	# model.load_state_dict(torch.load(PATH))

	# outputs = model(images)


	# _, predicted = torch.max(outputs, 1)


	# print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
	#                               for j in range(4)))

	correct = 0
	total = 0
	with torch.no_grad():
	    for data in test_gen:
	        images, labels = data
	        outputs = model(images)
	        _, predicted = torch.max(outputs.data, 1)
	        print([labels, predicted])
	        total += labels.size(0)
	        correct += (predicted == labels).sum().item()

	print('Accuracy of the network on the 50 test images: %d %%' % (
	    100 * correct / total))