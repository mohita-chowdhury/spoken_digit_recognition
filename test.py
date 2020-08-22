import torch
import model


device = torch.device("cuda")

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
	        images, labels = images.to(device), labels.to(device)
	        outputs = model(images)
	        # import pdb; pdb.set_trace()
	        _, predicted = torch.max(torch.unsqueeze(outputs.data,0), 1)
	        print([labels, predicted])
	        total += labels.size(0)
	        correct += (predicted == labels).sum().item()

	print('Accuracy of the network on the 100 test images: %d %%' % (
	    100 * correct / total))