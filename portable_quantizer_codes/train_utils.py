import torch
import os
import torch.nn as nn

def test(model, test_loader):
	total, correct = 0, 0
	model.eval()
	with torch.no_grad():
		for batch_idx, (inputs, targets) in enumerate(test_loader):
			# print(batch_idx)
			inputs, targets = inputs.cuda(), targets.cuda()
			outputs = model(inputs)
			_, predicted = outputs.max(1)
			total += targets.size(0)
			correct += predicted.eq(targets).sum().item()
			acc = correct/total
			# if batch_idx % 10 == 0:
			# print('Acc: %.2f%% (%d/%d)'% (100. * acc, correct, total))
	print('Final acc: %.2f%% (%d/%d)'% (100. * acc, correct, total))
	model.train()
	return acc


def train(model, train_loader, test_loader, epoch, work_dir=None, lr=None):
	if work_dir is not None:
		if not os.path.exists(work_dir):
			os.makedirs(work_dir)
	else:
		work_dir = ''
	print('Training...')
	acc = test(model, test_loader)
	print('Accuray for model before fine-tuning = {}'.format(acc))
	
	crit = nn.CrossEntropyLoss().cuda()
	optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
	best_acc = 0
	for i in range(epoch):
		total_loss = 0
		for batch_idx, (inputs, targets) in enumerate(train_loader):
			optimizer.zero_grad()
			inputs, targets = inputs.cuda(), targets.cuda()
			outputs = model(inputs)
			loss = crit(outputs, targets)
			total_loss += loss.item()
			loss.backward()
			optimizer.step()
			if batch_idx % 50 == 9:
				print('Epoch = {}, iteration = {}, loss = {}'.format(i + 1, batch_idx + 1, total_loss))
				total_loss = 0
		acc = test(model, test_loader)
		if acc > best_acc:
			torch.save(model.state_dict(), work_dir + '/best_model.pth')
			best_acc = acc
		print('Accuray after fine-tuning epoch {} = {}, best accuray = {}'.format(epoch, acc, best_acc))
		torch.save(model.state_dict(), work_dir + '/epoch{}.pth'.format(i))
