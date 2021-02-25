import matplotlib.pyplot as plt

train_loss_hist = []
test_loss_hist = []
train_acc = []
test_acc = []



with open('resultsResnet.txt') as f:
	lines = (line.rstrip() for line in f)
	lines = (line for line in lines if line)


	for line in lines:
		if line.split()[0] == 'train':
			train_loss_hist.append(float(line.split()[2]))
			train_acc.append(float(line.split()[4]))
		if line.split()[0] == 'test':
			test_loss_hist.append(float(line.split()[2]))
			test_acc.append(float(line.split()[4]))





# Post processing
plt.xlabel("Eğitim Epocları")
plt.plot(range(1,  len(train_loss_hist)+ 1), train_loss_hist, label="Eğitim kayip")
plt.plot(range(1,  len(train_loss_hist)+ 1), train_acc, label="Eğitim keskinlik")
plt.plot(range(1,  len(test_loss_hist)+ 1), test_loss_hist, label="Test kayip")
plt.plot(range(1,  len(test_acc)+ 1), test_acc, label="Test keskinlik")
plt.legend()

plt.show()