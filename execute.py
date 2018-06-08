import trafficSigns_loader
import time
import ensemble
import network
from network import Network
from network import Trainer


def optimize(training_data, validation_data, test_data, writer, eta=0.03, lmbda=0.1, my=0.007, max_epochs=20,
             mini_batch_size=25, dropout=0.7, net_name="net"):

    trainer = Trainer(training_data, validation_data, test_data, eta, lmbda, my, max_epochs, writer, mini_batch_size, dropout, net_name)

    trainer.optimizeParameter(3, "lmbda")
    trainer.optimizeParameter(3, "my")
    trainer.optimizeParameter(3, "eta")

    trainer.max_epochs = 200
    trainer.train(early_stopping=False)

    return trainer




#open a file to log results

writer = network.CsvWriter('./results.csv')
writer.open()


#measure data loading time

start = time.time()
training_data, validation_data, test_data,\
training_names, validation_names, test_names = trafficSigns_loader.load_data_from_images(smallSet=True)

print "data loaded in time [s]: {0}".format(time.time() - start)


#measure execution time

start = time.time()



#train a new network using test and validation data

net = Network(network.getLayers(20, 0.5), writer, 20, "./network")
net.MBSGD(training_data, validation_data, test_data, 16, 0.1, 0.1, 0.008, False)
net.save()




"""
#optimize parameters to a network setup with validation data

optimize(training_data, validation_data, test_data, writer)
"""



"""
#load the network 'network' and train it for 16 epochs, then save it as 'network2'

net = network.load("./network")
net.writer = writer
net.SGD(training_data, validation_data, test_data, 16, 0.1, 0.1, 0.008, False)
net.save("./network2")
"""



"""
#load these networks and do an ensemble classification on the provided data, storing wrong classifications as .csv

ensemble(["./Nets/storedNet1", "./Nets/storedNet2", "./Nets/storedNet3", "./Nets/storedNet4", "./Nets/multitrain1", "./Nets/obstructed_final",
          "./Nets/loweredcontrast_final", "./Nets/blurred_final", "./Nets/obstructed_longTraining", "./Nets/obstructed_final_2",
          "./Nets/contrastlowered_final_2", "./Nets/blurred_final_2"], test_data, test_names)
"""



print "finished in time [s]: {0}".format(time.time() - start)

writer.close()