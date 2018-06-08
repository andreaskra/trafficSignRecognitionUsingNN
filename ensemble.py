import network
import numpy as np
import cPickle


def accuracy_max(classification, test_y):
    """

    The classification of the network that returned the highest probability for it's
    decision is selected as ensemble-output.

    """


    classification_final = []
    wrongs = []
    max_prob_per_example = np.zeros(len(classification[0]))
    max_class_per_example = np.zeros(len(classification[0]))

    for i in range(len(classification)):        #12

        for j in range(len(classification[i])):     #12630

            max = np.argmax(classification[i][j])

            if classification[i][j][max] > max_prob_per_example[j]:

                max_prob_per_example[j] = classification[i][j][max]
                max_class_per_example[j] = max


            if i == len(classification)-1:

                final_class = int(max_class_per_example[j])
                classification_final.append(final_class)

                #if max_class_per_example[j] != test_y[j]:
                    #wrongs.append(j)

    writer = network.CsvWriter('wrong_classifications.csv')
    writer.open()
    writer.write(["true class", "classification", "probability"])

    for j in wrongs:
        line = []
        line.append(test_y[j])
        line.append(max_class_per_example[j])
        line.append(max_prob_per_example[j])
        line.append("")
        line.append("")

        for i in range(len(classification)):
            max = np.argmax(classification[i][j])
            prb = classification[i][j][max]
            line.append(max)
            line.append(prb)
            line.append("")
            line.append("")

        writer.write(line)

    writer.close()

    return np.equal(classification_final, test_y)


def accuracy_maxVote(classification, test_y):
    """

    Every network gets to vote with it's classification on the final result.
    The class that received the most votes is selected.

    """


    classification_final = []

    d1 = np.size(classification[0])
    d2 = np.size(classification[0][0])

    votes_per_class_per_example = np.zeros((d1, d2)) #12630x43  votes per class per example


    for i in range(len(classification)):        #12

        for j in range(len(classification[i])):     #12630

            max = np.argmax(classification[i][j])
            votes_per_class_per_example[j][max] = votes_per_class_per_example[j][max] + 1

            if i == len(classification)-1:

                final_class = np.argmax(votes_per_class_per_example[j])
                classification_final.append(final_class)


    return np.equal(classification_final, test_y)


def accuracy_maxVote_ultmax(classification, test_y):
    """
    Every network gets to vote with it's classification on the final result.
    The class that received the most votes is selected.

    In addition to voting, the network that returned the highest probability of all is given
    an extra 5 votes in order to make it harder for several unsure networks to overrule the surest one.

    """


    classification_final = []

    max_prob_per_example = np.zeros(len(classification[0]))
    max_class_per_example = np.zeros(len(classification[0]))
    votes_per_class_per_example = np.zeros((len(classification[0]), len(classification[0][0]))) #12630x43  votes per class per example


    for i in range(len(classification)):        #12

        for j in range(len(classification[i])):     #12630

            max = np.argmax(classification[i][j])
            votes_per_class_per_example[j][max] = votes_per_class_per_example[j][max] + 1

            if classification[i][j][max] > max_prob_per_example[j]:

                max_prob_per_example[j] = classification[i][j][max]
                max_class_per_example[j] = max


            if i == len(classification)-1:

                ultmax = int(max_class_per_example[j])
                votes_per_class_per_example[j][ultmax] = votes_per_class_per_example[j][ultmax] + 5

                final_class = np.argmax(votes_per_class_per_example[j])
                classification_final.append(final_class)


    return np.equal(classification_final, test_y)


def accuracy_maxVote_ultmax_probcast(classification, test_y, weight=5):
    """

    Every network gets to vote with it's classification on the final result.
    The class that received the most votes is selected.

    Instead of counting every network's decision as one vote, the probabilities
    for the decisions are counted and the network with the highest probability
    is given extra votes by multiplying it's probability by 5.

    """

    classification_final = []

    max_prob_per_example = np.zeros(len(classification[0]))
    max_class_per_example = np.zeros(len(classification[0]))
    votes_per_class_per_example = np.zeros((len(classification[0]), len(classification[0][0]))) #12630x43  votes per class per example


    for i in range(len(classification)):        #12

        for j in range(len(classification[i])):     #12630

            max = np.argmax(classification[i][j])
            votes_per_class_per_example[j][max] = votes_per_class_per_example[j][max] + classification[i][j][max]

            if classification[i][j][max] > max_prob_per_example[j]:

                max_prob_per_example[j] = classification[i][j][max]
                max_class_per_example[j] = max


            if i == len(classification)-1:

                ultmax = int(max_class_per_example[j])
                votes_per_class_per_example[j][ultmax] = votes_per_class_per_example[j][ultmax] + max_prob_per_example[j]*weight

                final_class = np.argmax(votes_per_class_per_example[j])
                classification_final.append(final_class)


    return np.equal(classification_final, test_y)


def accuracy_maxVote_ultmax_probcast_doubtfilter(classification, test_y, weight=5, doubt=0.7):
    """

    Every network gets to vote with it's classification on the final result.
    The class that received the most votes is selected.

    Instead of counting every network's decision as one vote, the probabilities
    for the decisions are counted and the network with the highest probability
    is given extra votes by multiplying it's probability by 5.

    Classifications with probabilities under 70% are not allowed to vote.

    """

    classification_final = []

    max_prob_per_example = np.zeros(len(classification[0]))
    max_class_per_example = np.zeros(len(classification[0]))
    votes_per_class_per_example = np.zeros((len(classification[0]), len(classification[0][0]))) #12630x43  votes per class per example


    for i in range(len(classification)):        #12

        for j in range(len(classification[i])):     #12630

            max = np.argmax(classification[i][j])

            if classification[i][j][max] > doubt:
                votes_per_class_per_example[j][max] = votes_per_class_per_example[j][max] + classification[i][j][max]

            if classification[i][j][max] > max_prob_per_example[j]:

                max_prob_per_example[j] = classification[i][j][max]
                max_class_per_example[j] = max


            if i == len(classification)-1:

                ultmax = int(max_class_per_example[j])
                votes_per_class_per_example[j][ultmax] = votes_per_class_per_example[j][ultmax] + max_prob_per_example[j]*weight

                final_class = np.argmax(votes_per_class_per_example[j])
                classification_final.append(final_class)


    return np.equal(classification_final, test_y)



def ensemble(networks=None, data=None, test_names=None):
    """

    Classifies the data using the provided networks.

    """

    fileName = './Ensemble/testClassifications.pkl'
    nets = []
    classification = []

    if data is None:

        t = file("./Ensemble/correctTestLabels.pkl", 'rb')
        test_y, test_names = cPickle.load(t)
        t.close()

    else:
        test_y = data[1].eval()

        t = file("./Ensemble/correctTrainingLabels.pkl", 'wb')
        cPickle.dump((test_y, test_names), t)
        t.close()


    if networks is None:

        f = file(fileName, 'rb')
        classification = cPickle.load(f)
        f.close()

    else:
        for name in networks:
            net = network.load(name)
            #nets.append(net)

            mbs = 1
            net.set_mini_batch_size(mbs)

            classification.append(net.classify(data))

            acc = net.evaluate(data)
            print "loaded {0}, accuracy: {1}".format(name, acc)


        f = file(fileName, 'wb')
        cPickle.dump(classification, f)
        f.close()

    acc = np.mean(accuracy_max(classification, test_y))
    print "accuracy_max:                                 {0:.3%}".format(acc)

    acc = np.mean(accuracy_maxVote(classification, test_y))
    print "accuracy_max:                                 {0:.3%}".format(acc)

    acc = np.mean(accuracy_maxVote_ultmax(classification, test_y))
    print "accuracy_max:                                 {0:.3%}".format(acc)

    acc = np.mean(accuracy_maxVote_ultmax_probcast(classification, test_y))
    print "accuracy_max:                                 {0:.3%}".format(acc)

    acc = np.mean(accuracy_maxVote_ultmax_probcast_doubtfilter(classification, test_y))
    print "accuracy_max:                                 {0:.3%}".format(acc)


































