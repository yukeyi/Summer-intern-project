'''Trains a LSTM on the IMDB sentiment classification task.
The dataset is actually too small for LSTM to be of any advantage
compared to simpler, much faster methods such as TF-IDF + LogReg.
Notes:
- RNNs are tricky. Choice of batch size is important,
choice of loss and optimizer is critical, etc.
Some configurations won't converge.
- LSTM loss decrease patterns during training can be quite different
from what you see with CNNs/MLPs/etc.
'''
from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.models import Model, Input
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding, Lambda
from keras.layers import LSTM
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import Merge,merge
from keras.datasets import imdb
from keras.optimizers import Adam
from keras.utils import plot_model
from keras import backend as K
import numpy as np
import tensorflow as tf
import copy
import os
import random
import math
import time


# Name
LSTM_name = "LSTM_Weight"
Accuracy_name = "accuracy.txt"
Hidden_name = "hidden.txt"
RL_name = "RL_Weight"
whole_name = "WholeName"
randomName = str(random.randint(10000,99999))

# Embedding
max_features = 20000
maxlen = 200
maxLSTMlen = 24
embedding_size = 128

# Convolution
kernel_size = 5
filters = 64
pool_size = 4

# LSTM
lstm_output_size = 70

# Training
batch_size = 30
epochs = 2
data_use_train = 25000
data_use_test = 5000
best_result = 0
test_or_train = False
zeronum = 0
onenum = 0

# RL
timecost = 0.02
gamma = 1
gate = 0.5
rl_batch = 16
epsilon = 1
feature_size = 40

def load_data():
    print('Loading data...')
    (x_train, raw_y_train), (x_test, raw_y_test) = imdb.load_data(num_words=max_features)
    x_train = sequence.pad_sequences(x_train, maxlen=maxlen, padding='post')
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen, padding='post')
    return x_train, x_test,raw_y_train, raw_y_test

def build_network_LSTM():
    print('Build LSTM model...')
    model = Sequential()
    model.add(Embedding(max_features, embedding_size, input_length=maxlen, name = 'embedding'))
    model.add(Dropout(0.25, name='dropout'))
    model.add(Conv1D(filters,
                     kernel_size,
                     padding='valid',
                     activation='relu',
                     strides=1,name='cnn1'))
    model.add(MaxPooling1D(pool_size=pool_size,name='pooling'))
    model.add(LSTM(lstm_output_size,return_sequences=False,name='lstm'))
    model.add(Dense(1,name='dense'))
    model.add(Activation('sigmoid', name='sg'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    #plot_model(model, to_file='model_lstm.png')
    return model

def train_LSTM(model):
    print('Train LSTM...')
    if(os.path.isfile(LSTM_name) == False):  # 如果不存在就返回False
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(x_test, y_test))
        score, acc = model.evaluate(x_test, y_test,
                                    batch_size=batch_size)
        print('Test score:', score)
        print('Test accuracy:', acc)
        model.save_weights(LSTM_name)

def precision(i, label):
    if(label == 0):
        i = 1-i
    else:
        i = i
    if(i < gate):
        return 0
    else:
        return i
        #return 1

'''
def convert_data(accuracy, length):
    temp = copy.deepcopy(accuracy[:length])
    for j in range(0,len(temp)):
        temp[j] = precision(temp[j])
    temp2 = [0] * len(temp)
    for j in range(1,len(temp)):
        temp2[j] = -1 + sigma * (temp[j] - temp[j-1])
    temp2[0] = -1 + sigma * temp[0]
    vt = 0
    for t in reversed(range(0, len(temp2))):
        vt = vt * gamma + temp2[t][0]
        temp2[t] = vt
    return temp2
'''

def my_loss_result(y_true, y_pred):
    #print(y_true.shape)
    #print(y_pred.shape)
    temp1 = y_pred[:,-1][:,0]
    #print(temp1.shape)
    temp2 = y_true[:,-1][:,0]
    temp = K.binary_crossentropy(temp1, temp2)
    loss = K.mean(temp, axis=-1)
    #print(loss.shape)
    return loss

def my_loss_action(y_true, y_pred):
    #print(y_true.shape)
    #print(y_pred.shape)
    temp0 = [0] * rl_batch
    for i in range(0,rl_batch):
        temp0[i] = y_pred[i]
    temp1 = tf.concat(temp0,0)
    temp_vt = [0] * rl_batch
    temp_action = [0] * rl_batch
    for i in range(0,rl_batch):
        temp2 = y_true[i]
        temp3 = tf.transpose(temp2)
        temp_action[i] = temp3[0]
        temp_vt[i] = temp3[1]
    action = tf.concat(temp_action,0)
    vt = tf.concat(temp_vt, 0)
    #print(temp1.shape)
    #print(action.shape)
    #print(vt.shape)
    neg_log_prob = tf.reduce_sum(-tf.log(temp1 + 1e-10) * tf.one_hot(tf.cast(action,dtype='int32'), 3), axis=1)
    loss = tf.reduce_mean(neg_log_prob * vt)  # reward guided loss
    return loss

def build_network_whole():
    main_input = Input(shape=(None,), dtype='int32', batch_shape=(rl_batch, None), name='main_input')
    x = Embedding(output_dim=embedding_size, input_dim=max_features, name = 'embedding', trainable=True)(main_input)
    x = Dropout(0.25, name='dropout',trainable=True)(x)
    x = Conv1D(filters,
               kernel_size,
               padding='valid',
               activation='relu',
               strides=1, name='cnn1',trainable=True)(x)
    x = MaxPooling1D(pool_size=pool_size, name='pooling',trainable=True)(x)
    x = LSTM(lstm_output_size, return_sequences=True, name='lstm', stateful=True,trainable=True)(x)
    #feature_input = Input(dtype='float32', batch_shape=(rl_batch, None, feature_size), name='feature_input')
    #merged = merge([x, feature_input], mode='concat',name='merge')
    y = Dense(1, name='dense',trainable=True)(x)
    output_result = Activation('sigmoid', name='sg',trainable=True)(y)
    z = Dense(64, name='dense1')(x)
    z = Activation('tanh',name='tanh1')(z)
    z = Dense(64, name='dense2')(z)
    z = Activation('tanh', name='tanh2')(z)
    z = Dense(3, name='dense3')(z)
    output_action = Activation('softmax', name='sm')(z)
    model = Model(inputs=main_input, outputs=[output_result, output_action])
    #plot_model(model, to_file='model_whole.png')
    model.compile(loss=[my_loss_result,my_loss_action],
                  optimizer=Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08),
                  metrics=['accuracy'])
    model.load_weights(LSTM_name, by_name=True)
    return model

def build_network_predict_up():
    main_input = Input(shape=(None,), dtype='int32', batch_shape=(1, None), name='main_input')
    x = Embedding(output_dim=embedding_size, input_dim=max_features, name = 'embedding')(main_input)
    x = Dropout(0.25, name='dropout')(x)
    x = Conv1D(filters,
               kernel_size,
               padding='valid',
               activation='relu',
               strides=1, name='cnn1')(x)
    x = MaxPooling1D(pool_size=pool_size, name='pooling')(x)
    x = LSTM(lstm_output_size, return_sequences=True, name='lstm', stateful=True)(x)
    y = Dense(1, name='dense')(x)
    output_result = Activation('sigmoid', name='sg')(y)
    output_hidden = Lambda(lambda a: a, name='hidden')(x)
    model = Model(inputs=[main_input], outputs=[output_result,output_hidden])
    #plot_model(model, to_file='model_predict_up.png')
    model.load_weights(LSTM_name, by_name=True)
    return model

def build_network_predict_down():
    hidden_input = Input(dtype='float32', shape=(lstm_output_size,), name='hidden_input')
    #feature_input = Input(dtype='float32', shape=(feature_size,), name='feature_input')
    #merged = merge([hidden_input, feature_input], mode='concat',name='merge')
    z = Dense(64, name='dense1')(hidden_input)
    z = Activation('tanh',name='tanh1')(z)
    z = Dense(64, name='dense2')(z)
    z = Activation('tanh', name='tanh2')(z)
    z = Dense(3, name='dense3')(z)
    output_action = Activation('softmax', name='sm')(z)
    model = Model(inputs=hidden_input, outputs=[output_action])
    #plot_model(model, to_file='model_predict_down.png')
    return model

def build_network_state():
    model = Sequential()
    model.add(Dense(64, input_shape=(lstm_output_size,)))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='sgd')
    return model

def convert_list(oldlist):
    result = []
    for i in oldlist:
        result.append(i[0])
    return result

def convert_vt(acc, length, label):
    le = (length+1)//5
    reward = [-timecost]*le
    reward[-1] = precision(acc,label)
    for i in range(le-2,-1,-1):
        reward[i] = reward[i+1]*gamma+reward[i]
    vt = [0]*length
    for i in range(3,length,5):
        vt[i] = reward[(i-3)//5]
    return vt, reward


'''
def convert_feature(start, acc, output):
    tempfeature = []
    tempfeature.append(start/10)
    hhisto = [0] * 22
    for i in output[1][0][-1]:
        if (10 * i > 1):
            hhisto[21] += 1
        elif (10 * i < -1):
            hhisto[0] += 1
        else:
            hhisto[int(100 * i) + 11] += 1
    for i in range(0, len(hhisto)):
        hhisto[i] = hhisto[i] / 10
    for i in range(0,22):
        hhisto[i] = (hhisto[i]-7/22)/5
    tempfeature += hhisto
    if (acc[-1] > 0.5):
        templabel = 1
    else:
        templabel = 0
    ahisto = [0] * 10
    for i in acc:
        ahisto[int(abs(templabel - i) * 10)] += 1
    for i in range(0, len(ahisto)):
        ahisto[i] = ahisto[i] / (len(acc)*10)
    tempfeature += ahisto
    for i in range(0, 4):
        tempfeature.append(abs(1 - templabel - acc[-i - 1])/10)
    if (abs(acc[-1] - templabel) < abs(acc[-2] - templabel)):
        tempfeature.append(1/10)
    else:
        tempfeature.append(0)
    if (abs(acc[-2] - templabel) < abs(acc[-3] - templabel)):
        tempfeature.append(1/10)
    else:
        tempfeature.append(0)
    if (abs(acc[-3] - templabel) < abs(acc[-4] - templabel)):
        tempfeature.append(1/10)
    else:
        tempfeature.append(0)
    return tempfeature
'''
nojump = [5,3,2.5,2.5,2,2,1.5,1.5,1,1,1,1,1,1,1,1,1,1,1,1,1]
testnum0 = 0
testnum1 = 0
#nojump = [2,2,2,2,1.5,1.5,1.5,1.5,1,1,1,1,1,1,1,1,1,1,1,1,1]
def exploration(weight, length):
    #global testnum0
    #global testnum1
    e0 = math.exp(weight[0]/epsilon)
    e1 = math.exp(weight[1]/epsilon)
    #if(e0 > e1):
    #    testnum0+=1
    #else:
    #    testnum1+=1
    e2 = math.exp(weight[2]/epsilon)/nojump[length]
    sum = e0+e1+e2
    temp0 = (e0+e1)/(sum*2)
    temp1 = (e0+e1)/sum
    r = random.random()
    if(r < temp0):
        return 0
    elif(r < temp1):
        return 1
    else:
        return 2

def pre_calculate(count, model_up, model_down, av = False):
    # 0 read again  1 continue  2 stop
    global zeronum
    global onenum
    if(test_or_train == False):
        train_data = x_train[count]
    else:
        train_data = x_test[count]
    startflag = True
    padding = [0,0,0,0]
    length = 0
    now = 0
    actionlist = []
    input = []
    hidden = []
    #feature = []
    '''
    output1 = model_up.predict(np.array([train_data[0:20]]), batch_size=1, verbose=0)
    model_up.reset_states()
    output2 = model_up.predict(np.array([train_data[0:12]]), batch_size=1, verbose=0)
    output2 = model_up.predict(np.array([train_data[8:12].tolist()+train_data[12:20].tolist()]), batch_size=1, verbose=0)
    '''
    while(True):
        if(startflag):
            output = model_up.predict(np.array([train_data[now:now+20]]), batch_size=1, verbose=0)
            padding = train_data[now+16:now+20]
            startflag = False
        else:
            output = model_up.predict(np.array([padding.tolist()+train_data[now:now+20].tolist()]), batch_size=1, verbose=0)
            padding = train_data[now+16:now+20]
        length += 1
        input = input+train_data[now:now+20].tolist()
        hidden = hidden+[output[1][0][-1].tolist()]
        acc = output[0][0].tolist()[-1][0]
        #tempfeature = convert_feature(start, acc, output)
        #feature.append(copy.deepcopy(tempfeature))

        #prob_weights = model_down.predict({'hidden_input':np.array([output[1][0][-1]]), 'feature_input':np.array([tempfeature])}, verbose=0)
        prob_weights = model_down.predict({'hidden_input':np.array([output[1][0][-1]])}, verbose=0)
        if(av == False):
            action = np.argmax(prob_weights[0])                                               # just choose
        else:
            action = exploration(prob_weights[0], length)

        if (length >= 20):
            if(av):
                model_up.reset_states()
                actionlist = actionlist + [0, 0, 0, 0, 2]
                actionlist = actionlist[1:]
                action_vt = []
                le = len(actionlist)
                vt, reward = convert_vt(acc,le,y_train[count])
                for i in range(0,le):
                    action_vt.append([actionlist[i],vt[i]])
                return length, action_vt, input, reward, hidden
            else:
                model_up.reset_states()
                return length,acc
        elif (action == 2 or now == maxlen-20):
            if(av):
                model_up.reset_states()
                actionlist = actionlist + [0, 0, 0, 0, 2]
                actionlist = actionlist[1:]
                action_vt = []
                le = len(actionlist)
                vt, reward = convert_vt(acc,le,y_train[count])
                for i in range(0,le):
                    action_vt.append([actionlist[i],vt[i]])
                return length, action_vt, input, reward, hidden
            else:
                model_up.reset_states()
                return length,acc
        elif (action == 0):
            #print('loop')
            zeronum += 1
            actionlist = actionlist + [0,0,0,0,0]
            now += 20
        else:
            actionlist = actionlist + [0, 0, 0, 0, 1]
            now += 20
            onenum += 1


def test_on_train():
    global zeronum
    zeronum = 0
    global onenum
    onenum = 0
    layer_number = [0]*20
    correct_number = [0]*20
    # wrong_example = []
    sumOfReward = 0
    for i in range(0,data_use_train):
        if(i % 1000 == 0):
            print(i)
        length, accuracy = pre_calculate(i, model_up, model_down, False)
        true_reward = precision(accuracy,y_train[i])-(length-1)*timecost
        layer_number[length - 1] += 1
        sumOfReward += true_reward
        if (abs(y_train[i] - accuracy) < 0.5):
            correct_number[length - 1] += 1
        #else:
        #    wrong_example.append(i)
    print('On training data......')
    print(zeronum)
    print(onenum)
    print(sumOfReward)
    f = open('Btc' + str(timecost) + 'trs' + str(data_use_train) + 'tes' + str(data_use_test) + 'bs' + str(rl_batch) + 'exsm', 'a')
    f.write(str(sumOfReward))
    f.write('\n')
    f.write('total correct number: ' + str(np.sum(correct_number)))
    f.write('\n')
    f.close()
    print('layer_number:')
    print(layer_number)
    print('correct_number:')
    print(correct_number)
    print('total correct number: ' + str(np.sum(correct_number)))

def test_on_test():
    global zeronum
    zeronum = 0
    global onenum
    onenum = 0
    global best_result
    global test_or_train
    test_or_train = True
    layer_number = [0]*20
    correct_number = [0]*20
    correct_rate = [0] * 20
    sumOfReward = 0
    # wrong_example = []
    for i in range(0,data_use_test):
        length, accuracy = pre_calculate(i, model_up, model_down, False)
        true_reward = precision(accuracy, y_test[i]) - (length - 1) * timecost
        layer_number[length - 1] += 1
        sumOfReward += true_reward
        if (abs(y_test[i] - accuracy) < 0.5):
            correct_number[length - 1] += 1
        #else:
        #    wrong_example.append(i)
    print(zeronum)
    print(onenum)
    for i in range(0,20):
        correct_rate[i] = correct_number[i]/(layer_number[i]+0.001)
    print('On testing data......')
    print(sumOfReward)
    print('layer_number:')
    print(layer_number)
    print('correct_number:')
    print(correct_number)
    #print('correct_rate:')
    #print(correct_rate)
    print('total correct number: ' + str(np.sum(correct_number)))
    cost = 0
    for i in range(0,20):
        cost = cost + layer_number[i]*(i+1)
    print('total cost: '+str(cost))
    f = open('Btc' + str(timecost) + 'trs' + str(data_use_train) + 'tes' + str(data_use_test) + 'bs' + str(rl_batch) + 'exsm', 'a')
    f.write(str(sumOfReward))
    f.write('\n')
    f.write('layer_number:')
    f.write(str(layer_number))
    f.write('\n')
    f.write('correct_number:')
    f.write(str(correct_number))
    f.write('\n')
    f.write('correct_rate:')
    f.write(str(correct_rate))
    f.write('\n')
    f.write('total correct number: ' + str(np.sum(correct_number)))
    f.write('\n')
    f.write('total cost: '+str(cost))
    f.write('\n')
    f.close()
    #if(np.sum(correct_number) >= best_result):
    #    model.save(whole_name+str(np.sum(correct_number)))
    #    best_result = np.sum(correct_number)
    test_or_train = False


if __name__ == "__main__" :
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    x_train, x_test, y_train, y_test = load_data()
    model_baseline = build_network_LSTM()
    train_LSTM(model_baseline)
    model_up = build_network_predict_up()
    model_down = build_network_predict_down()
    model_fit = build_network_whole()
    model_Q = build_network_state()
    print('Train Reinforcement...')
    count = 0
    ep = 1
    memory_length = [0]*20
    memory_input = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
    memory_answer = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
    memory_action_vt = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
    memory_hidden = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
    memory_reward = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
    memory_fit_hidden = []
    memory_fit_reward = []
    #memory_feature = [[],[],[],[],[]]
    print('epoch: 0')
    #model_down.load_weights('84011', by_name=True)
    #model_up.load_weights('84011', by_name=True)
    #test_on_train()
    #test_on_test()
    '''
    test sample reward

    sam_length = []
    sam_action_vt = []
    sam_reward = []
    for si in range(0,100):
        length, action_vt, _, reward, _ = pre_calculate(0, model_up, model_down, True)
        sam_length.append(length)
        sam_action_vt.append(copy.deepcopy(action_vt))
        sam_reward.append(copy.deepcopy(reward))
    '''
    while(True):
        if(count % 500 == 0):
            print(count)
        #print("time 1")
        #print(time.time())

        length, action_vt, input, reward, hidden = pre_calculate(count, model_up, model_down, True)
        answer = [[y_train[count]]] * (len(action_vt))
        memory_input[length-1].append(input)
        memory_answer[length-1].append(answer)
        memory_action_vt[length-1].append(action_vt)
        memory_hidden[length-1].append(hidden)
        memory_reward[length-1].append(reward)
        memory_fit_hidden = memory_fit_hidden + hidden
        memory_fit_reward = memory_fit_reward + reward
        #memory_feature[length-1].append(feature)
        memory_length[length-1]+=1
        #print("time 2")
        #print(time.time())
        for i in range(0,20):
            if(memory_length[i]>=rl_batch):
                #print("time 3")
                #print(time.time())
                model_Q.fit(np.array(memory_fit_hidden), np.array(memory_fit_reward), epochs=10, verbose=0)
                #print("time 4")
                #print(time.time())
                base = model_Q.predict(np.array([j for row in memory_hidden[i] for j in row]), verbose=0)
                #print("time 5")
                #print(time.time())
                for j in range(0,rl_batch):
                    le = len(memory_action_vt[i][j])
                    for t in range(3, le, 5):
                        memory_action_vt[i][j][t][1] -= base[j*(i+1)+(t-3)//5][0]
                #print("time 6")
                #print(time.time())
                model_fit.fit({'main_input': np.array(memory_input[i])},
                          {'sg': np.array(memory_answer[i]),'sm': np.array(memory_action_vt[i])}, batch_size=rl_batch, epochs=1, verbose=0)
                #print("time 7")
                #print(time.time())
                memory_length[i] = 0
                memory_input[i] = []
                memory_answer[i] = []
                memory_action_vt[i] = []
                memory_hidden[i] = []
                memory_reward[i] = []
                memory_fit_hidden = []
                memory_fit_reward = []
                #memory_feature[i] = []
                model_fit.reset_states()
                model_fit.save_weights(randomName)
                model_down.load_weights(randomName, by_name=True)
                model_up.load_weights(randomName, by_name=True)
                #print("time 8")
                #print(time.time())

        count = count+1
        #print("time 9")
        #print(time.time())
        if(count % 5000 == 0):
            f = open('Btc'+str(timecost)+'trs'+str(data_use_train)+'tes'+str(data_use_test)+'bs'+str(rl_batch)+'exsm', 'a')
            f.write('epoch: '+str(ep))
            f.write('\n')
            f.close()
            print('epoch: '+str(ep))
            #test_on_train()
            test_on_test()
        if (count == data_use_train):
            count = 0
            ep+=1
            epsilon = 1/math.sqrt(ep)