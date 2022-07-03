from random import shuffle
import numpy as np
import matplotlib.pyplot as plt


# to make data no change
np.random.seed(2022)

def generate_linear(n=100):
    pts=np.random.uniform(0,1,(n,2))
    inputs=[]
    labels=[]
    for pt in pts:
        inputs.append([pt[0],pt[1]])
        distance=(pt[0]-pt[1])/1.414
        if pt[0]>pt[1]:
            labels.append(0)
        else:
            labels.append(1)
    return np.array(inputs), np.array(labels).reshape(n,1)

def generate_XOR_easy():
    inputs=[]
    labels=[]
    for i in range(11):
        inputs.append([0.1*i,0.1*i])
        labels.append(0)
        if 0.1*i==0.5:
            continue
        inputs.append([0.1*i,1-0.1*i])
        labels.append(1)
    return np.array(inputs), np.array(labels).reshape(21,1)

def tanh(x):
    return (1-np.exp(-2*x))/(1+np.exp(-2*x))

def derivative_tanh(x):
    return (4*np.exp(-2*x))/np.square(1+np.exp(-2*x))


def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def derivative_sigmoid(x):
    return np.array(np.multiply(x,1.0-x))

def show_result(x,y,pred_y,filename="result.png"):
    plt.clf()
    plt.subplot(1,2,1)
    plt.title('Ground Truth',fontsize=18)
    for i in range(x.shape[0]):
        if y[i]==0:
            plt.plot(x[i][0],x[i][1],'ro')
        else:
            plt.plot(x[i][0],x[i][1],'bo')
    
    plt.subplot(1,2,2)
    plt.title('Predict Truth',fontsize=18)
    for i in range(x.shape[0]):
        if pred_y[i]==0:
            plt.plot(x[i][0],x[i][1],'ro')
        else:
            plt.plot(x[i][0],x[i][1],'bo')
    plt.savefig(f'./{filename}.png')
    plt.show()
    


def linear_back_propogation(X,y,learn_rate=0.01,n_hidden=[4,8],epoch=1000,filename=""):
    # set random (cancel prior affect)
    np.random.seed()
    # neuron in hidden layer
    n_inputSize=X.shape[1]
    w_hidden_1=np.random.rand(n_inputSize,n_hidden[0])
    w_hidden_2=np.random.rand(n_hidden[0],n_hidden[1])
    w_output=np.random.rand(n_hidden[1],1)

    # numbers of smple 
    n_sample=X.shape[0]
    min_batch=1
    Loss=[]
    
    # train
    index=list(range(n_sample))

    for ep in range(epoch):
        #print(f'\nepoch: {ep}')
        shuffle(index)
        loss=[]
        for i in range(0,n_sample,min_batch):
        # foreward
            # layer1
            #input=np.append(X[index[i],:],-1 )# add bias
            input=X[index[i],:]
            t_hidden_1=np.dot(input,w_hidden_1)
            o_hidden_1=tanh(t_hidden_1)

            # layer2
            t_hidden_2=np.dot(o_hidden_1,w_hidden_2)
            o_hidden_2=tanh(t_hidden_2)

            # output layer
            t_output=np.dot(o_hidden_2,w_output)
            o_output=tanh(t_output)
            
            error=0.5*(o_output.item(0)-y[index[i]])**2
            loss.append(error)
            #print(y[index[i]],o_output,error)
            

        # backward
            delta_out=(y[index[i]]-o_output)*derivative_tanh(o_output)  # 1x1         
            
            temp_hidden_2=np.dot(delta_out,w_output.T) # shape of w_output
            delta_hidden_2=temp_hidden_2*derivative_tanh(o_hidden_2) 
            #print(delta_hidden_2.shape)
     
            temp_hidden_1=np.dot(delta_hidden_2,w_hidden_2.T) # see delta_hidden_2 as scalar
            delta_hidden_1=temp_hidden_1*derivative_tanh(o_hidden_1)
            #print(delta_hidden_1.shape)


        # update weight            
            for n in range(n_hidden[0]):
                w_hidden_1[:,n]+=learn_rate*delta_hidden_1[n]*input
            
            o_hidden_1=np.array(o_hidden_1).reshape(-1)
            for n in range(n_hidden[1]):
                w_hidden_2[:,n]+=learn_rate*delta_hidden_2[n]*o_hidden_1
            
            o_hidden_2=np.array(o_hidden_2).reshape(-1,1)
            w_output+=learn_rate*(delta_out.item(0)*o_hidden_2)
                
        total_error=np.sum(loss)
        Loss.append(total_error)
        if (ep+1)%100==0:
            print(f'epoch {ep+1} loss: {total_error}')
    
    # show training process
    plt.clf()
    plt.title(f"{filename}_loss")
    plt.plot(range(epoch),Loss,'-')
    plt.savefig(f"./{filename}_loss.png")
    plt.show()

    # predict
    
    t_hidden_1=X*np.mat(w_hidden_1)
    o_hidden_1=tanh(t_hidden_1)
    # layer2
    t_hidden_2=o_hidden_1*w_hidden_2
    o_hidden_2=tanh(t_hidden_2)
    # output layer
    t_output=o_hidden_2*w_output
    o_output=tanh(t_output)

    thres=0.5
    predict_label=o_output.copy()
    predict_label[predict_label>thres]=1
    predict_label[predict_label<=thres]=0

    acc=0
    for i in range(n_sample):
        if y[i]==predict_label[i]:
            acc+=1
        print(y[i],o_output[i],predict_label[i])
    print(f'accuracy: {acc}/{n_sample}, {acc/n_sample*100}%')

    return predict_label

if __name__=="__main__":
    # case 1
    X1,y1=generate_linear(n=100)
    predict_1=linear_back_propogation(X1,y1,learn_rate=0.1,n_hidden=[4,4],filename='test/linear',epoch=1000)
    show_result(X1,y1,predict_1,'test/linear')

    # case 2
    X2,y2=generate_XOR_easy()
    predict_2=linear_back_propogation(X2,y2,learn_rate=0.1,n_hidden=[4,4],filename='test/XOR',epoch=10000)
    show_result(X2,y2,predict_2,'test/XOR')
    #plt.savefig("./data.png")


