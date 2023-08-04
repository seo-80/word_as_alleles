import numpy as np
import matplotlib.pyplot as plt
DATA_COUNT=100

class Learner():
    def __init__(self,data,elements_count=2,param=0.01) -> None:
        self.data_count=1000
        self.elements_count=elements_count
        self.hypothesis=self.learn(data,param)

    def learn(self,data,param):
        tmp=np.array([(x+param/self.elements_count) for x in data])
        return tmp/np.sum(tmp)
    
    def speak(self,data_count):
        return np.random.multinomial(n=data_count,pvals=self.hypothesis)




def one_learner():
    generations_count=10000
    record=np.empty((generations_count,2))
    l=Learner(data=np.ones(2)*50)
    data=l.speak()

    for i in range(generations_count):
        l=Learner(data=data)
        data=l.speak()
        record[i]=l.hypothesis
    plt.plot(record)
    plt.show()

def generate_datas(leaners,data_flow):
    datas=np.zeros((len(leaners),leaners[0].elements_count))

    for i,_ in enumerate(leaners):
        for j,leaner in enumerate(leaners):
            datas[i]+=leaners[j].speak(data_count=data_flow[i][j])
    # print(datas)
    return datas

def generate_data_flow_count(data_flow_rate,total_data_count):
    data_flow_count=np.empty_like(data_flow_rate)
    for i,rate in enumerate(data_flow_rate):
        data_flow_count[i]=np.random.multinomial(n=total_data_count,pvals=rate)
    # print(data_flow_count)
    return data_flow_count

def two_learner(generations_count=10):
    
    data_k=2


    total_data_count=100
    data_flow_rate=np.array([[1,0,0],
                             [0,1,0],
                             [0,0,1]])
    data_flow_rate=np.array([[0.999,0.001],[0.001,0.999]])
    # data_flow_rate=np.array([[1,0],[0,1]])
    # data_flow_rate=np.array([[0.99,0.01],[0.01,0.99]])
    data_flow_rate=np.array([[1]])

    learner_count=len(data_flow_rate)
    record=np.empty((generations_count,learner_count))
    
    data_flow_count=generate_data_flow_count(data_flow_rate,total_data_count)
    learners=[Learner(data=np.ones(data_k)*total_data_count/data_k) for _ in range(learner_count)]
    datas=generate_datas(learners,data_flow_count)
    #print(datas)
    for i in range(generations_count):
        record[i]=[learners[k].hypothesis[0] for  k in range(learner_count)]
        data_flow_count=generate_data_flow_count(data_flow_rate,total_data_count)
        learners=[Learner(data=datas[j]) for j in range(learner_count)]
        datas=generate_datas(learners,data_flow_count)
        # print(data_flow_count)
        # print(datas)
    return record
def main():
    simulation_count=20
    generation_count=200
    records=np.empty((simulation_count,generation_count,1))
    for i in range(simulation_count):
        plt.plot(two_learner(generations_count=generation_count))
    plt.show()

if __name__=="__main__":
    main() 
