import numpy as np
import matplotlib.pyplot as plt
from os.path import dirname
import time

DATA_COUNT=100
DATA_PATH=dirname(__file__)+"/../data/"

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

def generate_data_flow_count(data_flow_rate,total_data_count,datacounts=None):
    if np.all(datacounts==None):
        datacounts=[total_data_count for _ in data_flow_rate]
    data_flow_count=np.empty_like(data_flow_rate)
    for i,rate in enumerate(data_flow_rate):
        data_flow_count[i]=np.random.multinomial(n=datacounts[i],pvals=rate)
    # print(data_flow_count)
    return data_flow_count

def two_learner(data_flow_rate,generations_count=10):
    
    data_k=2


    total_data_count=100
    data_flow_rate=np.array([[1,0,0],
                             [0,1,0],
                             [0,0,1]])
    #data_flow_rate=np.array([[0.999,0.001],[0.001,0.999]])
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
def simulate_many_oneleaner():
    simulation_count=20
    generation_count=200
    records=np.empty((simulation_count,generation_count,1))
    for i in range(simulation_count):

        plt.plot(two_learner(generations_count=generation_count))
    plt.show()
def generate_data_flow_rate(leaner_count,exchange_rate,graphtype="complete",bloadcast_infuluence=0):
    data_flow_rate=np.identity((leaner_count))
    if "complete" in graphtype:
        for row in range(leaner_count):
            for columun in range(leaner_count):
                if row==columun:
                    data_flow_rate[(row,columun)]-=exchange_rate
                else:
                    data_flow_rate[(row,columun)]+=exchange_rate/(leaner_count-1)
    if "chain" in graphtype:
        for row in range(leaner_count):
            for columun in range(leaner_count):
                if row==columun:
                    data_flow_rate[(row,columun)]-=exchange_rate
                if abs(row-columun)==1:
                    if (row==0 and columun==1)or(columun==leaner_count-2 and row==leaner_count-1 ):
                        data_flow_rate[(row,columun)]+=exchange_rate
                    else: 
                        data_flow_rate[(row,columun)]+=exchange_rate/2
    if "bloadcast" in graphtype:
        for li in range(leaner_count):
            if li>0:
                data_flow_rate[li][0]+=bloadcast_infuluence
                data_flow_rate[li][li]-=bloadcast_infuluence


                
    return data_flow_rate
def generate_one_way_data_flow_rate(exchange_rate,leaner_count):
    ret=np.zeros((leaner_count,leaner_count))
    for l in range(leaner_count):
        for s in range(leaner_count):
            if l==s:
                ret[l][s]=1-exchange_rate
            elif l-1==s:
                ret[l][s]=exchange_rate
    ret[0][0]=1
    return ret


def exchangelate_difference():
    learner_count=2
    simulation_count=10000
    generation_count=10000200
    waste_generation_count=200
    total_data_count=100
    data_k=2
    exchange_rates=np.arange(start=0,stop=0.01,step=0.0005)
    result=np.empty_like(exchange_rates)
    for index,exchange_rate in enumerate(exchange_rates):

        data_flow_rate=generate_data_flow_rate(learner_count,exchange_rate)
    
        data_flow_count=generate_data_flow_count(data_flow_rate,total_data_count)
        learners=[Learner(data=np.ones(data_k)*total_data_count/data_k) for _ in range(learner_count)]
        datas=generate_datas(learners,data_flow_count)
        record=np.empty((generation_count,learner_count))
        for i in range(generation_count):
            record[i]=[learners[k].hypothesis[0] for  k in range(learner_count)]
            data_flow_count=generate_data_flow_count(data_flow_rate,total_data_count)
            learners=[Learner(data=datas[j]) for j in range(learner_count)]
            datas=generate_datas(learners,data_flow_count)
        result[index]=np.mean(abs(record[:,0][waste_generation_count:]-record[:,1][waste_generation_count:]))
    plt.plot(exchange_rates,result)
    plt.show()
def savetocsv(data,name="data"):
    np.savetxt(DATA_PATH+name+".csv",data,fmt="%d")


def  simulate_fixtime(
    alpha=0.0,
    learner_counts=[2],
    simulation_count=1000,
    total_data_count=100,
    plot_max=1000,
    data_k=2,
    totaldatacounts=None,
    exchange_rates=np.array([0.01]),
    rural_s=np.array([100]),
    rural_ss=None,
    stop_conditions=None,
    fixed_dataflow=None
):
    print("simulating")
    # time=time.getime()
    max_learner_count=max(learner_counts)
    result_time=np.empty((len(learner_counts),len(rural_s),len(exchange_rates),max_learner_count,data_k,simulation_count),dtype=int)
    result_count=np.zeros((len(learner_counts),len(rural_s),len(exchange_rates),max_learner_count,data_k),dtype=int)
    result_count_test=np.zeros(len(rural_s),dtype=int)
    for lcindex,learner_count in enumerate(learner_counts):
        print("learner_count:",learner_count)
        alphas=[0.]
        # if stop_conditions==None:
        stop_conditions=np.zeros((learner_count,data_k),dtype=bool)
        for li in range(1,learner_count):
            stop_conditions[li][1]=True
        for rindex,rs in enumerate(rural_s):
            print("rural s:",rs)
            totaldatacounts=np.array([1]+[rs for _ in range(learner_count-1)])
            #totaldatacounts=np.array([1]+[rs for _ in range(learner_count-2)]+[100])#二つ目のruralだけ値を変更
            print("total_data_count",totaldatacounts)
            
            for erindex,exchange_rate in enumerate(exchange_rates):
                if not fixed_dataflow==None:
                    exchange_rate=fixed_dataflow/rs
                print("exchange rate:",exchange_rate)
                data_flow_rate=generate_one_way_data_flow_rate(exchange_rate,learner_count)
            
                learners=[Learner(data=np.ones(data_k)*total_data_count/data_k,param=alpha) for li in range(learner_count)]
                record=np.empty((10000,learner_count))

                fix_counts=np.zeros((learner_count,data_k),np.int32)
                fix_times=np.empty((learner_count,data_k,simulation_count),dtype=np.int32)
                counts=[0,0]
                for simi in range(simulation_count):
                    stop_conditions_achieved=np.zeros_like(stop_conditions,dtype=bool)          
                    data_flow_count=generate_data_flow_count(data_flow_rate,total_data_count,datacounts=totaldatacounts)
                    # datas=np.array([[0,total_data_count]+[[total_data_count-1,1]for _ in range(learner_count-1)]])
                    datas=np.array([[0,total_data_count]]+[[total_data_count-1,1] for _ in range(learner_count-1)])
                    i=0
                    stop_flag=False
                    while True:
                        data_flow_count=generate_data_flow_count(data_flow_rate,total_data_count,datacounts=totaldatacounts)
                        learners=[Learner(data=datas[j]) for j in range(learner_count)]
                        datas=generate_datas(learners,data_flow_count)
                        i+=1
                        # print(datas[2][1])
                        
                        for li,lc in enumerate(stop_conditions):
                            for di , c in enumerate(lc):
                                if c:
                                    if not stop_conditions_achieved[li][di]:
                                        if datas[li][di]==totaldatacounts[li]:
                                            # print(result_time.shape)
                                            # print(lcindex,rindex,erindex,li,di,result_count[rindex][erindex][li][di])
                                            print(result_count.shape)
                                            print(rindex,erindex,li,di)
                                            result_time[lcindex][rindex][erindex][li][di][result_count[lcindex][rindex][erindex][li][di]]=i
                                            result_count[lcindex][rindex][erindex][li][di]+=1
                                            stop_conditions_achieved[li][di]=True

                                            if np.all(stop_conditions==stop_conditions_achieved):
                                                stop_flag=True
                                        # elif li>0:
                                        #     if datas[li][di]==rs:
                                        #         result_time[rindex][erindex][li][di][result_count[rindex][erindex][li][di]]=i
                                        #         result_count[rindex][erindex][li][di]+=1
                                        #         result_count_test[rindex]+=1
                                        #         stop_flag=True
                        if stop_flag:
                            break



                        # if datas[1][0]==totaldatacounts[1]:
                        #     fix_times[1][0][fix_counts[1][0]]=i
                        #     fix_counts[1][0]+=1
                        #     counts[0]+=1
                        #     break
                        # if datas[1][1]==totaldatacounts[1]:
                        #     fix_times[1][1][fix_counts[1][1]]=i
                        #     fix_counts[1][1]+=1
                        #     counts[1]+=1
                        #     break
        return result_count,result_time,result_count_test
        fix_times_list=[[0 for _ in range(data_k)] for _ in range(learner_count)]
        for li in range(learner_count):
            for di in range(data_k):
                fix_times_list[li][di]=(fix_times[li][di][:fix_counts[li][di]])
        print(fix_counts[1][1])
        print(np.average(fix_times_list[1][1]))
        # plot_max=int(np.max(fix_times_list[1][1]))
        barplot_datas=[[np.zeros(plot_max) for _ in range(data_k)]for _ in range(learner_count)]
        for li in range(learner_count):
            for di in range(data_k):
                for time in fix_times_list[li][di]:
                    if time>=plot_max:
                        print("too large value to plot")
                        continue
                    barplot_datas[li][di][int(time)]+=1
        print(fix_times_list[1][1])
        print(counts)
        savetocsv(fix_times_list[1][1],"param0_oneway_1_times")
        savetocsv(fix_times_list[1][0],"param0_oneway_0_times")
        index=np.arange(plot_max)
        barwidth=1/2
        plt.bar(index,barplot_datas[1][1],width=barwidth)
        plt.bar(index+barwidth,barplot_datas[1][0],width=barwidth)
        plt.show()

def simulate_diff(
    alphas=None,
    learner_count=4,
    # simulation_count=100,
    simulation_count=100000000,
    total_data_count=100,
    plot_max=1000,
    data_k=2,
    total_data_counts=None,
    data_flow_rate=None,
    rural_s=np.array([100]),
    rural_ss=None,
    stop_conditions=None,
    fixed_dataflow=None,
    waste_generation_count=1000,
    randomize_initial_value=True
):
    print("simulation count :",simulation_count)
    # time=time.getime()
    diffs=np.empty((simulation_count,learner_count,learner_count),dtype=float)
    processing_count=simulation_count/10
    if np.all(alphas==None):
        alphas=np.zeros(learner_count)
    if np.all(total_data_counts==None):
        total_data_counts=np.array([total_data_count for _ in range(learner_count)])
    if np.all(data_flow_rate==None):
        data_flow_rate=np.identity(learner_count)   
    debug_data=np.empty((learner_count,simulation_count))    
    datas=np.array([[total_data_counts[i]/data_k for _ in range(data_k)]for i in range(learner_count)])
    for simi in range(simulation_count):
        if simi%processing_count==0:
            pass
            # print(simi/processing_count,time.getime()-time)
        learners=[Learner(param=alphas[i],data=datas[i]) for i in range(learner_count)]
        data_flow_count=generate_data_flow_count(data_flow_rate,total_data_count=total_data_count,datacounts=total_data_counts)

        datas=generate_datas(learners,data_flow_count)
        for i in range(learner_count):
            debug_data[i][simi]=datas[i][0]-datas[0][0]
            for j in range(learner_count):
                diffs[simi][i][j]=abs(datas[i][0]-datas[j][0])
    print(np.average(diffs[:waste_generation_count],axis=0))
    # plt.plot(debug_data.T)
    # plt.legend(list(range(learner_count)))
    # plt.show()
def average(count,time):
    retu_ave=np.empty_like(count,dtype=float)
    for lci,lct in enumerate(time):
        for ri, rt in enumerate(lct):
            for eri,ert in enumerate(rt):
                for li,lt in enumerate(ert):
                    for di , t in enumerate(lt):
                        if count[lci][ri][eri][li][di]==0:
                            retu_ave[lci][ri][eri][li][di]=0
                        else:
                            retu_ave[lci][ri][eri][li][di]=np.average(t[:count[lci][ri][eri][li][di]])
    return retu_ave
def temp():
    rural_s=np.arange(start=1,stop=101,step=2)
    rural_s=np.array([30,50,60])
    learner_counts=[2,3,5]
    count,time,test=simulate_fixtime(
        learner_counts=learner_counts,
        simulation_count=50,
        rural_s=rural_s,
        exchange_rates=[0.05],
    )

    ave_time=average(count,time)

    # plt.bar(rural_s,ave_time[:,0,1,1],width=1)
    # fixation_time=(ave_time[:,0,1,1]*count[:,0,1,1]+ave_time[:,0,1,0]*count[:,0,1,0])/(count[:,0,1,1]+count[:,0,1,0])
    # fixation_time=[(ave_time[i,0,1,1]*count[i,0,1,1]+ave_time[i,0,1,0]*count[i,0,1,0])/(count[i,0,1,1]+count[i,0,1,0]) for i in range(len(rural_s))]

    # print([count[i][0][1][1] for i in range(len(count))])
    print()
    print(test)
    print("fixtime",ave_time)
    print("fixcount",count)
    # for li in range(learner_count-1,0,-1):
    #     plt.bar(rural_s,ave_time[:,0,li,1],alpha=0.2,width=2)
    for lci in range(len(learner_counts)):
        pass
    print(ave_time.shape)
    last_leaner_fixtime=[ave_time[i,0,0,learner_counts[i]-1,1] for i in range(len(learner_counts))]
    plt.bar(learner_counts,last_leaner_fixtime)
    plt.show()
if __name__=="__main__":
    learner_count=4
    alphas=np.array([0.01 for _ in range(learner_count)])
    alphas=np.array([0.01  ]+[0 for _ in range(learner_count-1)])
    data_flow_rate=generate_data_flow_rate(4,0.001,graphtype="chain")
    data_flow_rate=generate_data_flow_rate(4,0.001,graphtype="chain-bloadcast",bloadcast_infuluence=0.0005)
    # data_flow_rate=np.array([
    #     [0.995,0.005,0.00,0.00],
    #     [0.01,0.985,0.005,0.00],
    #     [0.005,0.005,0.985,0.005],
    #     [0.005,0.00,0.05,0.99],
    # ])
    print(alphas)
    print(data_flow_rate)
    simulate_diff(
        learner_count=learner_count,
        alphas=alphas,
        data_flow_rate=data_flow_rate,
    )

