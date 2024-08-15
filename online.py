import json

import numpy as np

from Simulation_parameter import sim_dict
from WITTLE_INDEX_CLASS import  MDP,W_fair_drop,wittle_index, W_fair_ecn_drop
from simulation_zip import queue_simulation
from configure_API import CONFIGURE
import time
from heatmap.WITTLE_HEATMAP import DATA
from Simulation_parameter import parameter
# 在线学习部分：
#     1.观察时间T,采集样本到MDP_MODEL的经验 POOL中
#     2.采集样本，并使用MDP_MODEL的exp_to_ptran计算PTRAN(转移概率矩阵)
#     3.创建数组R0 R1,对应动作0 1,循环状态空间vs,使用MDP_MODEL的函数s_to_u_qlen将状态空间映射的公平性参数u和队列长度qlen
#     4.根据公平性参数u和队列长度qlen，使用REWARD_MODEL的Wreward函数计算出RO R1
#     5.得到RO R1 PTRAN(转移概率矩阵)后，调用WITTLE_MODEL计算出WITTLE值
#     6.周期性▲t(建议5min)循环步骤2-5

#===========自定义数据=========================
sim_pfile = "paper4_1_2.json"
p = parameter(sim_josn=sim_pfile)
configure = CONFIGURE(pool_size=p.pool_size)
REWARD_MODEL =  W_fair_ecn_drop(wf=p.wf,we=p.we,wd=p.wd,queue_size=p.queue_size,u_unit=p.u_unit)
MDP_MODEL = MDP(p.queue_size,p.u_unit,drop_size=p.drop_size,R_class=REWARD_MODEL)
MDP_MODEL.Reward_matrix()#提前算出奖励矩阵

WITTLE_MODEL = wittle_index((p.queue_size+1+p.drop_size)*p.fair)
#==============端口模拟创建========================
simulation  = queue_simulation(p.queue_size,p.u_unit,p.Scheduling_algorithm,p.Congestion_handling,p.pcome,burst=p.bstart_tim,burst_version=p.burst_version)
#==============端口注册，分配对应经验池=============
configure.registration(simulation)


data = np.array(DATA).reshape(-1)
print(data)
for q in range(8):
    configure.WITTLE_UPDATE(data,simulation.port_index,q)
start = time.time()
for tim in range(p.total_time):
    print("tim:"+str(tim))
    simulation.run(tim)
    if simulation.UP_LOAD == True:
        configure.Experience_upload(simulation.port_index)
        simulation.EXP_Clear()
    if tim == 3000000:
        #场景切换
        simulation.pcome =[0.00933082,0.00567012,0.13053439,0.07643199,0.02908332,0.01893222,0.14480572,0.12235003]
    if tim!=0 and tim%p.wittle_update_cycle==0 and 0:
        MDP_MODEL.file_exp_to_ptran(simulation.port_index,q=1)
        data = WITTLE_MODEL.calculate_WITTLE(MDP_MODEL.R[1],MDP_MODEL.R[0],MDP_MODEL.ptran)
        data = data[0:126]
        for q in range(8):
            configure.WITTLE_UPDATE(data, simulation.port_index, q)
simulation.show_performance()
len = 0
drop = 0
ecn = 0
for q in range(8):
    len = len + simulation.performance[f'q{q}']['len']
    drop = drop + simulation.performance[f'q{q}']['drop']
    ecn = ecn+simulation.performance[f'q{q}']['ecn']
print(len,drop,ecn)

    # if tim !=0 and tim%20==0:
    #     for q in range(simulation.priority):
    #         MDP_MODEL.file_exp_to_ptran(simulation.port_index,q)
    #         WI = WITTLE_MODEL.calculate_WITTLE(R1=MDP_MODEL.R[1],R0=MDP_MODEL.R[0],ptran=MDP_MODEL.ptran)
    #         configure.WITTLE_UPDATE(WI,simulation.port_index,q)
end = time.time()
print("simulation_times:{}s".format(end-start))
