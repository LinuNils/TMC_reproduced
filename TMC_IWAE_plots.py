import numpy as np
import matplotlib.pyplot as plt
import csv
plt.rcParams.update({'font.size': 14})
from argparse import ArgumentParser

"""
For the Non-factorized small models
"""
iwae_nf_s_loss = np.load('/Test Results/IWAE_model_non_fac_small_k_20_iwae_obj/tot_obj_loss_k_20_non_fac_small.npy')
tmc_nf_s_loss = np.load('/Test Results/TMC_model_non_fac_small_k_20_objective_TMC/tot_obj_loss_k_20_non_fac_small.npy')
with open('nfs_iwa_std.csv', 'r') as f:
    temp = csv.reader('nfs_iwa_std.csv', delimiter=',')
    original_tmc_nf_s_loss = np.array(temp)

print(original_tmc_nf_s_loss)
mean_iwae_nf_s = np.mean(iwae_nf_s_loss[-50:])
mean_tmc_nf_s = np.mean(tmc_nf_s_loss[-50:])
std_iwae_nf_s = np.std(iwae_nf_s_loss[-50:])
std_tmc_nf_s = np.std(tmc_nf_s_loss[-50:])


print('---------------------------------------------------')
print('Mean loss for the last 50 epochs on IWAE small NF: ', -mean_iwae_nf_s)
print('StdDev for the last 50 epochs on IWAE small NF: ', std_iwae_nf_s)
print('Mean loss for the last 50 epochs on TMC small NF: ', -mean_tmc_nf_s)
print('StdDev for the last 50 epochs on TMC large NF: ', std_tmc_nf_s)
print('---------------------------------------------------')


x_ax_data = np.arange(iwae_nf_s_loss.shape[-1])
fig = plt.figure()
plt.plot(x_ax_data, -iwae_nf_s_loss, x_ax_data, -tmc_nf_s_loss)
plt.legend(['IWAE test loss on IWAE obj.', 'TMC test loss on TMC obj.'], loc='upper right')
plt.xlabel('Epochs')
plt.ylim(-100, -90)
plt.xlim(1, 1200)
plt.ylabel('Objective value')
plt.title('Non-factorized small models evaluated on IWAE/TMC objective')
plt.show()

"""
For the Non-factorized large models
"""
iwae_nf_l_loss = np.load('/Test Results/IWAE_model_non_fac_large_k_20_iwae_obj/tot_obj_loss_k_20_non_fac_large.npy')
tmc_nf_l_loss = np.load('/Test Results/TMC_model_non_fac_large_k_20_objective_TMC/tot_obj_loss_k_20_non_fac_large.npy')
mean_iwae_nf_l = np.mean(iwae_nf_l_loss[-50:])
mean_tmc_nf_l = np.mean(tmc_nf_l_loss[-50:])
std_iwae_nf_l = np.std(iwae_nf_l_loss[-50:])
std_tmc_nf_l = np.std(tmc_nf_l_loss[-50:])

print('---------------------------------------------------')
print('Mean loss for the last 50 epochs on IWAE large NF: ', -mean_iwae_nf_l)
print('StdDev for the last 50 epochs on IWAE large NF: ', std_iwae_nf_l)
print('Mean loss for the last 50 epochs on TMC large NF: ', -mean_tmc_nf_l)
print('StdDev for the last 50 epochs on TMC large NF: ', std_tmc_nf_l)
print('---------------------------------------------------')

fig_2 = plt.figure()
plt.plot(x_ax_data, -iwae_nf_l_loss, x_ax_data, -tmc_nf_l_loss)
plt.legend(['IWAE test loss on IWAE obj.', 'TMC test loss on TMC obj.'], loc='upper right')
plt.xlabel('Epochs')
plt.ylim(-100, -90)
plt.xlim(1, 1200)
plt.ylabel('Objective value')
plt.title('Non-factorized large models evaluated on IWAE/TMC objective')
plt.show()

"""
For the hyper parameter search
"""
# TODO
# TODO Will need several subplots here
# TODO we will have 4 different k's and 5 different learning rates
# TODO so what should we have as structure? 4 subplots with 5 curves in each?


x_hp_data = np.arange(500)
# # for k=1
tmc_1_lr1 = np.load('/Test Results/Hyper parameter search/k_1/TMC_model_hyp_param_srch_non_fac_small_k_1_learn_rate_0.001/TMC_hyp_param_srch_tot_obj_loss_k_1_non_fac_small_lr_0.001.npy')  # lr1 = 1e-3
tmc_1_lr2 = np.load('/Test Results/Hyper parameter search/k_1/TMC_model_hyp_param_srch_non_fac_small_k_1_learn_rate_0.000101/TMC_hyp_param_srch_tot_obj_loss_k_1_non_fac_small_lr_0.000101.npy')  # lr1 = 1e-4
tmc_1_lr3 = np.load('/Test Results/Hyper parameter search/k_1/TMC_model_hyp_param_srch_non_fac_small_k_1_learn_rate_1e-05/TMC_hyp_param_srch_tot_obj_loss_k_1_non_fac_small_lr_1e-05.npy')  # lr1 = 1e-5

fig_3 = plt.figure()
plt.plot(x_hp_data, -tmc_1_lr1, x_hp_data, -tmc_1_lr2, x_hp_data, -tmc_1_lr3)
plt.legend(['lr=1e-3', 'lr=1e-4', 'lr=1e-5'], loc='lower right')
plt.xlabel('Epochs')
plt.ylim(-135, -91)
plt.xlim(1, 500)
plt.ylabel('Objective value')
plt.title('TMC hyper-param search with K=1')
plt.show()

# # for k=5
tmc_5_lr1 = np.load('/Test Results/Hyper parameter search/k_5/TMC_model_hyp_param_srch_non_fac_small_k_5_learn_rate_0.001/TMC_hyp_param_srch_tot_obj_loss_k_5_non_fac_small_lr_0.001.npy')  # lr1 = 1e-3
tmc_5_lr2 = np.load('/Test Results/Hyper parameter search/k_5/TMC_model_hyp_param_srch_non_fac_small_k_5_learn_rate_0.0001/TMC_hyp_param_srch_tot_obj_loss_k_5_non_fac_small_lr_0.0001.npy')  # lr1 = 1e-4
tmc_5_lr3 = np.load('/Test Results/Hyper parameter search/k_5/TMC_model_hyp_param_srch_non_fac_small_k_5_learn_rate_1e-05/TMC_hyp_param_srch_tot_obj_loss_k_5_non_fac_small_lr_1e-05.npy')  # lr1 = 1e-5

fig_4 = plt.figure()
plt.plot(x_hp_data, -tmc_5_lr1, x_hp_data, -tmc_5_lr2, x_hp_data, -tmc_5_lr3)
plt.legend(['lr=1e-3', 'lr=1e-4', 'lr=1e-5'], loc='lower right')
plt.xlabel('Epochs')
plt.ylim(-135, -91)
plt.xlim(1, 500)
plt.ylabel('Objective value')
plt.title('TMC hyper-param search with K=5')
plt.show()

# # for k=20
tmc_20_lr1 = np.load('/Test Results/Hyper parameter search/k_20/TMC_model_hyp_param_srch_non_fac_small_k_20_learn_rate_0.001/TMC_hyp_param_srch_tot_obj_loss_k_20_non_fac_small_lr_0.001.npy')  # lr1 = 1e-3
tmc_20_lr2 = np.load('/Test Results/Hyper parameter search/k_20/TMC_model_hyp_param_srch_non_fac_small_k_20_learn_rate_0.0001/TMC_hyp_param_srch_tot_obj_loss_k_20_non_fac_small_lr_0.0001.npy')  # lr1 = 1e-4
tmc_20_lr3 = np.load('/Test Results/Hyper parameter search/k_20/TMC_model_hyp_param_srch_non_fac_small_k_20_learn_rate_1e-05/TMC_hyp_param_srch_tot_obj_loss_k_20_non_fac_small_lr_1e-05.npy')  # lr1 = 1e-5

fig_5 = plt.figure()
plt.plot(x_hp_data, -tmc_20_lr1, x_hp_data, -tmc_20_lr2, x_hp_data, -tmc_20_lr3)
plt.legend(['lr=1e-3', 'lr=1e-4', 'lr=1e-5'], loc='lower right')
plt.xlabel('Epochs')
plt.ylim(-135, -91)
plt.xlim(1, 500)
plt.ylabel('Objective value')
plt.title('TMC hyper-param search with K=20')
plt.show()

# for k=50
tmc_50_lr1 = np.load('/Test Results/Hyper parameter search/k_50/TMC_model_hyp_param_srch_non_fac_small_k_50_learn_rate_0.001/TMC_hyp_param_srch_tot_obj_loss_k_50_non_fac_small_lr_0.001.npy')  # lr1 = 1e-3
tmc_50_lr2 = np.load('/Test Results/Hyper parameter search/k_50/TMC_model_hyp_param_srch_non_fac_small_k_50_learn_rate_0.0001/TMC_hyp_param_srch_tot_obj_loss_k_50_non_fac_small_lr_0.0001.npy')  # lr1 = 1e-4
tmc_50_lr3 = np.load('/Test Results/Hyper parameter search/k_50/TMC_model_hyp_param_srch_non_fac_small_k_50_learn_rate_1e-05/TMC_hyp_param_srch_tot_obj_loss_k_50_non_fac_small_lr_1e-05.npy')  # lr1 = 1e-5

fig_6 = plt.figure()
plt.plot(x_hp_data, -tmc_50_lr1, x_hp_data, -tmc_50_lr2, x_hp_data, -tmc_50_lr3)
plt.legend(['lr=1e-3', 'lr=1e-4', 'lr=1e-5'], loc='lower right')
plt.xlabel('Epochs')
plt.ylim(-135, -91)
plt.xlim(1, 500)
plt.ylabel('Objective value')
plt.title('TMC hyper-param search with K=50')
plt.show()

# put all HP_searches within one plot
# figure_7 = plt.figure()
# plt.plot(x_hp_data, -tmc_1_lr1, color='red')
# plt.plot(x_hp_data, -tmc_1_lr2, marker='^', color='red')
# plt.plot(x_hp_data, -tmc_1_lr3, marker='s', color='red')
# plt.plot(x_hp_data, -tmc_5_lr1, color='blue')
# plt.plot(x_hp_data, -tmc_5_lr2, marker='^', color='blue')
# plt.plot(x_hp_data, -tmc_5_lr3, marker='s', color='blue')
# plt.plot(x_hp_data, -tmc_20_lr1, color='green')
# plt.plot(x_hp_data, -tmc_20_lr2, marker='^', color='green')
# plt.plot(x_hp_data, -tmc_20_lr3, marker='s', color='green')
# plt.plot(x_hp_data, -tmc_20_lr1, color='black')
# plt.plot(x_hp_data, -tmc_20_lr2, marker='^', color='black')
# plt.plot(x_hp_data, -tmc_20_lr3, marker='s', color='black')
#
# plt.ylabel('Objective value')
# plt.xlabel('Epochs')
# plt.ylim(-135, -91)
# plt.xlim(1, 500)
# print(tmc_20_lr1)
# plt.legend(['K=1 lr=1e-3', 'K=1 lr=1e-4', 'K=1 lr=1e-5',
#             'K=5 lr=1e-3', 'K=5 lr=1e-4', 'K=5 lr=1e-5',
#             'K=20 lr=1e-3', 'K=20 lr=1e-4', 'K=20 lr=1e-5',
#             'K=50 lr=1e-3', 'K=50 lr=1e-4', 'K=50 lr=1e-5'],
#            bbox_to_anchor=(1, 1.05), loc="upper left")
# plt.show()
# for te case of using vanilla IWAE structure and TMC with vanilla IWAE structure
vanilla_iwae = np.load('/Test Results/Vanilla_IWAE_model_non_fac_small_k_5_iwae_obj/tot_obj_loss_k_5_non_fac_small_vanilla_IWAE.npy')  # IWAE with the structure proposed by the IWAE authors
tmc_vanilla_iwae = np.load('/Test Results/TMC_model_with_Vanilla_IWAE_structure_non_fac_small_k_5_TMC_obj/tot_obj_loss_k_5_non_fac_small.npy')  # TMC with IWAE structure proposed by the IWAE authors
x_data = np.arange(vanilla_iwae.shape[-1])
fig_8 = plt.figure()
plt.plot(x_data, -vanilla_iwae, x_data, -tmc_vanilla_iwae)
plt.legend(['Vanilla IWAE', 'TMC with vanilla structure'], loc='lower right')
plt.xlabel('Epochs')
plt.ylim(-140, -115)
plt.xlim(1, 1200)
plt.ylabel('Objective value')
plt.title('Ablation study')
plt.show()




