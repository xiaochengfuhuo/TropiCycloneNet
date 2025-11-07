import argparse
import os
import torch
import copy
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(),'..')))

from attrdict import AttrDict

from sgan.data.loader import data_loader
# from sgan.models import TrajectoryGenerator
from sgan.models_prior_unet import TrajectoryGenerator
from sgan.losses import displacement_error, final_displacement_error,toNE,trajectory_displacement_error,value_error,trajectory_diff,value_diff
from sgan.utils import relative_to_abs, get_dset_path,dic2cuda


os.environ["CUDA_VISIBLE_DEVICES"] = '0'
areas = ['EP','NA','NI','SI','SP','WP']
# areas = ['WP']
pt_num = '16000'
save_ana = False
test_year = '2023'
# tensor([23.8483, 45.0369, 67.9328, 95.3904]) tensor([[1.3138, 0.7240],
#         [2.0427, 1.1541],
#         [2.5818, 1.4840],
#         [3.1927, 1.8151]])
parser = argparse.ArgumentParser()
parser.add_argument('--model_path',default='model_save/model_name', type=str)
parser.add_argument('--num_samples', default=6, type=int)
parser.add_argument('--dset_type', default='test', type=str)
parser.add_argument('--areas', default=areas, type=str)
parser.add_argument('--TC_data_path', default=r'path\bst_divi10_train_val_test_inlcude15_2023', type=str)


def get_generator(checkpoint):
    args = AttrDict(checkpoint['args'])
    generator = TrajectoryGenerator(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        embedding_dim=args.embedding_dim,
        encoder_h_dim=args.encoder_h_dim_g,
        decoder_h_dim=args.decoder_h_dim_g,
        mlp_dim=args.mlp_dim,
        num_layers=args.num_layers,
        noise_dim=args.noise_dim,
        noise_type=args.noise_type,
        noise_mix_type=args.noise_mix_type,
        pooling_type=args.pooling_type,
        pool_every_timestep=args.pool_every_timestep,
        dropout=args.dropout,
        bottleneck_dim=args.bottleneck_dim,
        neighborhood_size=args.neighborhood_size,
        grid_size=args.grid_size,
        batch_norm=args.batch_norm,
    )
    generator.load_state_dict(checkpoint['g_state'])
    generator.cuda()
    generator.eval()
    return generator


def getmin_helper(error,an,timeanpv, seq_start_end):
    sum_ = 0
    error = torch.stack(error, dim=1)
    an = torch.stack(an, dim=1)
    timeanpv = torch.stack(timeanpv, dim=1)
    minpoint,minpoint_pv = [],[]


    for (start, end) in seq_start_end:
        start = start.item()
        end = end.item()
        _error = error[start:end]
        _error = torch.sum(_error, dim=0)
        _error = torch.argmin(_error)
        # _error = torch.mean(_error)
        minpoint.append(an[start:end,_error.data.cpu()])
        minpoint_pv.append(timeanpv[start:end,_error.data.cpu()])

    minpoint = torch.stack(minpoint,dim=1).squeeze()
    minpoint_pv = torch.stack(minpoint_pv,dim=1).squeeze()
    return {'tr':minpoint,'pv':minpoint_pv}

def evaluate_helper(error, seq_start_end):
    sum_ = 0
    error = torch.stack(error, dim=1)

    for (start, end) in seq_start_end:
        start = start.item()
        end = end.item()
        _error = error[start:end]
        _error = torch.sum(_error, dim=0)
        _error = torch.min(_error)
        # _error = torch.mean(_error)

        sum_ += _error
    return sum_

def ve_evaluate_helper(error, seq_start_end):
    sum_p = 0
    sum_v = 0
    error = torch.stack(error, dim=1)

    for (start, end) in seq_start_end:
        start = start.item()
        end = end.item()
        _error = error[start:end]
        _error = torch.sum(_error, dim=0)
        _error = torch.min(_error,dim=0)
        # _error = _error[0]
        # _error = torch.mean(_error,dim=0)
        sum_p += _error[0][0]
        sum_v += _error[0][1]
    return sum_p,sum_v

def evaluate(args, loader, generator, num_samples,sava_path):
    ade_outer, fde_outer,tde_outer,ve_outer,ana_outer,pv_outer,gt = [], [],[],[],[],[],[]
    total_traj = 0
    with torch.no_grad():
        for batch in loader:

            env_data = dic2cuda(batch[-2])
            batch = [tensor.cuda() for tensor in batch[:-2]]
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
             non_linear_ped, loss_mask, seq_start_end,obs_traj_Me, pred_traj_gt_Me, obs_traj_rel_Me, pred_traj_gt_rel_Me,
             obs_date_mask, pred_date_mask,image_obs,image_pre) = batch

            ade, fde,tde,ve = [], [],[],[]
            analyse,analyse_pv = [],[]
            total_traj += pred_traj_gt.size(1)
            obs_traj = torch.cat([obs_traj, obs_traj_Me], dim=2)
            gt.append(torch.cat([pred_traj_gt.permute(1, 0, 2), pred_traj_gt_Me.permute(1, 0, 2)], dim=2))
            # pred_traj_gt = torch.cat([pred_traj_gt, pred_traj_gt_Me], dim=2)
            obs_traj_rel = torch.cat([obs_traj_rel, obs_traj_rel_Me], dim=2)
            pred_traj_gt_rel = torch.cat([pred_traj_gt_rel, pred_traj_gt_rel_Me], dim=2)


            pred_traj_fake_rel,_,_,_ = generator(
                obs_traj, obs_traj_rel, seq_start_end,image_obs,env_data,
                num_samples=num_samples, all_g_out=False)

            pred_traj_fake_relt = pred_traj_fake_rel
            pred_traj_fake_rel = pred_traj_fake_relt[:,:,:,:2]
            pred_traj_fake_rel_Me = pred_traj_fake_relt[:,:,:,2:]

            # pred_traj_fake_rel 用来预测后12个点与第8点的偏差
            pred_traj_fake = relative_to_abs(
                pred_traj_fake_rel, obs_traj[-1,:,:2]
            )
            pred_traj_fake_rel_Me = relative_to_abs(
                pred_traj_fake_rel_Me, obs_traj_Me[-1]
            )
            # 只看坐标的偏差

            # 函数会改变参数变量
            real_pred_traj_gt,real_pred_traj_gt_Me = toNE(copy.deepcopy(pred_traj_gt),copy.deepcopy(pred_traj_gt_Me))

                # ade.append(displacement_error(
                #     pred_traj_fake, pred_traj_gt, mode='raw'
                # ))
            for sample_i in range(num_samples):
                real_pred_traj_fake, real_pred_traj_fake_Me = toNE(pred_traj_fake[:,sample_i].squeeze(),
                                                                   pred_traj_fake_rel_Me[:,sample_i].squeeze())
                analyse.append(trajectory_diff(
                    real_pred_traj_fake, real_pred_traj_gt, mode='raw'
                ))
                analyse_pv.append(value_diff(
                    real_pred_traj_fake_Me, real_pred_traj_gt_Me, mode='raw'
                ))
                tde.append(trajectory_displacement_error(
                    real_pred_traj_fake, real_pred_traj_gt, mode='raw'
                ))
                ve.append(value_error(
                    real_pred_traj_fake_Me, real_pred_traj_gt_Me, mode='raw'
                ))

                # fde.append(final_displacement_error(
                #     pred_traj_fake[-1], pred_traj_gt[-1], mode='raw'
                # ))
            time_tde_sum = []
            time_ve_sum = []
            time_an_sum = []
            time_anpv_sum = []

            for i in range(args.pred_len):
                timeade = [x[:,i] for x in tde]
                timean = [x[:, i] for x in analyse]
                timeanpv = [x[:, i] for x in analyse_pv]
                out = getmin_helper(timeade,timean,timeanpv, seq_start_end)
                time_an_sum.append(out['tr'])
                time_anpv_sum.append(out['pv'])
            for i in range(args.pred_len):
                timeade = [x[:,i] for x in tde]
                time_tde_sum.append(evaluate_helper(timeade, seq_start_end))

            for i in range(args.pred_len):
                timeade = [x[:,i] for x in ve]
                time_ve_sum.append(ve_evaluate_helper(timeade, seq_start_end))
            # ade_sum = evaluate_helper(ade, seq_start_end)
            # fde_sum = evaluate_helper(fde, seq_start_end)

            tde_outer.append(time_tde_sum)
            ve_outer.append(time_ve_sum)
            time_an_sum = torch.stack(time_an_sum,dim=1)
            ana_outer.append(time_an_sum)
            time_anpv_sum = torch.stack(time_anpv_sum, dim=1)
            pv_outer.append(time_anpv_sum)
            # fde_outer.append(fde_sum)
        # ade_outer = torch.stack(ade_outer, dim=1)cvpr
        if save_ana == True:
            saveana(ana_outer,pv_outer,gt,sava_path)
        tde_outer = torch.tensor(tde_outer)
        ve_outer = torch.tensor(ve_outer)
        ade = torch.sum(tde_outer,dim=0) / (total_traj)
        ve = torch.sum(ve_outer,dim=0) / (total_traj)
        fde = 0
        return ade, ve

import numpy as np
def saveana(ana_outer,ve_outer,gt,sava_path):
    ana_outer = torch.cat(ana_outer, dim=0)
    ana_outer_np = ana_outer.data.cpu().numpy()
    ve_outer = torch.cat(ve_outer, dim=0)
    ve_outer_np = ve_outer.data.cpu().numpy()
    gt = torch.cat(gt, dim=0)
    gt_np = gt.data.cpu().numpy()
    traj_path = os.path.join(sava_path,'trajectory.npy')
    pv_path = os.path.join(sava_path, 'pvdif.npy')
    gt_path = os.path.join(sava_path, 'gt.npy')
    if os.path.exists(traj_path):
        tra = np.load(traj_path)
        np.save(traj_path,(ana_outer_np+tra)/2)
    else:
        np.save(traj_path, ana_outer_np)
    if os.path.exists(pv_path):
        pv = np.load(pv_path)
        np.save(pv_path, (ve_outer_np+pv)/2)
    else:
        np.save(pv_path, ve_outer_np)
    np.save(gt_path, gt_np)
# def saveana(ana_outer,ve_outer,gt):
#     ana_outer = torch.cat(ana_outer, dim=0)
#     ana_outer_np = ana_outer.data.cpu().numpy()
#     ve_outer = torch.cat(ve_outer, dim=0)
#     ve_outer_np = ve_outer.data.cpu().numpy()
#     gt = torch.cat(gt, dim=0)
#     gt_np = gt.data.cpu().numpy()
#     if os.path.exists('trajectory.npy'):
#         tra = np.load('trajectory.npy')
#         np.save('trajectory.npy',(ana_outer_np+tra)/2)
#     else:
#         np.save('trajectory.npy', ana_outer_np)
#     if os.path.exists('pvdif.npy'):
#         pv = np.load('pvdif.npy')
#         np.save('pvdif.npy', (ve_outer_np+pv)/2)
#     else:
#         np.save('pvdif.npy', ve_outer_np)
#     np.save('gt.npy', gt_np)
#
#     pass

def print_log(modelpath,tde,ve,_args,f_path,mode):
    f = open(f_path,mode)
    print(os.path.split(modelpath)[1])
    print('Dataset: {}, Pred Len: {}'.format(
        _args.dataset_root, _args.pred_len))
    print('TDR:', tde)
    print('TDR:', ve)
    print(os.path.split(modelpath)[1],file=f)
    print('Dataset: {}, Pred Len: {}'.format(
        _args.dataset_root, _args.pred_len),file=f)
    print('TDR:', tde,file=f)
    print('TDR:', ve,file=f)
    f.close()

def main(args):
    sava_path = args.model_path
    areas_str = ''

    for a in areas:
        areas_str = areas_str+a

    log_file = os.path.join(sava_path,'result_log_'+areas_str+'_'+args.dset_type+'_'+pt_num+'.txt')
    best_file = os.path.join(sava_path,'result_best_'+areas_str+'_'+args.dset_type+'_'+pt_num+'.txt')
    min_error = 10000000
    if os.path.isdir(args.model_path):
        filenames = os.listdir(args.model_path)
        filenames.sort()
        paths = [
            os.path.join(args.model_path, file_) for file_ in filenames
        ]
    else:
        paths = [args.model_path]

    for path in paths:


        if 'no_' in path or 'pt' not in path or pt_num not in path:
            continue
        modelpath = path
        checkpoint = torch.load(path)
        # print(checkpoint['args'])
        generator = get_generator(checkpoint)
        _args = AttrDict(checkpoint['args'])
        path = get_dset_path(_args.dataset_root, args.dset_type)
        _args.areas = args.areas
        _, loader = data_loader(_args, {'root':args.TC_data_path,'type':'test'},test_year=test_year)
        tde, ve = evaluate(_args, loader, generator, args.num_samples,sava_path)
        print_log(modelpath,tde,ve,_args,log_file,'a')
        if torch.mean(tde).item() < min_error:
            min_error =torch.mean(tde).item()
            print('now_best==================:')
            print_log(modelpath, tde, ve, _args, best_file,'w')


    return tde,ve


def seed_torch():
    seed = 1024 # 用户设定
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    seed_torch()
    args = parser.parse_args()
    num = 1
    for i in range(num):
        ted,ve = main(args)
        if i == 0:
            ated, ave = ted,ve
        else:
            ated += ted
            ave+=ve
    print(ated/num,ave/num)

# tensor([ 26.5824,  55.9043,  90.2042, 131.4698, 176.3860, 223.9666, 275.9354,
#         330.8814]) tensor([[2.0437, 1.0676],
#         [3.4686, 1.7413],
#         [4.6360, 2.3311],
#         [5.6804, 2.9030],
#         [6.5302, 3.3760],
#         [7.2055, 3.7640],
#         [7.6929, 4.0698],
#         [8.0324, 4.2986]])

# TDR: tensor([24.1299, 47.1854, 70.9537, 99.6390])
# TDR: tensor([[1.4675, 0.8048],
#         [2.3847, 1.2670],
#         [3.1876, 1.7535],
#         [3.8575, 2.2104]])

# checkpoint_with_model_5100.pt
# Dataset: 1950_2019, Pred Len: 4
# TDR: tensor([ 23.9360,  47.1055,  72.3829, 103.1691])
# TDR: tensor([[1.4391, 0.7826],
#         [2.3931, 1.3135],
#         [3.1570, 1.7947],
#         [3.8620, 2.2161]])
