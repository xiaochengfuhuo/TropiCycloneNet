import argparse
import os
import torch
import copy
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(),'..')))

from attrdict import AttrDict
import matplotlib.image as img
from TCNM.data.loader import data_loader
from TCNM.models_prior_unet import TrajectoryGenerator
from TCNM.losses import toNE
from TCNM.utils import relative_to_abs, get_dset_path,dic2cuda

import matplotlib.pyplot as plt

from scipy.spatial import ConvexHull
import numpy as np



os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# areas = ['EP','NA','NI','SI','SP','WP']
# Define the target area for analysis
areas = ['WP'] # Western Pacific (WP)
pt_num = '16000'
save_ana = False

# Define command-line argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--model_path',default='model_save/best', type=str,help="Path to the trained model")
parser.add_argument('--num_samples', default=6, type=int, help="Number of samples to generate")
parser.add_argument('--dset_type', default='test', type=str, help="Dataset type (train/val/test)")
parser.add_argument('--areas', default=areas, type=str, help="Regions for analysis")
parser.add_argument('--TC_name', default='MALIKSI', type=str, help="Tropical cyclone name")
parser.add_argument('--TC_date', default='2018061006', type=str, help="Tropical cyclone timestamp (YYYYMMDDHH)")
parser.add_argument('--TC_img_path', default='G:/data/TYDataset/TY-Airmass', type=str, help="Path to satellite images")
parser.add_argument('--TC_data_path', default='G:/data/data4Tropicyclone/visualization_data', type=str, help="Path to cyclone trajectory data")

def get_generator(checkpoint):
    '''
    Load the trained model from a checkpoint and return the generator model.
    '''
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


def getPicName(tyid,args):
    '''
    Retrieve the corresponding satellite image file path for the given tropical cyclone
    '''
    try:
        tyname = tyid[0]['new'][1]
        date = tyid[0]['new'][0]
        year = date[:4]
        root = args.TC_img_path
        # datef = date[:-2]+'_'+date[-2:]
        filelist = os.listdir(os.path.join(root,year,tyname))
        for filename in filelist :
            if date in filename and '.xml' not in filename:
                # print(os.path.join(root,year,tyname,filename))
                return os.path.join(root,year,tyname,filename)
    except:
        return 0

def getclosetra(i,gt,pre):

    pres = np.stack(pre)
    pres = pres[:,:,i,:]
    x = pres[:,:,0]-gt[:,0]
    y = pres[:,:,1]-gt[:,1]
    dist = x**2+y**2
    sumdist = np.sum(dist,axis=1)
    mindist = np.min(sumdist)
    mask = (sumdist==mindist)*1
    return mask


def encircle(x,y, ax=None, **kw):
    '''
    Draw a convex hull around the given points on the plot.
    '''
    if not ax: ax=plt.gca()
    p = np.c_[x,y]
    hull = ConvexHull(p)
    poly = plt.Polygon(p[hull.vertices,:], **kw)
    ax.add_patch(poly)


def evaluate(args, loader, generator, num_samples,sava_path,plot_all=False,tc_name='MALIKSI',tc_date='2018061006'):
    '''
    Perform evaluation and visualization of the predicted cyclone trajectories.
    '''
    ade_outer, fde_outer,tde_outer,ve_outer,ana_outer,pv_outer,gt = [], [],[],[],[],[],[]
    total_traj = 0
    generator.eval()
    with torch.no_grad():
        print('Loading model and data...')
        for batch in loader:
            tc_info = batch[-1]
            # print(tc_info)
            # continue

            in_this_batch = False
            ty_target_id = 0
            for sample_id, tc_one_info in enumerate(tc_info):
                if tc_date != tc_one_info[0]['new'][0] or tc_name!=tc_one_info[0]['new'][1]:
                    continue
                else:
                    in_this_batch = True
                    ty_target_id = sample_id
                    break
            if not in_this_batch:
                continue

            print('Predicting...')
            env_data = dic2cuda(batch[-2])
            batch = [tensor.cuda() for tensor in batch[:-2]]
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
             non_linear_ped, loss_mask, seq_start_end,obs_traj_Me, pred_traj_gt_Me, obs_traj_rel_Me, pred_traj_gt_rel_Me,
             obs_date_mask, pred_date_mask,image_obs,image_pre) = batch


            total_traj += pred_traj_gt.size(1)
            obs_traj = torch.cat([obs_traj, obs_traj_Me], dim=2)
            gt.append(torch.cat([pred_traj_gt.permute(1, 0, 2), pred_traj_gt_Me.permute(1, 0, 2)], dim=2))
            # pred_traj_gt = torch.cat([pred_traj_gt, pred_traj_gt_Me], dim=2)

            real_obs_traj_gt, real_obs_traj_gt_Me = toNE(copy.deepcopy(obs_traj[:,:,:2]), copy.deepcopy(obs_traj[:,:,2:]))
            obs_traj_rel = torch.cat([obs_traj_rel, obs_traj_rel_Me], dim=2)
            pred_traj_gt_rel = torch.cat([pred_traj_gt_rel, pred_traj_gt_rel_Me], dim=2)


            pred_traj_fake_rel,_,_,_ = generator(
                obs_traj, obs_traj_rel, seq_start_end,image_obs,env_data,
                num_samples=num_samples, all_g_out=False)

            print('Complete prediction!')
            print('Drawing the prediction!')
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

            real_pred_traj_fake_list = []
            for sample_i in range(num_samples):
                real_pred_traj_fake, real_pred_traj_fake_Me = toNE(copy.deepcopy(pred_traj_fake[:,sample_i]),
                                                                   copy.deepcopy(pred_traj_fake_rel_Me[:,sample_i]))
                real_pred_traj_fake_list.append(real_pred_traj_fake.cpu().numpy())

            # name = tc_info[ty_target_id][0]['new'][1] + str(tc_info[ty_target_id][0]['new'][0])
            aa = torch.cat([real_obs_traj_gt[:,:,:2],real_pred_traj_gt],dim=0).cpu().numpy()
            # if name != 'HALONG2019110800':
            #     continue
            if getPicName(tc_info[ty_target_id],args) == 0:
                print('The background image of this TC is not provided. Please try other TC')
                continue

            bgimg = img.imread(getPicName(tc_info[ty_target_id],args))
            # plotCount += 1
            fig = plt.figure(figsize=(10, 10))
            fig.figimage(bgimg)
            ax = fig.add_axes([0, 0, 1, 1])

            ax.set_xlim(800, 2000)
            ax.set_ylim(-600, 600)
            traplot = [1, 1, 1, 1, 1, 1]
#  aa = real_pred_traj_gt   pred_list == analyse i == ty_target_id
            traplot = getclosetra(ty_target_id, aa[8:, ty_target_id, :], real_pred_traj_fake_list)
            color = ['#38FF39', '#38FF39', '#38FF39', '#38FF39', '#38FF39', '#38FF39']

            all_point = [aa[8, ty_target_id][np.newaxis, :]]
            for j, pred_traj_fakex in enumerate(real_pred_traj_fake_list):
                all_point.append(pred_traj_fakex[:, ty_target_id, :])
                if not plot_all:
                    if traplot[j] == 0:
                        continue
                out_a = pred_traj_fakex[:, ty_target_id, :]
                # bb=np.concatenate((input_a[:, i, :].cuda().data.cpu().numpy(),out_a.cuda().data.cpu().numpy()),axis=0)
                bb = out_a
                # global x1,y1
                x1 = bb[:, 0]
                y1 = bb[:, 1]
                ax.plot(x1, y1, '*', markersize=5, color=color[j])

            # plt.show()
            # ani = animation.FuncAnimation(fig, update_dot, frames = gen_dot, interval = 5)
            ax.plot(aa[:12, ty_target_id, 0], aa[:12, ty_target_id, 1], '.', color='red', markersize=5)
            # 覆盖区域===============
            all_point = np.concatenate(all_point, axis=0)
            encircle(all_point[:, 0], all_point[:, 1], ax, fc='#F52100', alpha=.5)

            ax.set_zorder(100)
            ax.set_axis_off()

            savePath = os.path.join('plot')
            if not os.path.exists(savePath):
                os.makedirs(savePath)
            name = tc_info[ty_target_id][0]['new'][1] + str(tc_info[ty_target_id][0]['new'][0])
            plt.savefig(os.path.join(savePath, name + '.png'))
            print('Completed! Please check the sample at '+os.path.join(savePath, name + '.png'))
            # plt.show()
            plt.close()




def main(args):
    '''
    Main function to load model and dataset, and run evaluation.
    '''
    sava_path = args.model_path
    areas_str = ''

    for a in areas:
        areas_str = areas_str+a

    if os.path.isdir(args.model_path):
        filenames = os.listdir(args.model_path)
        filenames.sort()
        paths = [
            os.path.join(args.model_path, file_) for file_ in filenames
        ]
    else:
        paths = [args.model_path]

    for path in paths:


        if 'no_' in path or 'pt' not in path:
            continue
        checkpoint = torch.load(path)
        generator = get_generator(checkpoint)
        _args = AttrDict(checkpoint['args'])
        _args.areas = args.areas
        _args.TC_img_path = args.TC_img_path
        _args.TC_data_path = args.TC_data_path
        _, loader = data_loader(_args, {'root':args.TC_data_path,'type':'test'},batch_size=_args.batch_size)
        evaluate(_args, loader, generator, args.num_samples,sava_path,tc_name=args.TC_name,tc_date=args.TC_date)



def seed_torch():
    seed = 1024 
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
    main(args)
