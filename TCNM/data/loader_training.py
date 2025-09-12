from torch.utils.data import DataLoader

# from sgan.data.trajectories import TrajectoryDataset, seq_collate
# from sgan.data.trajectoriesWithMe import TrajectoryDataset, seq_collate
from TCNM.data.trajectoriesWithMe_unet_training import TrajectoryDataset, seq_collate

def data_loader(args, path,test=None,test_year=None):
    dset = TrajectoryDataset(
        path,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        skip=args.skip,
        delim=args.delim,
        other_modal = args.other_modal,
        areas=args.areas,
        test_year=test_year)

    if test is None:
        shuffle = True

    else:
        shuffle = False
        args.loader_num_workers = 0
        
    loader = DataLoader(
        dset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.loader_num_workers,
        collate_fn=seq_collate)
    return dset, loader
