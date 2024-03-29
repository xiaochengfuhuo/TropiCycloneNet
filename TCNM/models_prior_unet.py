import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
# from TCNM.Unet3D_test import Unet3D
# from TCNM.Unet3D_merge import Unet3D
# from TCNM.PredRNN import Net
from TCNM.Unet3D_merge_tiny import Unet3D
from TCNM.env_net_transformer_gphsplit import Env_net


def make_mlp(dim_list, activation='relu', batch_norm=True, dropout=0):
    layers = []
    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
        layers.append(nn.Linear(dim_in, dim_out))
        if batch_norm:
            layers.append(nn.BatchNorm1d(dim_out))
        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'leakyrelu':
            layers.append(nn.LeakyReLU())
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
    return nn.Sequential(*layers)


def get_noise(shape, noise_type):
    if noise_type == 'gaussian':
        return torch.randn(*shape).cuda()
    elif noise_type == 'uniform':
        return torch.rand(*shape).sub_(0.5).mul_(2.0).cuda()
    raise ValueError('Unrecognized noise type "%s"' % noise_type)


class Encoder(nn.Module):
    """Encoder is part of both TrajectoryGenerator and
    TrajectoryDiscriminator"""
    def __init__(
        self, embedding_dim=64, h_dim=64, mlp_dim=1024, num_layers=1,
        dropout=0.0
    ):
        super(Encoder, self).__init__()

        self.mlp_dim = 1024
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.encoder = nn.LSTM(
            embedding_dim, h_dim, num_layers, dropout=dropout
        )

        self.spatial_embedding = nn.Linear(4, embedding_dim)
        self.time_embedding = nn.Linear(4,embedding_dim)

    def init_hidden(self, batch):
        return (
            torch.zeros(self.num_layers, batch, self.h_dim).cuda(),
            torch.zeros(self.num_layers, batch, self.h_dim).cuda()
        )

    def forward(self, obs_traj,img_embed_input):
        """
        Inputs:
        - obs_traj: Tensor of shape (obs_len, batch, 2)
        - img_embed_input: [obs_len,b,64]
        Output:
        - final_h: Tensor of shape (self.num_layers, batch, self.h_dim)
        """
        # Encode observed Trajectory
        batch = obs_traj.size(1)
        inputDim = obs_traj.size(2)
        # inputDim_date = obs_date_mask.size(2)
        obs_traj_embedding = self.spatial_embedding(obs_traj.reshape(-1, inputDim))
        obs_traj_embedding = obs_traj_embedding.view(
            -1, batch, self.embedding_dim
        )
        obs_traj_embedding = obs_traj_embedding+img_embed_input
        state_tuple = self.init_hidden(batch)
        output, state = self.encoder(obs_traj_embedding, state_tuple)
        final_h = state
        return {'final_h':final_h,'output':output}


class Decoder(nn.Module):
    """Decoder is part of TrajectoryGenerator"""
    def __init__(
        self, seq_len, embedding_dim=64, h_dim=128, mlp_dim=1024, num_layers=1,
        pool_every_timestep=True, dropout=0.0, bottleneck_dim=1024,
        activation='relu', batch_norm=True, pooling_type='pool_net',
        neighborhood_size=2.0, grid_size=8,embeddings_dim=128,
            h_dims=128,
    ):
        super(Decoder, self).__init__()

        self.seq_len = seq_len
        self.mlp_dim = mlp_dim
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        self.pool_every_timestep = pool_every_timestep

        self.decoder = nn.LSTM(
            embedding_dim, h_dim, num_layers, dropout=dropout
        )

        if pool_every_timestep:
            if pooling_type == 'pool_net':
                pass
            elif pooling_type == 'spool':
                pass

            mlp_dims = [h_dim + bottleneck_dim, mlp_dim, h_dim]
            self.mlp = make_mlp(
                mlp_dims,
                activation=activation,
                batch_norm=batch_norm,
                dropout=dropout
            )


        self.spatial_embedding = nn.Linear(4, embedding_dim)
        self.time_embedding = nn.Linear(4, embedding_dim)
        self.hidden2pos = nn.Linear(h_dim, 4)
    def init_hidden(self, batch):
        return (
            torch.zeros(self.num_layers, batch, self.h_dim).cuda(),
            torch.zeros(self.num_layers, batch, self.h_dim).cuda()
        )
    def forward(self, obs_traj, obs_traj_rel, last_pos, last_pos_rel, state_tuple,
                seq_start_end,decoder_img,last_img):
        """
        Inputs:
        - last_pos: Tensor of shape (batch, 2)
        - last_pos_rel: Tensor of shape (batch, 2)
        - state_tuple: (hh, ch) each tensor of shape (num_layers, batch, h_dim)
        - seq_start_end: A list of tuples which delimit sequences within batch
        - decoder_img [len,batch,64]
        - last_img [batch,64]
        Output:
        - pred_traj: tensor of shape (self.seq_len, batch, 2)

        """
        batch = last_pos.size(0)
        pred_traj_fake_rel = []
        decoder_input = self.spatial_embedding(last_pos_rel)
        decoder_input = decoder_input.view(-1, batch, self.embedding_dim)
        # add img_information
        last_img = last_img.unsqueeze(0)
        decoder_input = decoder_input+last_img

        # obs_traj_rel_new = obs_traj_rel.clone()
        # obs_date_mask_new = obs_date_mask.clone()

        for i_step in range(self.seq_len):
            output, state_tuple = self.decoder(decoder_input, state_tuple)
            rel_pos = self.hidden2pos(output.view(-1, self.h_dim))
            curr_pos = rel_pos + last_pos


            # state_tuple = self.init_hidden(batch)
            rel_pos = rel_pos.unsqueeze(0)
            # obs_traj_rel_new = torch.cat([obs_traj_rel,rel_pos],dim=0)
            # obs_date_mask_new = torch.cat([obs_date_mask_new,pred_date_mask[i_step].unsqueeze(0)],dim=0)
            # state_tuple = self.encoders(obs_traj_rel_new,obs_date_mask_new)
            # state_tuple = state_tuple['final_h']

            # obs_traj_rel = obs_traj_rel_new[1:]
            # 更新obs_traj_rel
            embedding_input = rel_pos

            decoder_input = self.spatial_embedding(embedding_input)
            decoder_input = decoder_input.view(-1, batch, self.embedding_dim)
            # add img_information
            decoder_img_one = decoder_img[i_step].unsqueeze(0)
            decoder_input = decoder_input+decoder_img_one

            pred_traj_fake_rel.append(rel_pos.view(batch, -1))
            last_pos = curr_pos

        pred_traj_fake_rel = torch.stack(pred_traj_fake_rel, dim=0)
        # 预测的路径差值--与第八个的pos相加即能路径
        return pred_traj_fake_rel, state_tuple[0]





class TrajectoryGenerator(nn.Module):
    def __init__(
        self, obs_len, pred_len, embedding_dim=64, encoder_h_dim=64,
        decoder_h_dim=128, mlp_dim=1024, num_layers=1, noise_dim=(0, ),
        noise_type='gaussian', noise_mix_type='ped', pooling_type=None,
        pool_every_timestep=True, dropout=0.0, bottleneck_dim=1024,
        activation='relu', batch_norm=True, neighborhood_size=2.0, grid_size=8,num_gs=6,num_sample=6
    ):
        super(TrajectoryGenerator, self).__init__()

        if pooling_type and pooling_type.lower() == 'none':
            pooling_type = None

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.mlp_dim = mlp_dim
        self.encoder_h_dim = encoder_h_dim
        self.decoder_h_dim = decoder_h_dim
        self.embedding_dim = embedding_dim
        self.noise_dim = noise_dim
        self.num_layers = num_layers
        self.noise_type = noise_type
        self.noise_mix_type = noise_mix_type
        self.pooling_type = pooling_type
        self.noise_first_dim = 0
        self.pool_every_timestep = pool_every_timestep
        self.bottleneck_dim = 1024
        self.num_gs = num_gs
        self.num_sample = num_sample

        self.Unet = Unet3D(1,1)
        # self.predrnn = Net(32, 1, h_w=[50, 50], n_GPU=1)
        self.img_embedding = nn.Linear(64*64,32)
        self.img_embedding_real = nn.Linear(64 * 64, 32)
        self.env_net = Env_net()
        self.env_net_chooser = Env_net()
        self.feature2dech_env = nn.Linear(128,64)
        self.feature2dech = nn.Linear(128, 64)



        self.encoder = Encoder(
            embedding_dim=embedding_dim,
            h_dim=encoder_h_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            dropout=dropout
        )

        self.encoder_env = Encoder(
            embedding_dim=embedding_dim,
            h_dim=encoder_h_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            dropout=dropout
        )

        self.gs = nn.ModuleList()
        # [11-5,6and10,7-9]
        for i in range(num_gs):
            self.gs.append(Decoder(
                pred_len,
                embedding_dim=embedding_dim,
                h_dim=decoder_h_dim,
                mlp_dim=mlp_dim,
                num_layers=num_layers,
                pool_every_timestep=pool_every_timestep,
                dropout=dropout,
                bottleneck_dim=bottleneck_dim,
                activation=activation,
                batch_norm=batch_norm,
                pooling_type=pooling_type,
                grid_size=grid_size,
                neighborhood_size=neighborhood_size,
                embeddings_dim=embedding_dim,
                h_dims=encoder_h_dim,
            ))
        self.net_chooser = nn.Sequential(
            nn.Linear(encoder_h_dim, encoder_h_dim // 2),
            nn.ReLU(),
            nn.Linear(encoder_h_dim // 2, encoder_h_dim // 2),
            nn.ReLU(),
            nn.Linear(encoder_h_dim // 2, num_gs),
        )

        if pooling_type == 'pool_net':
            pass
        elif pooling_type == 'spool':
            pass
        if self.noise_dim[0] == 0:
            self.noise_dim = None
        else:
            self.noise_first_dim = noise_dim[0]

        # Decoder Hidden
        if pooling_type:
            input_dim = encoder_h_dim + bottleneck_dim
        else:
            input_dim = encoder_h_dim

        if self.mlp_decoder_needed():
            mlp_decoder_context_dims = [
                input_dim, mlp_dim, decoder_h_dim - self.noise_first_dim
            ]

            self.mlp_decoder_context = make_mlp(
                mlp_decoder_context_dims,
                activation=activation,
                batch_norm=batch_norm,
                dropout=dropout
            )

    def add_noise(self, _input, seq_start_end, user_noise=None):
        """
        Inputs:
        - _input: Tensor of shape (_, decoder_h_dim - noise_first_dim)
        - seq_start_end: A list of tuples which delimit sequences within batch.
        - user_noise: Generally used for inference when you want to see
        relation between different types of noise and outputs.
        Outputs:
        - decoder_h: Tensor of shape (_, decoder_h_dim)
        """
        if not self.noise_dim:
            return _input

        if self.noise_mix_type == 'global':
            noise_shape = (seq_start_end.size(0), ) + self.noise_dim
        else:
            noise_shape = (_input.size(0), ) + self.noise_dim

        if user_noise is not None:
            z_decoder = user_noise
        else:
            z_decoder = get_noise(noise_shape, self.noise_type)

        if self.noise_mix_type == 'global':
            _list = []
            for idx, (start, end) in enumerate(seq_start_end):
                start = start.item()
                end = end.item()
                _vec = z_decoder[idx].view(1, -1)
                _to_cat = _vec.repeat(end - start, 1)
                _list.append(torch.cat([_input[start:end], _to_cat], dim=1))
            decoder_h = torch.cat(_list, dim=0)
            return decoder_h

        decoder_h = torch.cat([_input, z_decoder], dim=1)

        return decoder_h

    def mlp_decoder_needed(self):
        if (
            self.noise_dim or self.pooling_type or
            self.encoder_h_dim != self.decoder_h_dim
        ):
            return True
        else:
            return False

    def get_samples(self, enc_h, num_samples=6):
        """Returns generator indexes of shape (batch size, num samples)"""

        net_chooser_out = self.net_chooser(enc_h).reshape(-1,self.num_gs)

        dist = Categorical(logits=net_chooser_out)
        sampled_gen_idxs = dist.sample((num_samples,)).transpose(0, 1)
        return net_chooser_out, sampled_gen_idxs.detach().cpu().numpy()

    def mix_noise(self,final_encoder_h,seq_start_end,batch,user_noise=None):
        mlp_decoder_context_input = final_encoder_h.view(
            -1, self.encoder_h_dim)

        # Add Noise
        if self.mlp_decoder_needed():
            noise_input = self.mlp_decoder_context(mlp_decoder_context_input)
        else:
            noise_input = mlp_decoder_context_input
        decoder_h = self.add_noise(
            noise_input, seq_start_end, user_noise=user_noise)
        # decoder_h = torch.unsqueeze(decoder_h, 0)
        decoder_h = decoder_h.view(-1, batch, self.encoder_h_dim)

        decoder_c = torch.zeros(
            self.num_layers, batch, self.decoder_h_dim
        ).cuda()

        state_tuple = (decoder_h, decoder_c)
        return state_tuple

    def forward(self, obs_traj, obs_traj_rel, seq_start_end,image_obs,env_data,
                num_samples=1,all_g_out=False,predrnn_img=None,user_noise=None):
        """
        Inputs:
        - obs_traj: Tensor of shape (obs_len, batch, 2)
        - obs_traj_rel: Tensor of shape (obs_len, batch, 2)
        - seq_start_end: A list of tuples which delimit sequences within batch.
        - user_noise: Generally used for inference when you want to see
        relation between different types of noise and outputs.
        - image_obs: (b,c,obs_len,h,w)
        Output:
        - pred_traj_rel: Tensor of shape (self.pred_len, batch, 2)
        """
        batch = obs_traj_rel.size(1)
        obs_len = obs_traj_rel.size(0)


        # for netchooser
        img_input_real = image_obs.view(batch, self.obs_len, -1)
        encoder_img_real = self.img_embedding_real(img_input_real).permute(1, 0, 2)  # [12,b,64]
        final_encoder_env = self.encoder_env(obs_traj_rel, encoder_img_real)
        final_encoder_env_h = final_encoder_env['final_h'][0]

        evn_feature_chooser,traj_score,inte_score = self.env_net_chooser(env_data,image_obs)
        dec_h_evn = self.feature2dech_env(torch.cat([final_encoder_env_h.squeeze(0),evn_feature_chooser],dim=1)).unsqueeze(0)

        # for generator
        predrnnn_out = self.Unet(image_obs)
        first_img = image_obs[:,:,0].unsqueeze(2)
        all_img = torch.cat([first_img,predrnnn_out],dim=2)
        img_input = all_img.view(batch,self.obs_len+self.pred_len                                 ,-1)
        img_embed_input = self.img_embedding(img_input).permute(1,0,2)  #[12,b,64]
        encoder_img = img_embed_input[:obs_len]
        final_encoder = self.encoder(obs_traj_rel, encoder_img)
        final_encoder_h = final_encoder['final_h'][0]
        #evn_feature, traj_score, inte_score = self.env_net(env_data, image_obs[:, :, -1])
        dec_h = self.feature2dech(torch.cat([final_encoder_h.squeeze(0), evn_feature_chooser], dim=1)).unsqueeze(0)
        # Encode seq

        # output = final_encoder['output']

        image_out = all_img
        # 加个残差
        # final_encoder_h = unet_out['fusion_feature']+final_encoder_h
        # Pool States

        if all_g_out:
            last_pos = obs_traj[-1]
            last_pos_rel = obs_traj_rel[-1]
            decoder_img = img_embed_input[obs_len:]
            last_img = img_embed_input[obs_len - 1]
            preds_rel = []
            with torch.no_grad():
                state_tuple = self.mix_noise(dec_h, seq_start_end, batch)
                for g_i , g in enumerate(self.gs):
                    pred_traj_fake_rel, final_decoder_h = g(
                        obs_traj,
                        obs_traj_rel,
                        last_pos,
                        last_pos_rel,
                        state_tuple,
                        seq_start_end,
                        decoder_img,
                        last_img
                    )
                    preds_rel.append(pred_traj_fake_rel.reshape(self.pred_len, 1, batch, 4))
            pred_traj_fake_rel_nums = torch.cat(preds_rel, dim=1)
            # [prelen,g_num,batch,4]
            net_chooser_out, sampled_gen_idxs = self.get_samples(dec_h_evn.squeeze(), num_samples)

        else:
            with torch.no_grad():
                net_chooser_out, sampled_gen_idxs = self.get_samples(dec_h_evn.squeeze(),num_samples)
            # Predict Trajectory
            preds_rel = []
            for sample_i in range(num_samples):
                pred_traj_fake_rel_reverse = torch.ones((self.pred_len, batch, 4), requires_grad=True).cuda()
                gs_index = sampled_gen_idxs[:,sample_i]
                for g_i in range(np.unique(gs_index).shape[0]):
                    # sampled_gen_idxs   [b,num_samples]
                    now_data_index = gs_index == g_i
                    if np.sum(now_data_index) < 1:
                        continue
                    last_pos = obs_traj[-1, now_data_index]
                    last_pos_rel = obs_traj_rel[-1, now_data_index]
                    decoder_img = img_embed_input[obs_len:, now_data_index]
                    last_img = img_embed_input[obs_len - 1, now_data_index]

                    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    state_tuple = self.mix_noise(dec_h[:,now_data_index], seq_start_end[now_data_index], np.sum(now_data_index))
                    # the method of sampling is not same as the MG-GAN!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    decoder = self.gs[g_i]
                    decoder_out = decoder(
                        obs_traj[:,now_data_index],
                        obs_traj_rel[:,now_data_index],
                        last_pos,
                        last_pos_rel,
                        state_tuple,
                        seq_start_end[now_data_index],
                        decoder_img,
                        last_img
                    )
                    pred_traj_fake_rel, final_decoder_h = decoder_out
                    pred_traj_fake_rel_reverse[:, now_data_index] = pred_traj_fake_rel
                preds_rel.append(pred_traj_fake_rel_reverse.reshape(self.pred_len, 1, batch, 4))
            pred_traj_fake_rel_nums = torch.cat(preds_rel, dim=1)
            # [prelen,num_samples,batch,4]
            # pred_traj_fake_rel_list.append(pred_traj_fake_rel)


        return pred_traj_fake_rel_nums,image_out,net_chooser_out, sampled_gen_idxs


class TrajectoryDiscriminator(nn.Module):
    def __init__(
        self, obs_len, pred_len, embedding_dim=64, h_dim=64, mlp_dim=1024,
        num_layers=1, activation='relu', batch_norm=True, dropout=0.0,
        d_type='local'
    ):
        super(TrajectoryDiscriminator, self).__init__()

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = obs_len + pred_len
        self.mlp_dim = mlp_dim
        self.h_dim = h_dim
        self.d_type = d_type

        self.img_embedding = nn.Linear(64 * 64, 32)
        self.encoder = Encoder(
            embedding_dim=embedding_dim,
            h_dim=h_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            dropout=dropout
        )

        real_classifier_dims = [h_dim, mlp_dim, 1]
        info_classifier_dims = [h_dim, mlp_dim, 2]
        self.real_classifier = make_mlp(
            real_classifier_dims,
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout
        )
        if d_type == 'global':
            mlp_pool_dims = [h_dim + embedding_dim, mlp_dim, h_dim]
            pass

    def forward(self, traj, traj_rel, seq_start_end,img):
        """
        Inputs:
        - traj: Tensor of shape (obs_len + pred_len, batch, 2)
        - traj_rel: Tensor of shape (obs_len + pred_len, batch, 2)
        - seq_start_end: A list of tuples which delimit sequences within batch
        - img [b,c,len,h,w]
        Output:
        - scores: Tensor of shape (batch,) with real/fake scores
        """
        b,c,len,_,_ = img.shape
        img_embed_input = img.view(b,len,-1)
        img_embed = self.img_embedding(img_embed_input).permute(1,0,2)
        final_h = self.encoder(traj_rel,img_embed)
        final_h = final_h['final_h'][0]

        # output = final_encoder['output']
        # Note: In case of 'global' option we are using start_pos as opposed to
        # end_pos. The intution being that hidden state has the whole
        # trajectory and relative postion at the start when combined with
        # trajectory information should help in discriminative behavior.
        if self.d_type == 'local':
            classifier_input = final_h.squeeze()
        else:
            pass
        scores = self.real_classifier(classifier_input)

        return scores,classifier_input
