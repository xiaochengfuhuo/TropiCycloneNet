import torch
from torch import nn

# self.wind_embed = nn.Linear(1,16)
#         self.intencity_class_embed = nn.Linear(6,16)
#         self.move_velocity_embed = nn.Linear(1,16)
#         self.month_embed = nn.Linear(12, 16)
#         self.long_embed = nn.Linear(12,16)
#         self.lat_embed = nn.Linear(6,16)
#         self.history_d_12_embed = nn.Linear(8,16)
#         self.history_d_24_embed = nn.Linear(8, 16)
#         self.history_i_24_embed = nn.Linear(4, 16)

class Env_net(nn.Module):
    def __init__(self,obs_len=8):
        super(Env_net, self).__init__()

        embed_dim = 16  # Embedding dimension for input features
        self.data_embed = nn.ModuleDict()

        # Define embedding layers for different input features
        self.data_embed['wind'] = nn.Linear(1, embed_dim)
        self.data_embed['intensity_class'] = nn.Linear(6, embed_dim)
        self.data_embed['move_velocity'] = nn.Linear(1, embed_dim)
        self.data_embed['month'] = nn.Linear(12, embed_dim)
        self.data_embed['location_long'] = nn.Linear(36, embed_dim)
        self.data_embed['location_lat'] = nn.Linear(12, embed_dim)
        self.data_embed['history_direction12'] = nn.Linear(8, embed_dim)
        self.data_embed['history_direction24'] = nn.Linear(8, embed_dim)
        self.data_embed['history_inte_change24'] = nn.Linear(4, embed_dim)

        # Embedding layer for Geopotential Height (GPH) data
        self.GPH_embed = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=True),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(inplace=True),
            nn.AvgPool2d(8,8)
        )

        # Feature extraction network
        env_f_in = len(self.data_embed)*16+8*8
        self.evn_extract = nn.Sequential(
            nn.Linear(env_f_in,env_f_in//2),
            nn.ReLU(),
            nn.Linear(env_f_in//2, env_f_in // 2),
            nn.ReLU(),
            nn.Linear(env_f_in // 2, 64)
        )
        # self.time_weight_emb = nn.Linear(obs_len*32,obs_len)
        # self.softmax = nn.Softmax(dim=1)
        # self.encoder = nn.LSTM(
        #     32, 64, 2, dropout=0
        # )
        # Transformer-based encoder for temporal modeling
        encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=4)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # self.feature_fc = nn.Linear(64*obs_len, 64)

        # self.trajectory_fc = nn.Linear(32,8)
        # self.intensity_fc = nn.Linear(32, 4)

        # self.init_weights()

    def init_hidden(self, batch):
        """Initialize hidden states for LSTM (if used)."""
        return (
            torch.zeros(2, batch, 64).cuda(),
            torch.zeros(2, batch, 64).cuda()
        )

    def init_weights(self):
        """Initialize model weights using Kaiming initialization."""
        def init_kaiming(m):
            if type(m) in [nn.Conv2d, nn.ConvTranspose2d]:
                torch.nn.init.kaiming_normal_(m.weight, mode="fan_in")
                m.bias.data.fill_(0.01)

        self.apply(init_kaiming)


    def forward(self,env_data,gph):
        '''

        Forward pass for the model.

        :param env_data: Dictionary containing different input features (batch, obs_len, feature_dim)
        :param gph: Geopotential height data (batch, 1, obs_len, height, width)
        :return: Extracted features
        '''
        gph = gph.permute(0,2,1,3,4)
        batch,pre_len,channel,h,w = gph.shape
        embed_list = []
        gph_list = []
        # Apply embedding layers to each feature in env_data
        for key in self.data_embed:
            now_embed = self.data_embed[key](env_data[key])
            embed_list.append(now_embed)
        # Process each time step in gph data
        for i_len in range(pre_len):
            gph_list.append(self.GPH_embed(gph[:,i_len]).reshape(batch,-1))
        gph_feature = torch.stack(gph_list,dim=1)
        embed_list.append(gph_feature)
        embed = torch.cat(embed_list, dim=2)

        # embed_list.append(self.GPH_embed(gph.reshape(-1, channel, h, w)).reshape(batch, pre_len, -1))
#       batch,env_f_in

        # Extract high-level features
        feature_in = self.evn_extract(embed).permute(1,0,2)
        # time_weight = self.time_weight_emb(feature_in.permute(1,0,2).reshape(batch,-1)) #++
        # time_weight_0_1 = self.softmax(time_weight).unsqueeze(dim=-1)  # batch  obs_len++
        output = self.encoder(feature_in)
        feature = output[-1]
        # Data format: batch, input_size, seq_len
        # LSTM input format: seq_len, batch, input_size
        # state_tuple = self.init_hidden(batch)
        # output, state = self.encoder(feature_in, state_tuple)
        # all_time_feature = torch.sum(time_weight_0_1*output.permute(1,0,2),dim=1) #++
        # feature = self.feature_fc(output.permute(1,0,2).reshape(batch,-1))
        # feature_all = torch.cat([all_time_feature,feature],dim=-1) #++
        # feature = self.feature_fc(feature_all) #++
        # feature = feature+all_time_feature
        # classf_traj = self.trajectory_fc(feature)
        # classf_inte = self.intensity_fc(feature)
        return feature,0,0



if __name__ == '__main__':
    env_data = {}
    env_data['wind'] = torch.randn((4,8,1)).cuda()
    env_data['intensity_class'] = torch.randn((4,8,6)).cuda()
    env_data['move_velocity'] = torch.randn((4,8,1)).cuda()
    env_data['month'] = torch.randn((4,8,12)).cuda()
    env_data['location_long'] = torch.randn((4,8,36)).cuda()
    env_data['location_lat'] = torch.randn((4,8,12)).cuda()
    env_data['history_direction12'] = torch.randn((4,8,8)).cuda()
    env_data['history_direction24'] = torch.randn((4,8,8)).cuda()
    env_data['history_inte_change24'] = torch.randn((4,8,4)).cuda()



    gph = torch.randn((4,1,8,64,64)).cuda()

    env_net = Env_net().cuda()
    f,_,_  =  env_net(env_data,gph)
    print(f.shape)

