from torch import nn
from unetr_pp.network_architecture.neural_network import SegmentationNetwork
from unetr_pp.network_architecture.synapse.nnformer.model_components import Encoder,Decoder,final_patch_expanding, PosEnc

                                         
class nnFormer(SegmentationNetwork):

    def __init__(self, crop_size=[64,128,128],
                embedding_dim=192,
                input_channels=1, 
                num_classes=14, 
                conv_op=nn.Conv3d, 
                depths=[2,2,2,2],
                num_heads=[6, 12, 24, 48],
                patch_size=[2,4,4],
                window_size=[4,4,8,4],
                deep_supervision=True):
      
        super(nnFormer, self).__init__()
        
        self.img_size = crop_size
        
        self._deep_supervision = deep_supervision
        self.do_ds = deep_supervision
        self.num_classes=num_classes
        self.conv_op=conv_op
        
        self.posenc = PosEnc()
        
        embed_dim=embedding_dim
        depths=depths
        num_heads=num_heads
        patch_size=patch_size
        window_size=window_size
        self.model_down=Encoder(pretrain_img_size=crop_size,window_size=window_size,embed_dim=embed_dim,patch_size=patch_size,depths=depths,num_heads=num_heads,in_chans=input_channels)
        self.decoder=Decoder(pretrain_img_size=crop_size,embed_dim=embed_dim,window_size=window_size[::-1][1:],patch_size=patch_size,num_heads=num_heads[::-1][1:],depths=depths[::-1][1:])
        
        self.final=[]
        if self.do_ds:
            
            for i in range(len(depths)-1):
                self.final.append(final_patch_expanding(embed_dim*2**i,num_classes,patch_size=patch_size))

        else:
            self.final.append(final_patch_expanding(embed_dim,num_classes,patch_size=patch_size))
    
        self.final=nn.ModuleList(self.final)
    

    def forward(self, x, pos, is_posenc = False):
        if not is_posenc:
            posenc = self.posenc(pos, self.img_size)
        else:
            posenc = pos      
            
        seg_outputs=[]
        skips = self.model_down(x, posenc)
        neck=skips[-1]
       
        out=self.decoder(neck,skips, posenc)
            
        if self.do_ds:
            for i in range(len(out)):  
                seg_outputs.append(self.final[-(i+1)](out[i]))  #LSB
            return seg_outputs[::-1]    # revert to push into DeepSupervisionLoss
        else:
            seg_outputs.append(self.final[0](out[-1]))
            return seg_outputs[-1]
        
        
        
   

   
