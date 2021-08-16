from .dwnet import DWNet

def build_model(config):
    model_type = config.MODEL.TYPE
    if model_type == 'dwnet':
        model = DWNet(img_size=config.DATA.IMG_SIZE,
                        patch_size=config.MODEL.DWNET.PATCH_SIZE,
                        in_chans=config.MODEL.DWNET.IN_CHANS,
                        num_classes=config.MODEL.NUM_CLASSES,
                        embed_dim=config.MODEL.DWNET.EMBED_DIM,
                        depths=config.MODEL.DWNET.DEPTHS,
                        window_size=config.MODEL.DWNET.WINDOW_SIZE,
                        mlp_ratio=config.MODEL.DWNET.MLP_RATIO,
                        drop_rate=config.MODEL.DROP_RATE,
                        drop_path_rate=config.MODEL.DROP_PATH_RATE,
                        ape=config.MODEL.DWNET.APE,
                        patch_norm=config.MODEL.DWNET.PATCH_NORM,
                        use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                        dynamic=config.MODEL.DWNET.DYNAMIC)     
    elif model_type == 'ddwnet':
        model = DWNet(img_size=config.DATA.IMG_SIZE,
                        patch_size=config.MODEL.DWNET.PATCH_SIZE,
                        in_chans=config.MODEL.DWNET.IN_CHANS,
                        num_classes=config.MODEL.NUM_CLASSES,
                        embed_dim=config.MODEL.DWNET.EMBED_DIM,
                        depths=config.MODEL.DWNET.DEPTHS,
                        window_size=config.MODEL.DWNET.WINDOW_SIZE,
                        mlp_ratio=config.MODEL.DWNET.MLP_RATIO,
                        drop_rate=config.MODEL.DROP_RATE,
                        drop_path_rate=config.MODEL.DROP_PATH_RATE,
                        ape=config.MODEL.DWNET.APE,
                        patch_norm=config.MODEL.DWNET.PATCH_NORM,
                        use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                        dynamic=config.MODEL.DWNET.DYNAMIC)                           
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
