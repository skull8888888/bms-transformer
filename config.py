# ====================================================
# CFG
# ====================================================
class CFG:
    debug=False
    max_len=277
    num_workers=11

    backbone='tf_efficientnet_b0_ns'
    d_encoder=1280
    d_model=512
    n_head=8
    
    encoder_layers=1
    decoder_layers=2
    
    size=224

    epochs=10

    lr=5e-4
    div_factor=25
    pct_start=0.0
    batch_size=128
    l2=0
    dropout=0.5
    seed=42
    n_fold=5
    trn_fold=[0] # [0, 1, 2, 3, 4]
    train=True