python tools/train.py --config_file='configs/softmax_triplet.yml' \
                        MODEL.DEVICE_ID "('0')" \
                        DATASETS.NAMES "('sig')" \
                        DATASETS.ROOT_DIR "('/content/data')" \
                        OUTPUT_DIR "('/content/drive/My Drive/training/sig_ReID_w/o_BN')"
