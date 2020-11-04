original repo: https://github.com/michuanhaohao/reid-strong-baseline.git  

## Prepare dataset
* Prepare your dataset at "data/".
* The diretory structure should look like:  
```
data
  sig  
    id_1
      sig_1
      sig_2
      sig_3
      .
      .  
    id_2
      sig_1
      sig_2
      sig_3
      .
      .  
    ...
```
## Train
* Reconfig `train.sh` if needed, see 'config/' for default configs and 'configs/' for some particular configs.  
* Remember to set `OUTPUT_DIR` in `train.sh` to yours.
* Run `sh train.sh`.  

## Augment data
* Add your custom augmentation at "data/transforms/build.py"
