!bin/bash 
datasets=('VOC12')
archs=('dino_base' 'dino_small')
patch_sizes=(8)
box_sizes=(10 30 50 70 100 150 200)

for dataset in ${datasets[@]}
    do 
        for arch in ${archs[@]}
            do
                for patch_size in ${patch_sizes[@]}
                    do 
                        for box_size in ${box_sizes[@]}
                            do 
                                python main_stcut.py --dataset $dataset --arch $arch --patch_size $patch_size --min_box_size $box_size --depth -2
                            done 
                    done 
            done 
    done 