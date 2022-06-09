script_name1=`basename $0`
script_name=${script_name1:0:${#script_name1}-3}

python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --use_env \
    main.py  --batch_size 4 --num_queries 5\
    --coco_path ../data/coco \
    --output_dir output/$script_name