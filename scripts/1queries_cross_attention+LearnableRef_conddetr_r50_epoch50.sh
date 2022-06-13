script_name1=`basename $0`
script_name=${script_name1:0:${#script_name1}-3}

python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --use_env \
    main.py --batch_size 4 --num_queries 1\
    --coco_path ../coco \
    --output_dir output/$script_name