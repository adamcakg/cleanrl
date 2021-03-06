for seed in {10..10}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo_car_racing.py \
    --total-timesteps 10000000 \
    --wandb-project-name cleanrl.benchmark \
    --prod-mode \
    --wandb-entity cleanrl --cuda True \
    --capture-video \
    --seed $seed
    ) >& /dev/null &
done
