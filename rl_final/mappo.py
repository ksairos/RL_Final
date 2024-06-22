# import torch
# cuda0 = torch.device('cuda:0')
# print(torch.cuda.device_count())
# print(torch.ones([2, 4], dtype=torch.float64, device=cuda0))

from marllib import marl

# prepare env
env = marl.make_env(environment_name="gobigger", map_name="st_t1p2")

# print("\n\n", env, "\n\n")

# initialize algorithm with appointed hyper-parameters
mappo = marl.algos.mappo(hyperparam_source="gobigger")
# build agent model based on env + algorithms + user preference
model = marl.build_model(env, mappo, {"core_arch": "mlp", "encode_layer": "128-256"})
# start training
mappo.fit(env, model, stop={"timesteps_total": 1000000}, checkpoint_freq=100, local_mode=False, num_gpus=0, num_workers=10, share_policy="group")

# # rendering
# mappo.render(env, model,
#              restore_path={'params_path': "checkpoint/params.json",  # experiment configuration
#                            'model_path': "checkpoint/checkpoint-6250", # checkpoint path
#                            'render': True},  # render
#              local_mode=False, num_gpus=1, num_workers=10,
#              share_policy="group",
#              checkpoint_end=False)