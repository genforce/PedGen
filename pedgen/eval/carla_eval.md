## Evaluate PedGen on CARLA

Pull and run the CARLA docker:
```
docker pull carlasim/carla:0.9.15
cd pedgen/eval
docker-compose up
```
Generate the test images from CARLA:
```
python carla_generation.py carla_test 2000
```
Run inference on these images:
```
python scripts/main.py test -c cfgs/pedgen_with_context.yaml --data.test_carla True --exp_name carla_eval --version carla_eval
```
Use the predicted motion to evaluate in CARLA again. 
```
python carla_evaluation.py experiments/carla_eval/carla_eval carla_eval 2000
```
Detailed instructions to be updated.