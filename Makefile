.DEFAULT_GOAL := all

all: data_hier

data_hier:
	mkdir -p data
	mkdir -p data/log
	mkdir -p data/eval_result

clean:
	rm ./data/processed_dataset/processed/*.pt
	rm ./data/processed_dataset/raw/*.pkl

sanity_check:
	python ./test/test_rl.py $(unit_test_2) --log_name sanity_check.log --lr 0.0001 --eps_decay 0.99 --episode_iter 10000 --reward_type no_intermediate 
unit3:
	python ./test/test_rl.py $(unit_test_3) --log_name unit3_only_s.log --lr 0.0001 --eps_decay 0.999 --episode_iter 50000 --reward_type no_intermediate 
