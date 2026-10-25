You are a research agent, below is some instructions you need to follow.

# Coding

1. You need to use the `.venv` from `~/git/rar/`
2. activate this `.venv` through `source .venv/bin/activate`
3. Don't write any documents.
4. Write clean code, don't write like software engineering like code, it's a research project
5. Use `export CUDA_VISIBLE_DEVICES=1` to select my personal gpu
6. Use axolotl for finetune model, and use wandb to track experiments
7. give me the exact pyproject.toml and other related files so that I can build the environment on other server using uv

# TODO

We want to test whether small LLMs will have similar performance against bigger LLMs, so we can refer to `~/git/t5_gr`, there exists training configs with T5-Large, I want to test the performance on T5-base and T5-small, and see the difference. Use `nq` dataset as default. I'll push the code to github and pull in other server with more resources, so we just need one step training here to make sure the program itself can be work without bug. If anything is not sure, just ask me, don't write first.
