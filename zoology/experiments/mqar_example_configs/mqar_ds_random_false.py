import uuid
import numpy as np
from zoology.config import TrainConfig, DataConfig, LoggerConfig
from zoology.data.multiquery_ar import MQARConfig


sweep_id = uuid.uuid4().hex[:6]
sweep_name = "csa-hca-screening-" + sweep_id

VOCAB_SIZE = 8_192
CACHE_DIR = "/home/ssd2/harness/zoology/zoology/cache_dir"

# ---------------------------------------------------------------------------
# 1. Data configuration (same MQAR tasks as the standard sweep)
# ---------------------------------------------------------------------------

train_configs = [
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=64,  num_examples=100_000, num_kv_pairs=4,  random_non_queries=False),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=128, num_examples=20_000,  num_kv_pairs=8,  random_non_queries=False),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=256, num_examples=20_000,  num_kv_pairs=16, random_non_queries=False),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=256, num_examples=20_000,  num_kv_pairs=32, random_non_queries=False),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=256, num_examples=20_000,  num_kv_pairs=64, random_non_queries=False),
]

test_configs = [
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=64,   num_examples=1_000, num_kv_pairs=4,   random_non_queries=False),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=64,   num_examples=1_000, num_kv_pairs=8,   random_non_queries=False),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=64,   num_examples=1_000, num_kv_pairs=16,  random_non_queries=False),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=128,  num_examples=1_000, num_kv_pairs=32,  random_non_queries=False),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=256,  num_examples=1_000, num_kv_pairs=64,  random_non_queries=False),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=512,  num_examples=1_000, num_kv_pairs=128, random_non_queries=False),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=1024, num_examples=1_000, num_kv_pairs=256, random_non_queries=False),
]

input_seq_len = max([c.input_seq_len for c in train_configs + test_configs])
batch_size = 256
data = DataConfig(
    train_configs=train_configs,
    test_configs=test_configs,
    batch_size=(batch_size, batch_size // 8),
    cache_dir=CACHE_DIR,
)

# ---------------------------------------------------------------------------
# 2. Model configurations – CSA and HCA
# ---------------------------------------------------------------------------

models = []

model_factory_kwargs = {
    "state_mixer": dict(name="torch.nn.Identity", kwargs={}),
    "vocab_size": VOCAB_SIZE,
}

conv_mixer = dict(
    name="zoology.mixers.base_conv.BaseConv",
    kwargs={
        "l_max": input_seq_len,
        "kernel_size": 3,
        "implicit_long_conv": True,
    }
)

from zoology.experiments.models_repo import add_deepseek_nsa, add_csa, add_hca, add_deepseek_csa_hca, add_cla, add_mhcla, add_delta_net, add_ela, add_msd

# models = add_deepseek_nsa(models, conv_mixer, input_seq_len, model_factory_kwargs, num_layers=3)
# models = add_csa(models, conv_mixer, input_seq_len, model_factory_kwargs, num_layers=3)
# models = add_hca(models, conv_mixer, input_seq_len, model_factory_kwargs, num_layers=3)
# models = add_deepseek_csa_hca(models, conv_mixer, input_seq_len, model_factory_kwargs, num_layers=3)

# models = add_cla(models, conv_mixer, input_seq_len, model_factory_kwargs, num_layers=3)
# models = add_ela(models, conv_mixer, input_seq_len, model_factory_kwargs, num_layers=3)
# models = add_mhcla(models, conv_mixer, input_seq_len, model_factory_kwargs, num_layers=3)
models = add_msd(models, conv_mixer, input_seq_len, model_factory_kwargs, num_layers=3)
# models = add_delta_net(models, conv_mixer, input_seq_len, model_factory_kwargs, num_layers=3)

for model in models:
    model.embedding_init_type = "spherical"
    model.learnable_word_embeddings = True

# ---------------------------------------------------------------------------
# 3. Train configs (sweep learning rates)
# ---------------------------------------------------------------------------

configs = []
for model in models:
    for lr in np.logspace(-3, -1.5, 4):
    # for lr in np.logspace(-2.5, -2, 2):
    # for lr in np.logspace(-2.5, -2.5, 1):
    # for lr in [1.0e-3]:
        # if model.d_model in [64, 256]:
        #     continue
        run_id = f"{model.name}-d{model.d_model}-lr{lr:.1e}"
        config = TrainConfig(
            model=model,
            data=data,
            learning_rate=lr,
            max_epochs=32,
            logger=LoggerConfig(
                project_name="zoology-cla",
                entity="lijialin03"
            ),
            slice_keys=["num_kv_pairs"],
            sweep_id=sweep_name,
            run_id=run_id,
            predictions_path=f"{CACHE_DIR}/predictions/{run_id}",
            collect_predictions=True,
        )
        configs.append(config)