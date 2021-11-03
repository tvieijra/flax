## Machine Translation

Trains a Transformer-based model (Vaswani *et al.*, 2017) on the WMT Machine
Translation en-de dataset.

This example uses linear learning rate warmup and inverse square root learning
rate schedule.

Table of contents:

- [Requirements](#requirements)
- [Example runs](#example-runs)
- [Running on Cloud](#running-on-cloud)
  - [Preparing the dataset](#preparing-the-dataset)
  - [Google Cloud TPU](#google-cloud-tpu)
  
### Requirements

*   TensorFlow datasets `wmt17_translate/de-en` and `wmt14_translate/de-en` need
    to be downloaded and prepared. A sentencepiece tokenizer vocabulary will be
    automatically generated and saved on each training run.
*   This example additionally depends on the `sentencepiece` and
    `tensorflow-text` packages.

### Example runs

You should expect to get numbers similar to these:


Hardware | config  | Training time |      BLEU      |                             TensorBoard.dev                              |                                                          Workdir
-------- | ------- | ------------- | -------------- | ------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------
TPU v3-8 | default | 24m<br>13h18m | 25.55<br>32.87 | [2021-08-04](https://tensorboard.dev/experiment/nnH7JNCxTgC1ROakWePTlg/) | [gs://flax_public/examples/wmt/default](https://console.cloud.google.com/storage/browser/flax_public/examples/wmt/default)
GPU V100 x8 (Mixed Precision) | gpu_mixed_precision        | 1h 58m.       | 25.69 | [2021-07-07](https://tensorboard.dev/experiment/9S2WuqNWRDemmBuQE8K6Ew/) | -


### Running on Cloud

#### Preparing the WMT Datasets

We recommend downloading and preparing the TFDS datasets beforehand. For Cloud
TPUs, we recommend using a cheap standard instance and saving the prepared TFDS
data on a storage bucket, from where it can be loaded directly. Set the
`TFDS_DATA_DIR` to your storage bucket path (`gs://<bucket name>`).

You can download and prepare any of the WMT datasets using TFDS directly:
`python -m tensorflow_datasets.scripts.download_and_prepare
--datasets=wmt17_translate/de-en`

The typical academic BLEU evaluation also uses the WMT 2014 Test set: `python -m
tensorflow_datasets.scripts.download_and_prepare
--datasets=wmt14_translate/de-en`

#### Google Cloud TPU

Setup the TPU VM and install the Flax dependencies on it as described
[here](https://cloud.google.com/tpu/docs/jax-pods) for creating pod slices, or
[here](https://cloud.google.com/tpu/docs/jax-quickstart-tpu-vm) for a single
v3-8 TPU.

First create a single TPUv3-8 VM and connect to it (you can find more detailed
instructions [here](https://cloud.google.com/tpu/docs/jax-quickstart-tpu-vm)):

```
ZONE=us-central1-a
TPU_TYPE=v3-8
TPU_NAME=$USER-flax-wmt

gcloud alpha compute tpus tpu-vm create $TPU_NAME \
    --zone $ZONE \
    --accelerator-type $TPU_TYPE \
    --version v2-alpha

gcloud alpha compute tpus tpu-vm ssh $TPU_NAME --zone $ZONE -- \
    -L 6006:localhost:6006
```

When connected install JAX:

```
pip install "jax[tpu]>=0.2.16" \
    -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

Then install Flax + the example dependencies:

```
git clone --depth=1 --branch=main https://github.com/google/flax
cd flax
pip install -e .
cd examples/wmt
pip install -r requirements.txt
```

And finally start the training:

```
python3 main.py --workdir=$HOME/logs/wmt_256 \
    --config.per_device_batch_size=32 \
    --jax_backend_target="grpc://192.168.0.2:8470"
```

Note that you might want to set `TFDS_DATA_DIR` as explained above. You probably
also want to start the long-running command above in a `tmux` session and start
some monitoring in a separate pane (note that we forwarded port 6006 locally
above):

```
tensorboard --logdir=$HOME/logs
```

When running on pod slices, after creating the TPU VM, there are different ways
of running the training in SPMD fashion on the hosts connected to the TPUs that
make up the slice. We simply send the same installation/execution shell commands
to all hosts in parallel with the command below. If anything fails it's
usually a good idea to connect to a single host and execute the commands
interactively.

For convenience, the TPU creation commands are inlined below.

```shell
VM_NAME=wmt
REPO=https://github.com/google/flax
BRANCH=main
WORKDIR=gs://$YOUR_BUCKET/flax/examples/wmt/$(date +%Y%m%d_%H%M)

gcloud alpha compute tpus tpu-vm create $VM_NAME \
    --zone=$ZONE \
    --version v2-alpha --accelerator-type v3-32
FLAGS="--config.per_device_batch_size=32"

gcloud alpha compute tpus tpu-vm ssh $VM_NAME --zone $ZONE \
--worker=all --command "
pip install 'jax[tpu]>=0.2.21' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html &&
pip install --user git+$REPO.git &&
git clone --depth=1 -b $BRANCH $REPO &&
cd flax/examples/wmt &&
pip install -r requirements.txt &&
export TFDS_DATA_DIR=gs://$GCS_TFDS_BUCKET/datasets &&
python3 main.py --workdir=$WORKDIR --config=configs/default.py $FLAGS
"
```

