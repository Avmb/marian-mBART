# marian-mBART
Training harness to pretrain a Marian model using mBART

Provides the [mBART pretraining strategy](https://arxiv.org/abs/2001.08210) for [Marian](https://github.com/marian-nmt/marian-dev) neural machine translation models.

Implemented using an external training harness that reads monolingual data, applies mBART noise and sends it to a Marian training process using a pair of named pipes.

