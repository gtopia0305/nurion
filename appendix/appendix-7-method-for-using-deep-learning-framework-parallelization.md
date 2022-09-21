# 딥러닝 프레임워크 병렬화 사용법

## 가. Tensorflow에서 Horovod 사용법

다중노드에서 CPU를 활용할 경우 Horovod를 Tensorflow와 연동하여 병렬화가 가능하다. 아래 예시와 같이 Horovod 사용을 위한 코드를 추가해주면 Tensorflow와 연동이 가능하다. Tensorflow 및 Tensorflow에서 활용 가능한 Keras API 모두 Horovod와 연동이 가능하며 우선 Tensorflow에서 Horovod와 연동하는 방법을 소개한다.\
(예시: MNIST Dataset 및 LeNet-5 CNN 구조)



※ Tensorflow에서 Horovod 활용을 위한 자세한 사용법은 Horovod 공식 가이드 참조\
([https://github.com/horovod/horovod#usage](https://github.com/horovod/horovod#usage))

****

**◦ Tensorflow에서 Horovod 사용을 위한 import 및 메인 함수에서 Horovod 초기화**

```
import horovod.tensorflow as hvd
...
hvd.init()
```

※ horovod.tensorflow: Horovod를 Tensorflow와 연동하기 위한 모듈

※ Horovod를 사용하기 위하여 초기화한다.

****

**◦ 메인 함수에서 Horovod 활용을 위한 Dataset 설정**

```
(x_train, y_train), (x_test, y_test) = \
keras.datasets.mnist.load_data('MNIST-data-%d' % hvd.rank())
```

※ 각 작업별로 접근할 dataset을 설정하기 위하여 Horovod rank에 따라 설정 및 생성한다.

****

**◦ 메인 함수에서 optimizer에 Horovod 관련 설정 및 broadcast, 학습 진행 수 설정**

```
opt = tf.train.AdamOptimizer(0.001 * hvd.size())
opt = hvd.DistributedOptimizer(opt)
global_step = tf.train.get_or_create_global_step()
train_op = opt.minimize(loss, global_step=global_step)
hooks = [hvd.BroadcastGlobalVariablesHook(0),
                  tf.train.StopAtStepHook(last_step=20000 // hvd.size()), ... ]
```

※ Optimizer에 Horovod 관련 설정을 적용하고 각 작업에 broadcast를 활용하여 전달함

※ 각 작업들의 학습과정 step을 Horovod 작업 수에 따라 설정함

****

**◦ Inter operation 및 Intra operation의 병렬처리 설정**

```
config = tf.ConfigProto()
config.intra_op_parallelism_threads = int(os.environ[‘OMP_NUM_THREADS’])
config.inter_op_parallelism_threads = 2
```

※ config.intra\_op\_parallelism\_threads: 연산 작업에서 사용할 thread 개수를 설정하는데 사용되며 job script에서 설정한 OMP\_NUM\_THREADS를 불러와서 적용함 (본 예시의 경우 OMP\_NUM\_THREADS를 32로 설정함)

※ config.intra\_op\_parallelism\_threads: TensorFlow의 작업을 동시에 실행하는 thread 개수이며 예시와 같이 2로 설정할 경우 두 개의 작업이 병렬적으로 실행됨



◦ Rank 0 작업에 Checkpoint 설정

```
checkpoint_dir = './checkpoints' if hvd.rank() == 0 else None
...
with tf.train.MonitoredTrainingSession(checkpoint_dir=checkpoint_dir,
hooks=hooks,
config=config) as mon_sess:
```

|   |
| - |

※ Checkpoint 저장 및 불러오는 작업은 하나의 프로세스에서 수행되어야 하므로 rank 0번에 설정함

## 나. Intel Caffe에서 다중노드 사용법

Caffe의 다중노드 병렬화는 Horovod에서 공식적으로 지원하지 않으며 Intel에서 KNL에 최적화하여 개발한 Intel Caffe를 활용하여 병렬처리가 가능하다. Intel Caffe의 경우 병렬처리를 위한 작업들이 모두 코드 개발과정에서 적용되어 기존 Caffe에서 개발된 deploy.prototxt, solver.prototxt, train\_val.prototxt들을 그대로 사용할 수 있다.



※ Intel Caffe 활용을 위한 자세한 사용법은 Intel Caffe 공식 가이드 참조\
([https://github.com/intel/caffe/wiki/Multinode-guide](https://github.com/intel/caffe/wiki/Multinode-guide))



딥러닝 개발자에 의해 수정된 Caffe의 코드에 대하여 병렬처리를 수행할 경우 Intel Caffe 소스코드에 해당 부분을 업데이트하여 컴파일 후 실행하여야 한다.

****

**◦ Intel Caffe 병렬처리 수행 방법 (작업스크립트 예제)**

> \#!/bin/sh
>
> \#PBS -N test
>
> \#PBS -V
>
> \#PBS -l select=4:ncpus=68:mpiprocs=1:ompthreads=68
>
> \#PBS -q normal
>
> \#PBS -l walltime=04:00:00
>
> \#PBS -A caffe
>
>
>
> cd $PBS\_O\_WORKDIR
>
>
>
> module purge
>
> module load conda/intel\_caffe\_1.1.5
>
> source /apps/applications/miniconda3/envs/intel\_caffe/mlsl\_2018.3.008/intel64/bin/mlslvars.sh
>
>
>
> export KMP\_AFFINITY=verbose,granularity=fine,compact=1
>
> export KMP\_BLOCKTIME=1
>
> export KMP\_SETTINGS=1
>
> <mark style="color:blue;"></mark>
>
> <mark style="color:blue;">export OMP\_NUM\_THREADS=60</mark>
>
> <mark style="color:blue;">mpirun -PSM2 -prepend-rank caffe train \\</mark>
>
> <mark style="color:blue;">--solver ./models/intel\_optimized\_models/multinode/alexnet\_4nodes/solver.prototxt</mark>
>
> <mark style="color:blue;"></mark>
>
> <mark style="color:blue;"># 혹은</mark>
>
> <mark style="color:blue;"></mark>
>
> <mark style="color:blue;">./scripts/run\_intelcaffe.sh --hostfile $PBS\_NODEFILE \\</mark>
>
> <mark style="color:blue;">--caffe\_bin /apps/applications/miniconda3/envs/intel\_caffe/bin/caffe \\</mark>
>
> <mark style="color:blue;">--solver models/intel\_optimized\_models/multinode/alexnet\_4nodes/solver.prototxt \\</mark>
>
> <mark style="color:blue;">--network opa --ppn 1 --num\_omp\_threads 60</mark>
>
>
>
> exit 0

※ Network 옵션: Intel Onmi-Path Architecture (OPA)로 설정

※ PPN: Process per node의 약자로 노드당 작업수를 뜻함 (기본값: 1)

※ Script를 이용하지 않고도 MPI 실행을 통해 기존 Caffe 수행 방법과 동일하게 수행 가능



{% hint style="info" %}
2022년 4월 12일에 마지막으로 업데이트되었습니다.
{% endhint %}
