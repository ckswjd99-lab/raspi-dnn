MODEL_EFFVIT_B0=./data/efficientvit_b0.r224_in1k.onnx
MODEL_MNASNET=./data/mnasnet_100.rmsp_in1k.onnx
MODEL_MIXNET_S=./data/tf_mixnet_s.in1k.onnx
MODEL_MOBNETV3_L=./data/tf_mobilenetv3_large_075.in1k.onnx
MODEL_RESNET18=./data/resnet18-v1-7.onnx
MODEL_REPGHOSTNET=./data/repghostnet_080.in1k.onnx
MODEL_SQUEEZENET=./data/squeezenet1.1-7.onnx

MODEL_PATH=$MODEL_EFFVIT_B0
IMAGE_PATH=./data/european-bee-eater-2115564_1920.jpg
NUM_INTRA_THREADS=2
NUM_INTER_THREADS=1
NUM_MULTI_THREADS=4
NUM_TEST=10

./imnet_classifier.out $MODEL_PATH $IMAGE_PATH $NUM_INTRA_THREADS $NUM_INTER_THREADS $NUM_MULTI_THREADS $NUM_TEST