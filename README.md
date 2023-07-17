# Example of how to invoke TensorFlow Serving from Rust

This example uses the Rust TensorFlow Serving Client: https://docs.rs/tensorflow-serving-client/latest/tensorflow_serving_client/

## Running TensorFlow Serving using Docker

The steps described here are based on this tutorial: https://www.tensorflow.org/tfx/serving/docker

First, you need to pull the latest image:

```bash
docker pull tensorflow/serving
```

Then, you can run the image using the pre-trained model within the `model` folder available in this repository:

```bash
docker run -d --rm -p 8500:8500 -p 8501:8501 \
    --name tensorflowiris \
    -v "$(pwd)/model:/models/iris" \
    -e MODEL_NAME=iris \
    tensorflow/serving
```

To stop the container, run:

```bash
docker stop tensorflowiris
```

## Run the TensorFlow Serving client

There are some prerequisites for running this project, which can be found here: https://github.com/tikv/grpc-rs#prerequisites

Using an Ubuntu 22.04, I used the following command to install the dependencies:

```bash
sudo apt install cmake binutils clang
```

Then, you can run:

```bash
cargo run
```
