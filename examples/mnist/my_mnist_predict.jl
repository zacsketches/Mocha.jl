using Mocha

backend = CPUBackend()
init(backend)

#Create a memory data layer in the prediction model that we computation
#sequentially tie to the next image we want to predict on.
mem_data = MemoryDataLayer(name="data", tops=[:data], batch_size=1,
  data=Array[zeros(Float32, 28, 28, 1, 1)])

#Create the network with two conv/pooling layers and two fully
#connected inner product layers
conv_layer = ConvolutionLayer(name="conv1", n_filter=20, kernel=(5,5),
  bottoms=[:data], tops=[:conv1])
pool_layer = PoolingLayer(name="pool1", kernel=(2,2), stride=(2,2),
  bottoms=[:conv1], tops=[:pool1])
conv2_layer = ConvolutionLayer(name="conv2", n_filter=20, kernel=(5,5),
  bottoms=[:pool1], tops=[:conv2])
pool2_layer = PoolingLayer(name="pool2", kernel=(2,2), stride  = (2,2),
  bottoms=[:conv2], tops=[:pool2])
fc1_layer = InnerProductLayer(name="ip1", output_dim=500,
  neuron=Neurons.ReLU(), bottoms=[:pool2], tops=[:ip1])
fc2_layer = InnerProductLayer(name="ip2", output_dim=10,
  bottoms=[:ip1], tops=[:ip2])

#Instead of a SoftmaxLossLayer like the training script in this file we
#have a pure SoftmaxLayer in order to determine the probability of the 
#image compared against the different n classification options
softmax_layer = SoftmaxLayer(name="prob", tops=[:prob], bottoms=[:ip2])

#Build the network
common_layers = [conv_layer, pool_layer, conv2_layer, pool2_layer,
  fc1_layer, fc2_layer]
run_net = Net("imagenet", backend, [mem_data, common_layers..., softmax_layer])

#Load the latest snapshot from the training data
load_snapshot(run_net, "snapshots_bak/snapshot-010000.jld")

#Now load one image at a time
using HDF5
h5open("data/test.hdf5") do f
    get_layer(run_net, "data").data[1][:,:,1,1] = f["data"][:,:,1,1]
    println("Correct label index: ", Int64(f["label"][:,1][1]+1))
end

forward(run_net)
println()
println("Label Prob vector:")
println(run_net.output_blobs[:prob].data)

destroy(run_net)
shutdown(backend)
