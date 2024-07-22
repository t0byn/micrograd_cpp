#ifndef _NN_H_
#define _NN_H_

#include "engine.h"

#include <random>

std::random_device g_rd;
std::mt19937 g_gen(g_rd());
std::uniform_real_distribution<float> g_dis(-1.f, 1.f);

struct Neuron
{
    std::vector<ValueHandle> parameters;
};

struct NeuronHandle
{
    int idx;
};

#define MAX_NEURON_NUMBER 128
struct NeuronPool
{
    int neuron_count = 0;
    Neuron neurons[MAX_NEURON_NUMBER];
};
NeuronPool g_neuron_pool = {};

NeuronHandle create_neuron(int input)
{
    assert(g_neuron_pool.neuron_count < MAX_NEURON_NUMBER - 1);
    if (g_neuron_pool.neuron_count == MAX_NEURON_NUMBER - 1)
    {
        fprintf(stderr, "neuron pool reach maximum capacity %d! create neuron failed!", MAX_NEURON_NUMBER);
        return NeuronHandle{ .idx = -1 };
    }

    Neuron& neuron = g_neuron_pool.neurons[g_neuron_pool.neuron_count];
    neuron.parameters.resize(input + 1);
    for (int i = 0; i < input + 1; i++)
    {
        float rn = g_dis(g_gen);
        neuron.parameters[i] = create_value(rn);
    }

    return NeuronHandle{
        .idx = g_neuron_pool.neuron_count++
    };
}

bool valid_neuron(NeuronHandle h)
{
    assert(h.idx >= 0 && h.idx < g_neuron_pool.neuron_count);
    if (h.idx < 0 || h.idx >= g_neuron_pool.neuron_count)
    {
        return false;
    }
    return true;
}

Neuron* get_neuron(NeuronHandle h)
{
    assert(h.idx >= 0 && h.idx < g_neuron_pool.neuron_count);
    if (h.idx < 0 || h.idx >= g_neuron_pool.neuron_count)
    {
        return NULL;
    }
    return &g_neuron_pool.neurons[h.idx];
}

ValueHandle run_neuron(NeuronHandle h, std::vector<ValueHandle> input)
{
    Neuron* n = get_neuron(h);
    assert(n->parameters.size() == input.size() + 1);
    ValueHandle sum = n->parameters.back();
    for (int i = 0; i < input.size(); i++)
    {
        ValueHandle wx = n->parameters[i] * input[i];
        sum = sum + wx;
    }
    ValueHandle output = tanh(sum);
    return output;
}


struct Layer
{
    std::vector<NeuronHandle> neurons;
};

struct LayerHandle
{
    int idx;
};

#define MAX_LAYER_NUMBER 16
struct LayerPool
{
    int layer_count = 0;
    Layer layers[MAX_LAYER_NUMBER];
};
LayerPool g_layer_pool = {};

LayerHandle create_layer(int input, int output)
{
    assert(g_layer_pool.layer_count < MAX_LAYER_NUMBER - 1);
    if (g_layer_pool.layer_count == MAX_LAYER_NUMBER - 1)
    {
        fprintf(stderr, "layer pool reach maximum capacity %d! create layer failed!", MAX_LAYER_NUMBER);
        return LayerHandle{ .idx = -1 };
    }

    Layer& layer = g_layer_pool.layers[g_layer_pool.layer_count];
    layer.neurons.resize(output);
    for (int i = 0; i < output; i++)
    {
        layer.neurons[i] = create_neuron(input);
    }

    return LayerHandle{
        .idx = g_layer_pool.layer_count++
    };
}

bool valid_layer(LayerHandle h)
{
    assert(h.idx >= 0 && h.idx < g_layer_pool.layer_count);
    if (h.idx < 0 || h.idx >= g_layer_pool.layer_count)
    {
        return false;
    }
    return true;
}

Layer* get_layer(LayerHandle h)
{
    assert(h.idx >= 0 && h.idx < g_layer_pool.layer_count);
    if (h.idx < 0 || h.idx >= g_layer_pool.layer_count)
    {
        return NULL;
    }
    return &g_layer_pool.layers[h.idx];
}

std::vector<ValueHandle> run_layer(LayerHandle h, std::vector<ValueHandle> input)
{
    Layer* layer = get_layer(h);
    std::vector<ValueHandle> output;
    for (int i = 0; i < layer->neurons.size(); i++)
    {
        ValueHandle out = run_neuron(layer->neurons[i], input);
        output.push_back(out);
    }
    return output;
}


struct MLP
{
    std::vector<LayerHandle> layers;
};

void mlp_init(MLP& mlp, int input, std::vector<int> layers)
{
    std::vector<int> in_out;
    in_out.push_back(input);
    for (int i = 0; i < layers.size(); i++)
    {
        in_out.push_back(layers[i]);
    }

    mlp.layers.clear();
    for (int i = 0; i < in_out.size() - 1; i++)
    {
        LayerHandle layer = create_layer(in_out[i], in_out[i+1]);
        mlp.layers.push_back(layer);
    }
}

std::vector<ValueHandle> mlp_parameters(MLP& mlp)
{
    std::vector<ValueHandle> parameters;
    for (int i = 0; i < mlp.layers.size(); i++)
    {
        Layer* layer = get_layer(mlp.layers[i]);
        for (int j = 0; j < layer->neurons.size(); j++)
        {
            Neuron* neuron = get_neuron(layer->neurons[j]);
            for (int k = 0; k < neuron->parameters.size(); k++)
            {
                parameters.push_back(neuron->parameters[k]);
            }
        }
    }
    return parameters;
}

void mlp_zero_grad(MLP& mlp)
{
    for (int i = 0; i < mlp.layers.size(); i++)
    {
        Layer* layer = get_layer(mlp.layers[i]);
        for (int j = 0; j < layer->neurons.size(); j++)
        {
            Neuron* neuron = get_neuron(layer->neurons[j]);
            for (int k = 0; k < neuron->parameters.size(); k++)
            {
                Value* parameter = get_value(neuron->parameters[k]);
                parameter->gradient = 0.f;
            }
        }
    }
}

std::vector<ValueHandle> mlp_forward(MLP& mlp, std::vector<ValueHandle> input)
{
    for (int i = 0; i < mlp.layers.size(); i++)
    {
        input = run_layer(mlp.layers[i], input);
    }
    return input;
}

ValueHandle mean_squared_error(std::vector<ValueHandle> prediction, std::vector<ValueHandle> expect)
{
    assert(prediction.size() == expect.size());
    ValueHandle loss = pow(prediction[0] - expect[0], 2.f);
    for (int i = 1; i < prediction.size(); i++)
    {
        loss = loss + pow(prediction[i] - expect[i], 2.f);
    }
    return loss;
}

void mlp_backward(MLP& mlp, ValueHandle loss, float learning_rate)
{
    backward(loss);
    for (int i = 0; i < mlp.layers.size(); i++)
    {
        Layer* layer = get_layer(mlp.layers[i]);
        for (int j = 0; j < layer->neurons.size(); j++)
        {
            Neuron* neuron = get_neuron(layer->neurons[j]);
            for (int k = 0; k < neuron->parameters.size(); k++)
            {
                Value* parameter = get_value(neuron->parameters[k]);
                parameter->data += -learning_rate * parameter->gradient;
            }
        }
    }
}


#endif