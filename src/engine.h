#ifndef _ENGINE_H_
#define _ENGINE_H_

#include <assert.h>
#include <stdlib.h>
#include <stdint.h>
#include <vector>
#include <unordered_set>

enum MathOperation
{
    NONE = 0,
    ADD,
    MULTIPLE,
    POW,
    EXP,
    TANH,
};

struct ValueHandle
{
    int idx;
};

struct Value
{
    float data = 0.f;
    float gradient = 0.f;
    float exponent = 0.f;
    MathOperation op = MathOperation::NONE;
    std::vector<ValueHandle> input;
};

#define TEMP_VALUE_POOL_START { int temp_value_count = g_value_pool.value_count;
#define TEMP_VALUE_POOL_END g_value_pool.value_count = temp_value_count; }

#define MAX_VALUE_NUMBER 1024
struct ValuePool
{
    int value_count = 0;
    Value values[MAX_VALUE_NUMBER];
};
ValuePool g_value_pool = {};

ValueHandle create_value(float data, MathOperation op = MathOperation::NONE)
{
    assert(g_value_pool.value_count < MAX_VALUE_NUMBER - 1);
    if (g_value_pool.value_count == MAX_VALUE_NUMBER - 1)
    {
        fprintf(stderr, "value pool reach maximum capacity %d! create value failed!", MAX_VALUE_NUMBER);
        return ValueHandle{ .idx = -1 };
    }

    g_value_pool.values[g_value_pool.value_count].data = 0.f;
    g_value_pool.values[g_value_pool.value_count].gradient = 0.f;
    g_value_pool.values[g_value_pool.value_count].exponent = 0.f;
    g_value_pool.values[g_value_pool.value_count].op = MathOperation::NONE;
    g_value_pool.values[g_value_pool.value_count].input.clear();

    g_value_pool.values[g_value_pool.value_count] = {
        .data = data,
        .op = op
    };
    g_value_pool.value_count++;

    return ValueHandle{ .idx = g_value_pool.value_count - 1 };
}

bool valid_value(ValueHandle h)
{
    assert(h.idx >= 0 && h.idx < g_value_pool.value_count);
    if (h.idx < 0 || h.idx >= g_value_pool.value_count)
    {
        return false;
    }
    return true;
}

Value* get_value(ValueHandle h)
{
    assert(h.idx >= 0 && h.idx < g_value_pool.value_count);
    if (h.idx < 0 || h.idx >= g_value_pool.value_count)
    {
        return NULL;
    }
    return &g_value_pool.values[h.idx];
}

std::vector<ValueHandle> topo_sort()
{
    std::vector<ValueHandle> topo;
    std::vector<bool> child_sorted(g_value_pool.value_count, false);
    std::unordered_set<int> visited;
    for (int i = 0; i < g_value_pool.value_count; i++)
    {
        if (visited.contains(i)) continue;

        std::vector<int> dfs = { i };
        while (dfs.size() > 0)
        {
            int idx = dfs.back();
            if (child_sorted[idx])
            {
                dfs.pop_back();
                topo.push_back(ValueHandle{ .idx = idx });
                continue;
            }

            visited.insert(idx);

            Value& value = g_value_pool.values[idx];
            for (int j = 0; j < value.input.size(); j++)
            {
                ValueHandle child = value.input[j];
                if (!valid_value(child)) continue;
                int child_idx = child.idx;
                if (visited.contains(child_idx)) continue;
                dfs.push_back(child_idx);
            }

            child_sorted[idx] = true;
        }
    }

    return topo;
}

void calc_gradient(ValueHandle hout)
{
    Value* out = get_value(hout);
    switch(out->op)
    {
    case MathOperation::ADD:
        {
            assert(out->input.size() == 2);
            Value* a = get_value(out->input[0]);
            Value* b = get_value(out->input[1]);
            a->gradient += 1.f * out->gradient;
            b->gradient += 1.f * out->gradient;
        }
        break;
    case MathOperation::MULTIPLE:
        {
            assert(out->input.size() == 2);
            Value* a = get_value(out->input[0]);
            Value* b = get_value(out->input[1]);
            a->gradient += b->data * out->gradient;
            b->gradient += a->data * out->gradient;
        }
        break;
    case MathOperation::POW:
        {
            assert(out->input.size() == 1);
            Value* a = get_value(out->input[0]);
            a->gradient += out->exponent * powf(a->data, out->exponent - 1.f) * out->gradient;
        }
        break;
    case MathOperation::EXP:
        {
            assert(out->input.size() == 1);
            Value* a = get_value(out->input[0]);
            a->gradient += out->data * out->gradient;
        }
        break;
    case MathOperation::TANH:
        {
            assert(out->input.size() == 1);
            Value* a = get_value(out->input[0]);
            a->gradient += (1.f - powf(out->data, 2)) * out->gradient;
        }
        break;
    default:
        break;
    }
}

void backward(ValueHandle hroot)
{
    std::vector<ValueHandle> topo = topo_sort();

    Value* root = get_value(hroot);
    root->gradient = 1.f;

    for (int i = topo.size() - 1; i >= 0; i--)
    {
        ValueHandle hnode = topo[i];
        calc_gradient(hnode);
    }
}


ValueHandle operator+(ValueHandle ha, ValueHandle hb)
{
    Value* a = get_value(ha);
    Value* b = get_value(hb);
    ValueHandle ho = create_value(a->data + b->data, MathOperation::ADD);
    Value* out = get_value(ho);
    out->input.push_back(ha);
    out->input.push_back(hb);
    return ho;
}

ValueHandle operator+(ValueHandle ha, float s)
{
    ValueHandle hb = create_value(s);
    return ha + hb;
}

ValueHandle operator+(float s, ValueHandle ha)
{
    return ha + s;
}

ValueHandle operator*(ValueHandle ha, ValueHandle hb)
{
    Value* a = get_value(ha);
    Value* b = get_value(hb);
    ValueHandle ho = create_value(a->data * b->data, MathOperation::MULTIPLE);
    Value* out = get_value(ho);
    out->input.push_back(ha);
    out->input.push_back(hb);
    return ho;
}

ValueHandle operator*(ValueHandle ha, float s)
{
    ValueHandle hb = create_value(s);
    return ha * hb;
}

ValueHandle operator*(float s, ValueHandle ha)
{
    return ha * s;
}

ValueHandle operator-(ValueHandle ha)
{
    return ha * -1.f;
}

ValueHandle operator-(ValueHandle ha, ValueHandle hb)
{
    return ha + (-hb);
}

ValueHandle operator-(ValueHandle ha, float s)
{
    return ha + (-s);
}

ValueHandle operator-(float s, ValueHandle ha)
{
    return s + (-ha);
}

ValueHandle pow(ValueHandle ha, float s)
{
    Value* a = get_value(ha);
    ValueHandle ho = create_value(powf(a->data, s), MathOperation::POW);
    Value* out = get_value(ho);
    out->exponent = s;
    out->input.push_back(ha);
    return ho;
}

ValueHandle operator/(ValueHandle ha, ValueHandle hb)
{
    return ha * pow(hb, -1.f);
}

ValueHandle operator/(ValueHandle ha, float s)
{
    return ha * powf(s, -1.f);
}

ValueHandle operator/(float s, ValueHandle ha)
{
    return s * pow(ha, -1.f);
}

ValueHandle exp(ValueHandle ha)
{
    Value* a = get_value(ha);
    ValueHandle ho = create_value(expf(a->data), MathOperation::EXP);
    Value* out = get_value(ho);
    out->input.push_back(ha);
    return ho;
}

ValueHandle tanh(ValueHandle ha)
{
    Value* a = get_value(ha);
    ValueHandle ho = create_value(tanhf(a->data), MathOperation::TANH);
    Value* out = get_value(ho);
    out->input.push_back(ha);
    return ho;
}

#endif