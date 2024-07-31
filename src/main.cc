#include "engine.h"
#include "nn.h"

#include <stdio.h>

#include <graphviz/gvc.h>

void draw_dot(ValueHandle root, char graph_name[32])
{
    GVC_t* gvc = gvContext();
    Agraph_t* graph = agopen(graph_name, Agdirected, NULL);

    char randir_attr[8] = "rankdir";
    agattr(graph, AGRAPH, randir_attr, "LR");

    char shape_attr[6] = "shape";
    Agsym_t* node_shape_sym = agattr(graph, AGNODE, shape_attr, "record");
    assert(node_shape_sym != NULL);

    char label_attr[6] = "label";
    std::vector<ValueHandle> bfs;
    std::unordered_set<int> visited;
    bfs.push_back(root);
    while(bfs.size() > 0)
    {
        ValueHandle h = bfs.back();
        bfs.pop_back();
        if (visited.contains(h.idx)) continue;
        Value* value = get_value(h);

        char node_name[8];
        sprintf(node_name, "%d", h.idx);
        Agnode_t* node = agnode(graph, node_name, true);

        char label[32];
        sprintf(label, "{ %d | data %f }", h.idx, value->data);
        agset(node, label_attr, label);

        Agnode_t* op_node = NULL;
        switch(value->op)
        {
        case MathOperation::ADD:
            {
                char op_node_name[16];
                sprintf(op_node_name, "%d_OP_ADD", h.idx);
                op_node = agnode(graph, op_node_name, true);
                agset(op_node, label_attr, "+");
                agxset(op_node, node_shape_sym, "ellipse");
            }
            break;
        case MathOperation::MULTIPLE:
            {
                char op_node_name[16];
                sprintf(op_node_name, "%d_OP_MUL", h.idx);
                op_node = agnode(graph, op_node_name, true);
                agset(op_node, label_attr, "*");
                agxset(op_node, node_shape_sym, "ellipse");
            }
            break;
        case MathOperation::POW:
            {
                char op_node_name[16];
                sprintf(op_node_name, "%d_OP_POW", h.idx);
                op_node = agnode(graph, op_node_name, true);
                sprintf(label, "** %f", value->exponent);
                agset(op_node, label_attr, label);
                agxset(op_node, node_shape_sym, "ellipse");
            }
            break;
        case MathOperation::EXP:
            {
                char op_node_name[16];
                sprintf(op_node_name, "%d_OP_EXP", h.idx);
                op_node = agnode(graph, op_node_name, true);
                agset(op_node, label_attr, "exp");
                agxset(op_node, node_shape_sym, "ellipse");
            }
            break;
        case MathOperation::TANH:
            {
                char op_node_name[16];
                sprintf(op_node_name, "%d_OP_TANH", h.idx);
                op_node = agnode(graph, op_node_name, true);
                agset(op_node, label_attr, "tanh");
                agxset(op_node, node_shape_sym, "ellipse");
            }
            break;
        case MathOperation::RELU:
            {
                char op_node_name[16];
                sprintf(op_node_name, "%d_OP_RELU", h.idx);
                op_node = agnode(graph, op_node_name, true);
                agset(op_node, label_attr, "relu");
                agxset(op_node, node_shape_sym, "ellipse");
            }
            break;
        default:
            break;
        }

        if (op_node != NULL)
        {
            agedge(graph, op_node, node, NULL, 1);
        }

        for (int i = 0; i < value->input.size(); i++)
        {
            assert(op_node != NULL);

            ValueHandle ch = value->input[i];
            Value* child_value = get_value(ch);

            char child_node_name[8];
            sprintf(child_node_name, "%d", ch.idx);
            Agnode_t* child_node = agnode(graph, child_node_name, true);

            sprintf(label, "{ %d | data %f }", ch.idx, child_value->data);
            agset(child_node, label_attr, label);

            agedge(graph, child_node, op_node, NULL, 1);

            bfs.push_back(ch);
        }

        visited.insert(h.idx);
    }

    gvLayout(gvc, graph, "dot");
    //agwrite(graph, stdout);
    char filename[64];
    sprintf(filename, "./%s.svg", graph_name);
    gvRenderFilename(gvc, graph, "svg", filename);

    gvFreeLayout(gvc, graph);

    agclose(graph);

    gvFreeContext(gvc);
}

void engine_test_1()
{
    TEMP_VALUE_POOL_START;

    fprintf(stdout, "engine_test_1: \n");

    ValueHandle a = create_value(1.f);
    ValueHandle b = create_value(2.f);
    ValueHandle c = b * a + 1.f;
    ValueHandle d = b * c;
    ValueHandle e = pow(d, 2.f);
    ValueHandle f = e / 2.f;
    ValueHandle g = f - 16.f;
    ValueHandle h = exp(g);

    backward(h);

    fprintf(stdout, "%s, data: %f, gradient: %f\n", "a", get_value(a)->data, get_value(a)->gradient);
    fprintf(stdout, "%s, data: %f, gradient: %f\n", "b", get_value(b)->data, get_value(b)->gradient);
    fprintf(stdout, "%s, data: %f, gradient: %f\n", "c", get_value(c)->data, get_value(c)->gradient);
    fprintf(stdout, "%s, data: %f, gradient: %f\n", "d", get_value(d)->data, get_value(d)->gradient);
    fprintf(stdout, "%s, data: %f, gradient: %f\n", "e", get_value(e)->data, get_value(e)->gradient);
    fprintf(stdout, "%s, data: %f, gradient: %f\n", "f", get_value(f)->data, get_value(f)->gradient);
    fprintf(stdout, "%s, data: %f, gradient: %f\n", "g", get_value(g)->data, get_value(g)->gradient);
    fprintf(stdout, "%s, data: %f, gradient: %f\n", "h", get_value(h)->data, get_value(h)->gradient);

    char graph_name[32] = "engine_test_1";
    draw_dot(h, graph_name);

    TEMP_VALUE_POOL_END;
}

void engine_test_2()
{
    TEMP_VALUE_POOL_START;

    fprintf(stdout, "engine_test_2: \n");

    ValueHandle x1 = create_value(2.f);
    ValueHandle x2 = create_value(0.f);
    ValueHandle w1 = create_value(-3.f);
    ValueHandle w2 = create_value(1.f);
    ValueHandle b = create_value(6.88137358702f);
    ValueHandle x1w1 = x1 * w1;
    ValueHandle x2w2 = x2 * w2;
    ValueHandle x1w1x2w2 = x1w1 + x2w2;
    ValueHandle n = x1w1x2w2 + b;
    ValueHandle e = exp(2 * n);
    ValueHandle o = (e - 1) / (e + 1);

    backward(o);

    fprintf(stdout, "%s, data: %f, gradient: %f\n", "x1", get_value(x1)->data, get_value(x1)->gradient);
    fprintf(stdout, "%s, data: %f, gradient: %f\n", "x2", get_value(x2)->data, get_value(x2)->gradient);
    fprintf(stdout, "%s, data: %f, gradient: %f\n", "w1", get_value(w1)->data, get_value(w1)->gradient);
    fprintf(stdout, "%s, data: %f, gradient: %f\n", "w2", get_value(w2)->data, get_value(w2)->gradient);
    fprintf(stdout, "%s, data: %f, gradient: %f\n", "b", get_value(b)->data, get_value(b)->gradient);
    fprintf(stdout, "%s, data: %f, gradient: %f\n", "x1w1", get_value(x1w1)->data, get_value(x1w1)->gradient);
    fprintf(stdout, "%s, data: %f, gradient: %f\n", "x2w2", get_value(x2w2)->data, get_value(x2w2)->gradient);
    fprintf(stdout, "%s, data: %f, gradient: %f\n", "x1w1x2w2", get_value(x1w1x2w2)->data, get_value(x1w1x2w2)->gradient);
    fprintf(stdout, "%s, data: %f, gradient: %f\n", "n", get_value(n)->data, get_value(n)->gradient);
    fprintf(stdout, "%s, data: %f, gradient: %f\n", "e", get_value(e)->data, get_value(e)->gradient);
    fprintf(stdout, "%s, data: %f, gradient: %f\n", "o", get_value(o)->data, get_value(o)->gradient);

    char graph_name[32] = "engine_test_2";
    draw_dot(o, graph_name);

    TEMP_VALUE_POOL_END;
}

void engine_test_3()
{
    TEMP_VALUE_POOL_START;

    fprintf(stdout, "engine_test_3: \n");

    ValueHandle x1 = create_value(2.f);
    ValueHandle x2 = create_value(0.f);
    ValueHandle w1 = create_value(-3.f);
    ValueHandle w2 = create_value(1.f);
    ValueHandle b = create_value(6.88137358702f);
    ValueHandle x1w1 = x1 * w1;
    ValueHandle x2w2 = x2 * w2;
    ValueHandle x1w1x2w2 = x1w1 + x2w2;
    ValueHandle n = x1w1x2w2 + b;
    ValueHandle o = tanh(n);

    backward(o);

    fprintf(stdout, "%s, data: %f, gradient: %f\n", "x1", get_value(x1)->data, get_value(x1)->gradient);
    fprintf(stdout, "%s, data: %f, gradient: %f\n", "x2", get_value(x2)->data, get_value(x2)->gradient);
    fprintf(stdout, "%s, data: %f, gradient: %f\n", "w1", get_value(w1)->data, get_value(w1)->gradient);
    fprintf(stdout, "%s, data: %f, gradient: %f\n", "w2", get_value(w2)->data, get_value(w2)->gradient);
    fprintf(stdout, "%s, data: %f, gradient: %f\n", "b", get_value(b)->data, get_value(b)->gradient);
    fprintf(stdout, "%s, data: %f, gradient: %f\n", "x1w1", get_value(x1w1)->data, get_value(x1w1)->gradient);
    fprintf(stdout, "%s, data: %f, gradient: %f\n", "x2w2", get_value(x2w2)->data, get_value(x2w2)->gradient);
    fprintf(stdout, "%s, data: %f, gradient: %f\n", "x1w1x2w2", get_value(x1w1x2w2)->data, get_value(x1w1x2w2)->gradient);
    fprintf(stdout, "%s, data: %f, gradient: %f\n", "n", get_value(n)->data, get_value(n)->gradient);
    fprintf(stdout, "%s, data: %f, gradient: %f\n", "o", get_value(o)->data, get_value(o)->gradient);

    char graph_name[32] = "engine_test_3";
    draw_dot(o, graph_name);

    TEMP_VALUE_POOL_END;
}

void mlp_test()
{
    TEMP_VALUE_POOL_START;

    fprintf(stdout, "mlp_test: \n");

    MLP mlp;
    std::vector<int> layer = {4, 4, 1};
    mlp_init(mlp, 3, layer);

    std::vector<ValueHandle> parameters = mlp_parameters(mlp);
    for (int i = 0; i < parameters.size(); i++)
    {
        Value* parameter = get_value(parameters[i]);
        fprintf(stdout, "parameter %d, data: %f\n", i, parameter->data);
    }
    fprintf(stdout, "\n");

    std::vector<ValueHandle> input_data_0;
    input_data_0.push_back(create_value(2.f));
    input_data_0.push_back(create_value(3.f));
    input_data_0.push_back(create_value(-1.f));
    std::vector<ValueHandle> input_data_1;
    input_data_1.push_back(create_value(3.f));
    input_data_1.push_back(create_value(-1.f));
    input_data_1.push_back(create_value(0.5f));
    std::vector<ValueHandle> input_data_2;
    input_data_2.push_back(create_value(0.5f));
    input_data_2.push_back(create_value(1.f));
    input_data_2.push_back(create_value(1.f));
    std::vector<ValueHandle> input_data_3;
    input_data_3.push_back(create_value(1.f));
    input_data_3.push_back(create_value(1.f));
    input_data_3.push_back(create_value(-1.f));

    std::vector<ValueHandle> expect_set;
    expect_set.push_back(create_value(1.f));
    expect_set.push_back(create_value(-1.f));
    expect_set.push_back(create_value(-1.f));
    expect_set.push_back(create_value(1.f));

    for (int g = 0; g < 50; g++)
    {
        TEMP_VALUE_POOL_START;

        std::vector<ValueHandle> prediction_set;

        std::vector<ValueHandle> prediction_0 = mlp_forward(mlp, input_data_0);
        assert(prediction_0.size() == 1);
        prediction_set.push_back(prediction_0.back());

        std::vector<ValueHandle> prediction_1 = mlp_forward(mlp, input_data_1);
        assert(prediction_1.size() == 1);
        prediction_set.push_back(prediction_1.back());

        std::vector<ValueHandle> prediction_2 = mlp_forward(mlp, input_data_2);
        assert(prediction_2.size() == 1);
        prediction_set.push_back(prediction_2.back());

        std::vector<ValueHandle> prediction_3 = mlp_forward(mlp, input_data_3);
        assert(prediction_3.size() == 1);
        prediction_set.push_back(prediction_3.back());

        ValueHandle loss = mean_squared_error(expect_set, prediction_set);
        fprintf(stdout, "iteration %d, loss: %.5f\n", g, get_value(loss)->data);

        mlp_zero_grad(mlp);
        mlp_backward(mlp, loss, 0.05f);

        char graph_name[32];
        sprintf(graph_name, "./mlp_test_%d", g);
        draw_dot(loss, graph_name);

        TEMP_VALUE_POOL_END;
    }

    fprintf(stdout, "\n");
    for (int i = 0; i < parameters.size(); i++)
    {
        Value* parameter = get_value(parameters[i]);
        fprintf(stdout, "parameter %d, data: %f\n", i, parameter->data);
    }

    std::vector<ValueHandle> prediction_set;

    std::vector<ValueHandle> prediction_0 = mlp_forward(mlp, input_data_0);
    assert(prediction_0.size() == 1);
    prediction_set.push_back(prediction_0.back());

    std::vector<ValueHandle> prediction_1 = mlp_forward(mlp, input_data_1);
    assert(prediction_1.size() == 1);
    prediction_set.push_back(prediction_1.back());

    std::vector<ValueHandle> prediction_2 = mlp_forward(mlp, input_data_2);
    assert(prediction_2.size() == 1);
    prediction_set.push_back(prediction_2.back());

    std::vector<ValueHandle> prediction_3 = mlp_forward(mlp, input_data_3);
    assert(prediction_3.size() == 1);
    prediction_set.push_back(prediction_3.back());

    assert(prediction_set.size() == expect_set.size());
    fprintf(stdout, "\n");
    for (int i = 0; i < prediction_set.size(); i++)
    {
        fprintf(stdout, "prediction: %9f, expect: %9f\n", 
            get_value(prediction_set[i])->data, get_value(expect_set[i])->data);
    }

    TEMP_VALUE_POOL_END;
}

int main()
{
    engine_test_1();
    fprintf(stdout, "\n\n");

    engine_test_2();
    fprintf(stdout, "\n\n");

    engine_test_3();
    fprintf(stdout, "\n\n");

    mlp_test();

    return 0;
}