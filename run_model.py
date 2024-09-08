from symgraph.model import Model, model_contexts

print(model_contexts)
with Model("root") as root:
    print(model_contexts)

    with Model("first") as first:
        print(model_contexts.active_model)
        x = 1

    with Model("second") as second:
        y = 1

    
