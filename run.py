from syntax.model import Model, model_contexts

print("hi")

with Model("Nora") as nora:
    print("Hello")
    print(model_contexts)

print(model_contexts)

with Model("Chloe") as chloe:
    print("Hello")
    print(model_contexts)

print(model_contexts)

with nora:
    print(model_contexts)

print(model_contexts)
