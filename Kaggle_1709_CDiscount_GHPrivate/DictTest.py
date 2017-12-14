

dict = { "a": [], "b": [], "c": [], "d": []}
while True:
    items = dict.items()
    for idx, e in enumerate(items):
        del items[0]
    if len(dict.items()) == 0:
        break
    for key in dict:
        print(key)
        del dict[key]
print("Done")
