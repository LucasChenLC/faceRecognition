import json

colors = [(0, 191, 255), (255, 48, 48), (0, 255, 0), (255, 255, 0), (0, 206, 209), (107, 142, 34), (139, 26, 26)]

with open('json/colors.json', 'w') as file:
    json.dump(colors, file)
