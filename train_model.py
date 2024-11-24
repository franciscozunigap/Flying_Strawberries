from ultralytics import YOLO, checks, hub
checks()

hub.login('fbf921bfc039bf0941a80291c3ecea5218b6dee2d3')

model = YOLO('https://hub.ultralytics.com/models/74nqjy40zrKrUPnxNiQF')
results = model.train()