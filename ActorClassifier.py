import turicreate as turi

filepath = "actordataset/"

data = turi.image_analysis.load_images(filepath)

data["TonyLiangOrAerisLu"] = data["path"].apply(lambda path: "Liang" if "tonyliang" in path else "Lu")

data.save("liang_or_lu.sframe")

data.explore()

data = turi.SFrame("liang_or_lu.sframe")

train_data, test_data = data.random_split(0.8)

model = turi.image_classifier.create(train_data, target="TonyLiangOrAerisLu", max_iterations=30)

predictions = model.predict(test_data)

metrics = model.evaluate(test_data)

print(metrics["accuracy"])

model.save("qishengActor.model")


model.export_coreml("qishengActor.mlmodel")
