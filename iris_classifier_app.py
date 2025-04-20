
import gradio as gr

label_map = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}

def predict_iris(sepal_length, sepal_width, petal_length, petal_width):
    input_data = scaler.transform([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = knn.predict(input_data)[0]
    return label_map[prediction]

interface = gr.Interface(
    fn=predict_iris,
    inputs=[
        gr.Number(label="Sepal Length"),
        gr.Number(label="Sepal Width"),
        gr.Number(label="Petal Length"),
        gr.Number(label="Petal Width")
    ],
    outputs=gr.Text(label="Predicted Variety"),
    title="🌸 Iris Flower Classifier",
    description="Enter the flower's measurements to predict its variety (Setosa, Versicolor, Virginica)."
)

interface.launch()
