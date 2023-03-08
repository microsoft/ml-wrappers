# Object Detection Scenario Documentation

ML-Wrappers supports model wrapping of Pytorch object detection methods. We convert the model to a function by either using the predict_proba function, or, if it is not available, the predict function. 

## Schema 
For each image in the dataset, the model is used to generate predictions. Then, the predictions are filtered
using non maximal suppression (based on the iuo threshold parameter). 

The predictions is a list of Pytorch tensors. Each tensor is composed of the labels, boxes (bounding boxes), scores. Example:

```
detections = [{'boxes': tensor([[ 97.0986, 170.7908, 241.4255, 516.5880]], grad_fn=<StackBackward0>), 'labels': tensor([2]), 'scores': tensor([0.9905], grad_fn=<IndexBackward0>)}]

predict_output = [[[2.0, 97.09860229492188, 170.7908172607422, 241.425537109375, 516.5879516601562, 0.9904877543449402]]]
```

## Limitations
This wrapper functionality only supports Pytorch machine learning models. 
