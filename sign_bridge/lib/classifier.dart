import 'dart:typed_data';

import 'package:tflite_flutter/tflite_flutter.dart';

class Classifier {
  Interpreter? _interpreter;
  static const List<String> labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K"];

  Future<void> loadModel() async {
    try {
      _interpreter = await Interpreter.fromAsset('assets/model.tflite');
      print("Model loaded.");
    }
    catch (e) {
      print("Error loading.");
    }
  }

  String predict(List<double> handLandmarkers) {
    if (_interpreter == null) return "Model not loaded";

    if (handLandmarkers.length != 63) {
      return "Invalid";
    }

    var input = [Float32List.fromList(handLandmarkers)];
    var output = List<double>.filled(labels.length, 0).reshape([1, labels.length]);

    try {
      _interpreter!.run(input, output);
    }
    catch (e) {
      print("TFLite Interpreter Error: $e");
      return "Error: $e";
    }

    if (output.isEmpty || output[0].isEmpty) return "Empty Output";

    List<double> predictions = (output[0] as List).map((e) => (e as num).toDouble()).toList();
    
    int bestIndex = 0;
    double maxConfidence = -1;

    for (int i = 0; i < predictions.length; i++) {
      if (predictions[i] > maxConfidence) {
        maxConfidence = predictions[i];
        bestIndex = i;
      }
    }

    if (maxConfidence < 0.7) {
      return "Detecting...";
    }

    return _indexToLabel(bestIndex);
  }

  String _indexToLabel(int index) {
    if (index < labels.length) {
      return labels[index];
    }
    return "Unknown Sign";
  }

  void dispose() {
    _interpreter?.close();
  }
}