import 'dart:typed_data';

import 'package:tflite_flutter/tflite_flutter.dart';

class Classifier {
  Interpreter? _interpreter;
  static const List<String> labels = ["A", "B", "C", "Peace", "ILY"];

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
    var input = [Float32List.fromList(handLandmarkers)];
    var output = List<double>.filled(26, 0).reshape([1, 26]);

    try {
      _interpreter!.run(input, output);
    }
    catch (e) {
      return "Error: $e";
    }

    List<double> predictions = List<double>.from(output[0]);
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