import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:hand_landmarker/hand_landmarker.dart';
import '../classifier.dart';

class CamPage extends StatefulWidget {
  const CamPage({super.key});

  @override
  State<CamPage> createState() => _CamPageState();
}

class _CamPageState extends State<CamPage> {
  CameraController? _cameraController;
  HandLandmarkerPlugin? _plugin;
  final Classifier _classifier = Classifier();
  
  bool _isDetecting = false;
  bool _isInitialized = false;
  String _prediction = "Point your hand at the camera";

  @override
  void initState() {
    super.initState();
    _setupPipeline();
  }

  Future<void> _setupPipeline() async {
    await _classifier.loadModel();
    _plugin = HandLandmarkerPlugin.create(
      numHands: 1,
      minHandDetectionConfidence: 0.7,
      delegate: HandLandmarkerDelegate.gpu,
    );
    
    final camera = await availableCameras();
    _cameraController = CameraController(
      camera.firstWhere((c) => c.lensDirection == CameraLensDirection.front), 
      ResolutionPreset.medium,
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.yuv420,
    );

    await _cameraController!.initialize();

    _cameraController!.startImageStream((image) {
      if (!_isDetecting) {
        _isDetecting = true;
        _runInference(image);
      }
    });

    if (mounted) setState(() => _isInitialized = true);
  }

 void _runInference(CameraImage image) {
  try {
    final hands = _plugin!.detect(
      image,
      _cameraController!.description.sensorOrientation,
    );

    if (hands.isEmpty) {
      if (mounted) setState(() => _prediction = "Show your hand");
      return; 
    }

    final hand = hands.first;

    if (hand.landmarks.isEmpty) {
      return;
    }

    List<double> landmarkData = [];
    for (var point in hand.landmarks) {
      landmarkData.add(1.0 - point.x);
      landmarkData.add(point.y);
      landmarkData.add(point.z);
    }

    if (landmarkData.length == 63) {
      print("Landmark 0 (Wrist): ${landmarkData[0]}, ${landmarkData[1]}");
      final result = _classifier.predict(landmarkData);
      if (mounted) {
        setState(() => _prediction = result);
      }
    } else {
      debugPrint("landmarkData length ${landmarkData.length}");
    }

  } catch (e, stacktrace) {
    debugPrint("CRITICAL ERROR: $e");
    debugPrint("STACKTRACE: $stacktrace");
  } finally {
    Future.delayed(const Duration(milliseconds: 100), () {
      _isDetecting = false;
    });
  }
}
  @override
  void dispose() {
    _cameraController?.dispose();
    _plugin?.dispose();
    _classifier.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (!_isInitialized || _cameraController == null) {
      return const Scaffold(backgroundColor: Colors.black, body: Center(child: CircularProgressIndicator()));
    }
    return Scaffold(
      backgroundColor: Colors.black,
      body: Stack(
        children: [
          Center(child: CameraPreview(_cameraController!)),
          Align(
            alignment: Alignment.centerRight,
            child: Container(
              margin: const EdgeInsets.only(bottom: 40),
              padding: const EdgeInsets.all(20),
              decoration: BoxDecoration(
                color: Colors.black,
                borderRadius: BorderRadius.circular(16),
                border: Border.all(color: Colors.greenAccent),
              ),
              child: RotatedBox(
                quarterTurns: 1,
                child: Text(
                  _prediction,
                  style: const TextStyle(color: Colors.greenAccent, fontSize: 32, fontWeight: FontWeight.bold),
                ),
              ),
            ),
          )
        ],
      )
    );
  }
}