import AppKit
import Foundation
import Vision

let arguments = CommandLine.arguments
guard arguments.count >= 2 else {
    fputs("usage: vision_ocr <image-path>\n", stderr)
    exit(64)
}

let imagePath = arguments[1]
let imageURL = URL(fileURLWithPath: imagePath)

guard let image = NSImage(contentsOf: imageURL) else {
    fputs("failed to load image\n", stderr)
    exit(66)
}

var proposedRect = NSRect(origin: .zero, size: image.size)
guard let cgImage = image.cgImage(forProposedRect: &proposedRect, context: nil, hints: nil) else {
    fputs("failed to decode image\n", stderr)
    exit(65)
}

let request = VNRecognizeTextRequest()
request.recognitionLevel = .accurate
request.usesLanguageCorrection = false
request.minimumTextHeight = 0.02

let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
do {
    try handler.perform([request])
} catch {
    fputs("vision request failed: \(error)\n", stderr)
    exit(70)
}

let observations = request.results ?? []
var payload: [[String: Any]] = []

for observation in observations {
    guard let candidate = observation.topCandidates(1).first else {
        continue
    }
    let box = observation.boundingBox
    payload.append([
        "text": candidate.string,
        "confidence": candidate.confidence,
        "min_x": box.minX,
        "min_y": box.minY,
        "width": box.width,
        "height": box.height,
    ])
}

guard let jsonData = try? JSONSerialization.data(withJSONObject: payload, options: []) else {
    fputs("failed to encode OCR output\n", stderr)
    exit(70)
}

FileHandle.standardOutput.write(jsonData)
