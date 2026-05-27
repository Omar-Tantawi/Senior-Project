// Web implementation — triggers a browser "Save As" download.
// ignore: avoid_web_libraries_in_flutter
import 'dart:html' as html;

Future<void> downloadFile(List<int> bytes, String filename) async {
  final blob   = html.Blob([bytes]);
  final url    = html.Url.createObjectUrlFromBlob(blob);
  final anchor = html.AnchorElement(href: url)
    ..setAttribute('download', filename)
    ..click();
  html.Url.revokeObjectUrl(url);
}
